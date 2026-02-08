"""
Backtester — Historical replay, strategy evaluation, and performance tearsheets.

Phase 3 of the prop desk platform build. Replays historical market data through
the DependencyGraph, executes strategies via the TradingEngine, and generates
comprehensive performance analytics.

Classes:
    HistoricalDataLoader — Fetch and cache historical OHLCV data
    BacktestContext      — Simulated market environment for a single day
    BacktestStrategy     — Base class + 4 strategy implementations
    EquityCurve          — Daily portfolio value tracking
    Tearsheet            — Performance metrics and analytics
    Backtester           — Orchestrator for historical replay
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── HistoricalDataLoader ───────────────────────────────────────────────

class HistoricalDataLoader:
    """
    Fetch historical OHLCV data from yfinance and cache in MnTable/Barbara.
    """

    def __init__(self, barbara=None):
        self.barbara = barbara
        self._cache = {}  # ticker -> list of bar dicts

    def load(self, tickers, start_date, end_date):
        """
        Load historical data for given tickers and date range.

        Parameters
        ----------
        tickers    : list[str]
        start_date : str — "YYYY-MM-DD"
        end_date   : str — "YYYY-MM-DD"

        Returns
        -------
        dict[str, list[dict]] — ticker -> list of {date, open, high, low, close, volume}
        """
        import yfinance as yf

        result = {}
        for ticker in tickers:
            cache_key = f"{ticker}_{start_date}_{end_date}"
            if cache_key in self._cache:
                result[ticker] = self._cache[cache_key]
                continue

            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df is None or df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                bars = []
                for idx, row in df.iterrows():
                    bars.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row.get("Volume", 0)),
                    })

                result[ticker] = bars
                self._cache[cache_key] = bars

                if self.barbara:
                    bk = f"/Backtest/data/{ticker}/{start_date}_{end_date}"
                    self.barbara[bk] = bars

            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")

        return result

    def load_from_arrays(self, ticker_data):
        """
        Load data directly from dict of arrays (for testing without yfinance).

        Parameters
        ----------
        ticker_data : dict[str, list[dict]] — pre-built bar data
        """
        for ticker, bars in ticker_data.items():
            self._cache[f"{ticker}_manual"] = bars

    def get_price(self, ticker, date):
        """Get closing price for a specific ticker and date."""
        for key, bars in self._cache.items():
            if not key.startswith(ticker):
                continue
            for bar in bars:
                if bar["date"] == date:
                    return bar["close"]
        return None

    def get_bar(self, ticker, date):
        """Get full OHLCV bar for a specific ticker and date."""
        for key, bars in self._cache.items():
            if not key.startswith(ticker):
                continue
            for bar in bars:
                if bar["date"] == date:
                    return bar
        return None

    def get_dates(self):
        """Get sorted list of all unique dates across all tickers."""
        dates = set()
        for bars in self._cache.values():
            for bar in bars:
                dates.add(bar["date"])
        return sorted(dates)


# ── BacktestContext ─────────────────────────────────────────────────────

class BacktestContext:
    """
    Simulated market environment for a single trading day.

    Sets MarketData nodes to historical prices and triggers graph recalculation.
    """

    def __init__(self, graph, market_data_nodes, date, prices):
        """
        Parameters
        ----------
        graph             : DependencyGraph
        market_data_nodes : dict[str, MarketData] — ticker -> node
        date              : str — current date
        prices            : dict[str, float] — ticker -> close price
        """
        self.graph = graph
        self.market_data_nodes = market_data_nodes
        self.date = date
        self.prices = prices

    def apply(self):
        """Set all market data nodes to historical prices and recalculate."""
        for ticker, price in self.prices.items():
            node = self.market_data_nodes.get(ticker)
            if node and price > 0:
                node.set_price(price)
                self.graph.recalculate(node)


# ── BacktestStrategy ───────────────────────────────────────────────────

class BacktestStrategy:
    """Base class for backtest strategies."""

    def on_start(self, context, engine):
        """Called once before the backtest begins."""
        pass

    def on_bar(self, context, engine, date):
        """
        Called for each trading day.

        Parameters
        ----------
        context : BacktestContext
        engine  : TradingEngine
        date    : str — current date
        """
        raise NotImplementedError

    def on_end(self, context, engine):
        """Called once after the backtest ends."""
        pass


class MomentumBacktest(BacktestStrategy):
    """
    Buy top N by trailing return, sell bottom N.

    Position-aware: targets equal-weight allocation among top-N tickers,
    sizes orders relative to portfolio value, trades only the delta.
    """

    def __init__(self, tickers, lookback=20, top_n=3, trade_qty=100):
        self.tickers = tickers
        self.lookback = lookback
        self.top_n = min(top_n, len(tickers))
        self.trade_qty = trade_qty
        self._history = {t: [] for t in tickers}

    def on_bar(self, context, engine, date):
        # Track prices
        for t in self.tickers:
            price = context.prices.get(t, 0)
            if price > 0:
                self._history[t].append(price)

        # Compute trailing returns for tickers with enough history
        returns = {}
        for t in self.tickers:
            hist = self._history[t]
            if len(hist) < self.lookback:
                continue
            ret = (hist[-1] / hist[-self.lookback]) - 1.0
            returns[t] = ret

        if len(returns) < self.top_n:
            return

        # Rank
        sorted_tickers = sorted(returns.keys(), key=lambda x: returns[x], reverse=True)
        top = set(sorted_tickers[:self.top_n])

        # Get current positions and portfolio value
        positions = getattr(engine, '_bt_positions', {})
        capital = max(engine._backtest_capital, 1.0)

        # Target: equal weight in top-N, flat in everything else
        target_weight = 0.90 / self.top_n  # 90% invested, 10% cash buffer

        for t in self.tickers:
            price = context.prices.get(t, 0)
            if price <= 0:
                continue
            current_qty = positions.get(t, 0)
            if t in top:
                target_qty = int(capital * target_weight / price)
            else:
                target_qty = 0

            delta = target_qty - current_qty
            if abs(delta) < 1:
                continue

            side = "BUY" if delta > 0 else "SELL"
            engine.submit_order(
                trader=engine._backtest_trader,
                instrument_name=t,
                side=side,
                quantity=abs(delta),
                current_price=price,
                book_value=capital,
            )


class VolArbBacktest(BacktestStrategy):
    """
    Sell when realized vol < implied vol, buy back when it reverts.

    Position-aware: limits position size to a fraction of portfolio,
    tracks whether already positioned.
    """

    def __init__(self, tickers, rv_window=20, trade_qty=10):
        self.tickers = tickers
        self.rv_window = rv_window
        self.trade_qty = trade_qty
        self._history = {t: [] for t in tickers}

    def on_bar(self, context, engine, date):
        for t in self.tickers:
            price = context.prices.get(t, 0)
            if price > 0:
                self._history[t].append(price)

        positions = getattr(engine, '_bt_positions', {})
        capital = max(engine._backtest_capital, 1.0)
        max_position_pct = 0.15  # max 15% of capital per ticker

        for t in self.tickers:
            hist = self._history[t]
            if len(hist) < self.rv_window + 1:
                continue

            closes = np.array(hist[-self.rv_window - 1:])
            log_ret = np.diff(np.log(closes))
            rv = np.std(log_ret) * np.sqrt(252)

            iv = 0.25
            price = context.prices.get(t, 0)
            if price <= 0:
                continue

            current_qty = positions.get(t, 0)
            current_exposure = abs(current_qty * price)

            if rv < iv * 0.85 and current_exposure < capital * max_position_pct:
                # Sell vol — limit size
                qty = min(self.trade_qty, int(capital * max_position_pct / price))
                if qty > 0:
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t, side="SELL",
                        quantity=qty,
                        current_price=price,
                        book_value=capital,
                    )
            elif rv > iv * 1.15 and current_qty < 0:
                # Vol reverted up — buy back short
                engine.submit_order(
                    trader=engine._backtest_trader,
                    instrument_name=t, side="BUY",
                    quantity=abs(current_qty),
                    current_price=price,
                    book_value=capital,
                )


class StatArbBacktest(BacktestStrategy):
    """
    Z-score mean reversion on pairs.

    Position-aware: enters when |z| > threshold, exits when |z| < exit_threshold,
    sizes each leg to a fraction of capital.
    """

    def __init__(self, pairs, lookback=60, z_threshold=2.0, trade_qty=50):
        self.pairs = pairs
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.exit_threshold = 0.5
        self.trade_qty = trade_qty
        self._history = {}

    def on_bar(self, context, engine, date):
        for t_a, t_b in self.pairs:
            for t in [t_a, t_b]:
                if t not in self._history:
                    self._history[t] = []
                price = context.prices.get(t, 0)
                if price > 0:
                    self._history[t].append(price)

        positions = getattr(engine, '_bt_positions', {})
        capital = max(engine._backtest_capital, 1.0)
        leg_pct = 0.20  # 20% of capital per leg

        for t_a, t_b in self.pairs:
            hist_a = self._history.get(t_a, [])
            hist_b = self._history.get(t_b, [])
            min_len = min(len(hist_a), len(hist_b))

            if min_len < self.lookback:
                continue

            closes_a = np.array(hist_a[-self.lookback:])
            closes_b = np.array(hist_b[-self.lookback:])

            ratio = closes_a / np.maximum(closes_b, 1e-10)
            z = (ratio[-1] - np.mean(ratio)) / max(np.std(ratio), 1e-10)

            pos_a = positions.get(t_a, 0)
            pos_b = positions.get(t_b, 0)
            in_trade = (pos_a != 0 or pos_b != 0)

            price_a = context.prices.get(t_a, 0)
            price_b = context.prices.get(t_b, 0)
            if price_a <= 0 or price_b <= 0:
                continue

            # Exit: z reverted
            if in_trade and abs(z) < self.exit_threshold:
                if pos_a != 0:
                    side = "SELL" if pos_a > 0 else "BUY"
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t_a, side=side,
                        quantity=abs(pos_a),
                        current_price=price_a, book_value=capital,
                    )
                if pos_b != 0:
                    side = "SELL" if pos_b > 0 else "BUY"
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t_b, side=side,
                        quantity=abs(pos_b),
                        current_price=price_b, book_value=capital,
                    )
            # Entry: z extreme, not already in trade
            elif not in_trade and z > self.z_threshold:
                qty_a = int(capital * leg_pct / price_a)
                qty_b = int(capital * leg_pct / price_b)
                if qty_a > 0 and qty_b > 0:
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t_a, side="SELL", quantity=qty_a,
                        current_price=price_a, book_value=capital,
                    )
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t_b, side="BUY", quantity=qty_b,
                        current_price=price_b, book_value=capital,
                    )
            elif not in_trade and z < -self.z_threshold:
                qty_a = int(capital * leg_pct / price_a)
                qty_b = int(capital * leg_pct / price_b)
                if qty_a > 0 and qty_b > 0:
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t_a, side="BUY", quantity=qty_a,
                        current_price=price_a, book_value=capital,
                    )
                    engine.submit_order(
                        trader=engine._backtest_trader,
                        instrument_name=t_b, side="SELL", quantity=qty_b,
                        current_price=price_b, book_value=capital,
                    )


class MacroBacktest(BacktestStrategy):
    """
    Trend following using MA crossover.

    Position-aware: targets long/short/flat based on trend direction,
    sizes to a fraction of capital, trades only the delta.
    """

    def __init__(self, tickers, trade_qty=100):
        self.tickers = tickers
        self.trade_qty = trade_qty
        self._history = {t: [] for t in tickers}

    def on_bar(self, context, engine, date):
        for t in self.tickers:
            price = context.prices.get(t, 0)
            if price > 0:
                self._history[t].append(price)

        positions = getattr(engine, '_bt_positions', {})
        capital = max(engine._backtest_capital, 1.0)
        per_ticker_pct = 0.80 / max(len(self.tickers), 1)

        for t in self.tickers:
            hist = self._history[t]
            if len(hist) < 20:
                continue

            ma5 = np.mean(hist[-5:])
            ma20 = np.mean(hist[-20:])
            price = hist[-1]
            if price <= 0:
                continue

            current_qty = positions.get(t, 0)

            if ma5 > ma20 * 1.01:
                target_qty = int(capital * per_ticker_pct / price)
            elif ma5 < ma20 * 0.99:
                target_qty = -int(capital * per_ticker_pct / price)
            else:
                target_qty = current_qty  # hold

            delta = target_qty - current_qty
            if abs(delta) < 1:
                continue

            side = "BUY" if delta > 0 else "SELL"
            engine.submit_order(
                trader=engine._backtest_trader,
                instrument_name=t, side=side,
                quantity=abs(delta),
                current_price=price,
                book_value=capital,
            )


# ── EquityCurve ────────────────────────────────────────────────────────

class EquityCurve:
    """
    Track daily portfolio value, returns, drawdowns.

    Stored as an MnTable with columns:
    date, portfolio_value, daily_return, cumulative_return, drawdown, high_water_mark
    """

    def __init__(self, initial_capital=1_000_000):
        from bank_python.mntable import Table

        self.initial_capital = initial_capital
        self._table = Table([
            ("date", str),
            ("portfolio_value", float),
            ("daily_return", float),
            ("cumulative_return", float),
            ("drawdown", float),
            ("high_water_mark", float),
        ], name="equity_curve")
        self._hwm = initial_capital
        self._prev_value = initial_capital

    def record(self, date, portfolio_value):
        """Record a daily equity curve point."""
        daily_ret = (portfolio_value - self._prev_value) / max(self._prev_value, 1.0)
        cum_ret = (portfolio_value - self.initial_capital) / self.initial_capital
        self._hwm = max(self._hwm, portfolio_value)
        dd = (portfolio_value - self._hwm) / max(self._hwm, 1.0)

        self._table.append({
            "date": date,
            "portfolio_value": float(portfolio_value),
            "daily_return": float(daily_ret),
            "cumulative_return": float(cum_ret),
            "drawdown": float(dd),
            "high_water_mark": float(self._hwm),
        })

        self._prev_value = portfolio_value

    @property
    def table(self):
        return self._table

    def to_arrays(self):
        """Convert to numpy arrays for analytics."""
        rows = list(self._table)
        if not rows:
            return {
                "dates": [],
                "values": np.array([]),
                "returns": np.array([]),
                "cum_returns": np.array([]),
                "drawdowns": np.array([]),
            }
        return {
            "dates": [r["date"] for r in rows],
            "values": np.array([r["portfolio_value"] for r in rows]),
            "returns": np.array([r["daily_return"] for r in rows]),
            "cum_returns": np.array([r["cumulative_return"] for r in rows]),
            "drawdowns": np.array([r["drawdown"] for r in rows]),
        }


# ── Tearsheet ──────────────────────────────────────────────────────────

class Tearsheet:
    """
    Compute comprehensive performance metrics from equity curve + trade history.

    All computations use numpy only (no scipy/pandas dependencies).
    """

    def __init__(self, equity_curve, trade_history=None, risk_free_rate=0.05):
        self.equity_curve = equity_curve
        self.trade_history = trade_history or []
        self.rf = risk_free_rate
        self._metrics = {}

    def compute(self):
        """Compute all metrics. Returns dict of metrics."""
        data = self.equity_curve.to_arrays()
        returns = data["returns"]
        values = data["values"]
        drawdowns = data["drawdowns"]

        if len(returns) == 0:
            self._metrics = {"error": "No data"}
            return self._metrics

        n_days = len(returns)

        # Total and annualized return
        total_return = (values[-1] / self.equity_curve.initial_capital) - 1.0
        if total_return > -1.0:
            ann_return = (1 + total_return) ** (252.0 / max(n_days, 1)) - 1.0
        else:
            ann_return = -1.0  # total loss

        # Annualized volatility
        ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252)) if n_days > 1 else 0.0

        # Sharpe ratio
        sharpe = (ann_return - self.rf) / max(ann_vol, 1e-10)

        # Sortino ratio (downside deviation)
        neg_returns = returns[returns < 0]
        downside_std = float(np.std(neg_returns, ddof=1) * np.sqrt(252)) if len(neg_returns) > 1 else 1e-10
        sortino = (ann_return - self.rf) / max(downside_std, 1e-10)

        # Max drawdown
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Max drawdown duration (in days)
        dd_duration = 0
        curr_dd_start = 0
        max_dd_duration = 0
        in_dd = False
        for i, dd in enumerate(drawdowns):
            if dd < -1e-6:
                if not in_dd:
                    curr_dd_start = i
                    in_dd = True
            else:
                if in_dd:
                    dd_len = i - curr_dd_start
                    max_dd_duration = max(max_dd_duration, dd_len)
                    in_dd = False
        if in_dd:
            max_dd_duration = max(max_dd_duration, len(drawdowns) - curr_dd_start)

        # Win rate (daily)
        wins = np.sum(returns > 0)
        losses = np.sum(returns < 0)
        win_rate = float(wins / max(wins + losses, 1))

        # Profit factor
        gross_profit = float(np.sum(returns[returns > 0])) if wins > 0 else 0.0
        gross_loss = float(np.abs(np.sum(returns[returns < 0]))) if losses > 0 else 1e-10
        profit_factor = gross_profit / max(gross_loss, 1e-10)

        # Calmar ratio
        calmar = ann_return / max(abs(max_dd), 1e-10)

        # Best/worst day
        best_day = float(np.max(returns)) if n_days > 0 else 0.0
        worst_day = float(np.min(returns)) if n_days > 0 else 0.0

        # Monthly returns (approximate)
        monthly_returns = []
        if n_days >= 21:
            for i in range(0, n_days - 20, 21):
                chunk = returns[i:i + 21]
                monthly_returns.append(float(np.prod(1 + chunk) - 1))

        best_month = max(monthly_returns) if monthly_returns else 0.0
        worst_month = min(monthly_returns) if monthly_returns else 0.0

        self._metrics = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "max_dd_duration_days": max_dd_duration,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar,
            "best_day": best_day,
            "worst_day": worst_day,
            "best_month": best_month,
            "worst_month": worst_month,
            "n_trading_days": n_days,
            "final_value": float(values[-1]),
            "initial_capital": float(self.equity_curve.initial_capital),
        }

        return self._metrics

    @property
    def metrics(self):
        if not self._metrics:
            self.compute()
        return self._metrics

    def render(self, console):
        """Render tearsheet as Rich panels."""
        from rich.panel import Panel
        from rich.table import Table as RichTable
        from rich.text import Text

        m = self.metrics
        if "error" in m:
            console.print(f"[red]{m['error']}[/red]")
            return

        tbl = RichTable(
            title="Performance Tearsheet",
            expand=True,
            title_style="bold white",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Metric", style="bold", width=22)
        tbl.add_column("Value", justify="right", width=16)

        rows = [
            ("Total Return", f"{m['total_return']:.2%}"),
            ("Ann. Return", f"{m['annualized_return']:.2%}"),
            ("Ann. Volatility", f"{m['annualized_vol']:.2%}"),
            ("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}"),
            ("Sortino Ratio", f"{m['sortino_ratio']:.2f}"),
            ("Max Drawdown", f"{m['max_drawdown']:.2%}"),
            ("Max DD Duration", f"{m['max_dd_duration_days']} days"),
            ("Win Rate", f"{m['win_rate']:.1%}"),
            ("Profit Factor", f"{m['profit_factor']:.2f}"),
            ("Calmar Ratio", f"{m['calmar_ratio']:.2f}"),
            ("Best Day", f"{m['best_day']:.2%}"),
            ("Worst Day", f"{m['worst_day']:.2%}"),
            ("Best Month", f"{m['best_month']:.2%}"),
            ("Worst Month", f"{m['worst_month']:.2%}"),
            ("Trading Days", f"{m['n_trading_days']:,}"),
            ("Final Value", f"${m['final_value']:,.0f}"),
        ]

        for label, value in rows:
            tbl.add_row(label, value)

        console.print(Panel(tbl, border_style="cyan"))

    def render_monthly_returns(self, console):
        """Render monthly return heatmap."""
        from rich.panel import Panel
        from rich.table import Table as RichTable
        from rich.text import Text

        data = self.equity_curve.to_arrays()
        dates = data["dates"]
        returns = data["returns"]

        if len(dates) == 0:
            return

        # Group by month
        monthly = {}
        for date_str, ret in zip(dates, returns):
            ym = date_str[:7]  # YYYY-MM
            if ym not in monthly:
                monthly[ym] = []
            monthly[ym].append(ret)

        tbl = RichTable(
            title="Monthly Returns",
            expand=True,
            title_style="bold white",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Month", style="bold", width=10)
        tbl.add_column("Return", justify="right", width=10)
        tbl.add_column("Bar", width=30)

        for ym in sorted(monthly.keys()):
            rets = monthly[ym]
            monthly_ret = float(np.prod(1 + np.array(rets)) - 1)
            style = "green" if monthly_ret >= 0 else "red"

            bar_len = min(int(abs(monthly_ret) * 500), 25)
            bar = ("+" if monthly_ret >= 0 else "-") * bar_len

            tbl.add_row(ym, Text(f"{monthly_ret:.2%}", style=style),
                        Text(bar, style=style))

        console.print(Panel(tbl, border_style="cyan"))


# ── BacktestResult ─────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    equity_curve: EquityCurve
    tearsheet: Tearsheet
    trade_count: int
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float


# ── Backtester (orchestrator) ──────────────────────────────────────────

class Backtester:
    """
    Orchestrates historical replay of trading strategies.

    For each trading day:
      1. Update MarketData nodes to historical prices
      2. graph.recalculate()
      3. strategy.on_bar() → generates orders
      4. TradingEngine processes orders (slippage/fees)
      5. Record equity curve point
    Generate tearsheet, store in Barbara.
    """

    def __init__(self, barbara=None):
        """
        Parameters
        ----------
        barbara : BarbaraDB or None — uses in-memory if None
        """
        if barbara is None:
            from bank_python.barbara import BarbaraDB
            self.barbara = BarbaraDB.open("backtest;default", db_path=":memory:")
        else:
            self.barbara = barbara

        self._strategies = []  # list of (strategy, trader_name, capital)
        self._data_loader = HistoricalDataLoader(barbara=self.barbara)

    def add_strategy(self, strategy, trader_name="Backtest", initial_capital=1_000_000):
        """Register a strategy for backtesting."""
        self._strategies.append((strategy, trader_name, initial_capital))

    def run(self, tickers, start_date, end_date, data=None):
        """
        Run the backtest.

        Parameters
        ----------
        tickers    : list[str]
        start_date : str
        end_date   : str
        data       : dict[str, list[dict]] or None — pre-loaded data (for testing)

        Returns
        -------
        list[BacktestResult]
        """
        # Load data
        if data is not None:
            self._data_loader.load_from_arrays(data)
            all_dates = sorted(set(
                bar["date"] for bars in data.values() for bar in bars
            ))
        else:
            loaded = self._data_loader.load(tickers, start_date, end_date)
            all_dates = sorted(set(
                bar["date"] for bars in loaded.values() for bar in bars
            ))

        if not all_dates:
            logger.warning("No trading dates found")
            return []

        results = []

        for strategy, trader_name, capital in self._strategies:
            result = self._run_strategy(
                strategy, trader_name, capital, tickers, all_dates, data,
            )
            results.append(result)

            # Store in Barbara
            if self.barbara:
                strat_name = strategy.__class__.__name__
                ts = datetime.now().isoformat()
                bk = f"/Backtest/results/{strat_name}/{start_date}_{end_date}/{ts}"
                self.barbara[bk] = result.tearsheet.metrics

        return results

    def _run_strategy(self, strategy, trader_name, capital, tickers, dates, data):
        """Run a single strategy through the backtest."""
        from bank_python.dagger import DependencyGraph, MarketData, Book
        from bank_python.trading_engine import TradingEngine

        # Set up isolated infrastructure
        graph = DependencyGraph()
        md_nodes = {}
        for ticker in tickers:
            node = MarketData(f"{ticker}_BT", price=0.0)
            graph.register(node)
            md_nodes[ticker] = node

        # Trading engine with in-memory Barbara
        engine = TradingEngine(
            barbara=self.barbara,
            graph=graph,
            state={"graph": graph},
        )

        # Attach backtest metadata to engine
        @dataclass
        class _BTTrader:
            name: str
            book: Book = None
            def __post_init__(self):
                if self.book is None:
                    self.book = Book(self.name)

        from dataclasses import dataclass as _dc

        bt_trader = _BTTrader(name=trader_name)
        engine._backtest_trader = bt_trader
        engine._backtest_capital = capital

        # Equity curve
        equity_curve = EquityCurve(initial_capital=capital)

        # Position tracking (shares held)
        positions = {t: 0.0 for t in tickers}
        cash = float(capital)
        processed_fills = set()  # track order_ids already processed

        # Expose positions to strategy for position-aware sizing
        engine._bt_positions = positions
        engine._bt_cash = cash

        strategy.on_start(None, engine)

        for date in dates:
            # Build price map for today
            prices = {}
            for ticker in tickers:
                price = self._get_price(ticker, date, data)
                if price and price > 0:
                    prices[ticker] = price

            if not prices:
                continue

            # Update market data nodes
            context = BacktestContext(graph, md_nodes, date, prices)
            context.apply()

            # Run strategy
            strategy.on_bar(context, engine, date)

            # Process only NEW fills (avoid double-counting)
            for order in engine.order_book.get_fills():
                if order.order_id in processed_fills:
                    continue
                processed_fills.add(order.order_id)

                ticker = order.instrument_name
                if ticker not in positions:
                    positions[ticker] = 0.0

                if order.side == "BUY":
                    positions[ticker] += order.filled_qty
                    cash -= order.filled_qty * order.filled_price + order.fill_cost
                else:
                    positions[ticker] -= order.filled_qty
                    cash += order.filled_qty * order.filled_price - order.fill_cost

            # Compute portfolio value
            port_value = cash
            for ticker, qty in positions.items():
                price = prices.get(ticker, 0)
                port_value += qty * price

            equity_curve.record(date, port_value)
            engine._backtest_capital = port_value
            engine._bt_cash = cash

        strategy.on_end(None, engine)

        # Compute tearsheet
        trade_count = len(engine.order_book.get_fills())
        tearsheet = Tearsheet(equity_curve)
        tearsheet.compute()

        return BacktestResult(
            equity_curve=equity_curve,
            tearsheet=tearsheet,
            trade_count=trade_count,
            strategy_name=strategy.__class__.__name__,
            start_date=dates[0] if dates else "",
            end_date=dates[-1] if dates else "",
            initial_capital=capital,
            final_value=tearsheet.metrics.get("final_value", capital),
        )

    def _get_price(self, ticker, date, data):
        """Get price for ticker on date from data or cache."""
        if data:
            bars = data.get(ticker, [])
            for bar in bars:
                if bar["date"] == date:
                    return bar["close"]
        return self._data_loader.get_price(ticker, date)


# ── Rendering helpers ──────────────────────────────────────────────────

def render_backtest_results(results, console):
    """Render backtest results summary."""
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    tbl = RichTable(
        title="Backtest Results",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Strategy", style="bold", width=18)
    tbl.add_column("Period", width=25)
    tbl.add_column("Return", justify="right", width=10)
    tbl.add_column("Sharpe", justify="right", width=8)
    tbl.add_column("Max DD", justify="right", width=10)
    tbl.add_column("Trades", justify="right", width=8)
    tbl.add_column("Final Value", justify="right", width=14)

    for r in results:
        m = r.tearsheet.metrics
        ret_style = "green" if m.get("total_return", 0) >= 0 else "red"
        tbl.add_row(
            r.strategy_name,
            f"{r.start_date} to {r.end_date}",
            Text(f"{m.get('total_return', 0):.2%}", style=ret_style),
            f"{m.get('sharpe_ratio', 0):.2f}",
            f"{m.get('max_drawdown', 0):.2%}",
            f"{r.trade_count:,}",
            f"${m.get('final_value', 0):,.0f}",
        )

    console.print(Panel(tbl, border_style="cyan"))

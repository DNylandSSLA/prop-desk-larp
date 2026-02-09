"""
Background simulation engine — runs the prop desk in a daemon thread,
broadcasts state to WebSocket clients every ~2 seconds.
"""

import logging
import os
import random
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

import numpy as np

logger = logging.getLogger("desk_app")

# Suppress yfinance noise
warnings.filterwarnings("ignore", category=FutureWarning)


@contextmanager
def _quiet_stdout():
    """Temporarily redirect stdout to suppress print() noise from imported modules."""
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


class SimulationEngine:
    """Singleton background thread that drives the prop desk simulation."""

    _instance = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.running = False
        self.tick_count = 0
        self._thread = None
        self._snapshot = None
        self._snapshot_lock = threading.Lock()
        self._state = None
        self._db = None
        self._config = None
        self._traders = None
        self._rng = np.random.default_rng(int(time.time()) % 2**31)
        self._start_time = None
        self._recent_orders = []
        self._recent_alerts = []
        self._mc_results = None
        self._stress_results = None
        self._price_history = {}  # ticker -> list of last 30 prices

        # Analytics state
        self._pnl_history = []        # [{tick, ts, desk, traders: {name: pnl}}]
        self._prev_greeks = {}        # trader_name -> {delta, gamma, vega}
        self._prev_prices = {}        # ticker -> price (last tick)
        self._prev_vix = None
        self._attribution = None      # latest P&L attribution
        self._bump_ladder = None      # latest bump ladder
        self._correlation = None      # latest correlation matrix
        self._trader_pnl_series = {}  # trader_name -> [pnl_values]

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="sim-engine")
        self._thread.start()

    def get_snapshot(self):
        with self._snapshot_lock:
            return self._snapshot

    def _run(self):
        """Main simulation loop."""
        logger.info("Starting up...")
        try:
            self._initialize()
        except Exception as e:
            logger.exception(f"Init failed: {e}")
            self.running = False
            return

        logger.info("Ready — entering tick loop")
        self._start_time = time.time()

        while self.running:
            try:
                self._tick()
            except Exception as e:
                logger.exception(f"Tick {self.tick_count} error: {e}")
            time.sleep(2)

    def _initialize(self):
        """Set up the prop desk — imports and calls setup functions."""
        from prop_desk import (
            open_db, ensure_config, get_config, setup_desk,
            snapshot_initial_values, EQUITY_TICKERS,
        )

        # Suppress print() noise from prop_desk/bank_python during setup
        with _quiet_stdout():
            self._db = open_db()
            ensure_config(self._db)
            self._config = get_config(self._db)
            self._state = setup_desk(self._db)
            self._traders = self._state["traders"]

        logger.info("Fetching live market data...")
        self._state["mgr"].update_all()
        snapshot_initial_values(self._traders)

        # Seed price history
        for ticker in EQUITY_TICKERS:
            eq = self._state["equities"].get(ticker)
            if eq and eq.value > 0:
                self._price_history[ticker] = [eq.value]
                self._prev_prices[ticker] = eq.value

        # Load history for signals
        for ticker in EQUITY_TICKERS[:12]:
            try:
                self._state["mgr"].load_history(ticker, "1y")
            except Exception:
                pass

        # Snapshot initial Greeks for attribution
        self._snapshot_greeks()

        # Initialize VIX baseline
        self._prev_vix = self._state["vix_md"].value

        n_pos = sum(len(t.book.positions) for t in self._traders)
        logger.info(f"Desk online: {len(self._traders)} traders, {n_pos} positions")

        # Build and broadcast initial snapshot
        self._build_and_broadcast()

    def _tick(self):
        """One simulation tick — jiggle prices, revalue, check risk, broadcast."""
        self.tick_count += 1
        from showcase import _jiggle_market

        # Jiggle market prices
        intensity = 0.8 + 0.2 * min(self.tick_count / 100, 1.0)
        if self._rng.random() < 0.10:
            intensity *= 2.0
        _jiggle_market(self._state, self._rng, intensity=intensity)

        # Record price history for sparklines
        for ticker, hist in self._price_history.items():
            eq = self._state["equities"].get(ticker)
            if eq and eq.value > 0:
                hist.append(eq.value)
                if len(hist) > 30:
                    hist.pop(0)

        # Track P&L history
        self._record_pnl()

        # Compute P&L attribution (every tick, it's fast)
        self._compute_attribution()

        # Compute bump ladder (every 5 ticks)
        if self.tick_count % 5 == 0:
            self._compute_bump_ladder()

        # Compute correlation matrix (every 10 ticks, needs price history)
        if self.tick_count % 10 == 0:
            self._compute_correlation()

        # Snapshot Greeks for next tick's attribution
        self._snapshot_greeks()

        # Update prev prices
        for ticker in self._price_history:
            eq = self._state["equities"].get(ticker)
            if eq and eq.value > 0:
                self._prev_prices[ticker] = eq.value
        self._prev_vix = self._state["vix_md"].value

        # Every ~10 ticks: generate signals and execute orders
        if self.tick_count % 10 == 0:
            self._run_signals_and_orders()

        # Every ~30 ticks: run MC and stress tests (in a sub-thread to not block)
        if self.tick_count % 30 == 0:
            threading.Thread(
                target=self._run_heavy_analytics, daemon=True, name="sim-heavy"
            ).start()

        # Build snapshot and broadcast
        self._build_and_broadcast()

    def _snapshot_greeks(self):
        """Store current Greeks per trader for next tick's attribution."""
        from prop_desk import compute_trader_risk
        for t in self._traders:
            risk = compute_trader_risk(t)
            self._prev_greeks[t.name] = {
                "delta": risk["delta"],
                "gamma": risk["gamma"],
                "vega": risk["vega"],
            }

    def _record_pnl(self):
        """Record P&L for time series chart."""
        desk_pnl = sum(t.book.total_value - t.initial_value for t in self._traders)
        trader_pnls = {}
        for t in self._traders:
            pnl = t.book.total_value - t.initial_value
            trader_pnls[t.name] = round(pnl, 0)
            # Track per-trader series for stats
            if t.name not in self._trader_pnl_series:
                self._trader_pnl_series[t.name] = []
            self._trader_pnl_series[t.name].append(pnl)

        self._pnl_history.append({
            "tick": self.tick_count,
            "ts": datetime.now().strftime("%H:%M:%S"),
            "desk": round(desk_pnl, 0),
            "traders": trader_pnls,
        })
        # Keep last 300 ticks (~10 minutes)
        if len(self._pnl_history) > 300:
            self._pnl_history.pop(0)

    def _compute_attribution(self):
        """Approximate P&L attribution using Greeks and price changes."""
        from prop_desk import EQUITY_TICKERS
        if not self._prev_greeks or not self._prev_prices:
            return

        # Aggregate price move across equities
        total_delta_pnl = 0.0
        total_gamma_pnl = 0.0
        total_vega_pnl = 0.0

        for t in self._traders:
            prev = self._prev_greeks.get(t.name)
            if not prev:
                continue

            # Compute weighted average spot return for this trader's book
            # Approximate: use the avg equity move across the market
            spot_returns = []
            for ticker in list(self._prev_prices.keys())[:20]:
                eq = self._state["equities"].get(ticker)
                prev_px = self._prev_prices.get(ticker)
                if eq and prev_px and prev_px > 0 and eq.value > 0:
                    spot_returns.append((eq.value - prev_px) / prev_px)

            if not spot_returns:
                continue

            avg_return = np.mean(spot_returns)
            avg_price = np.mean([
                self._state["equities"][tk].value
                for tk in list(self._prev_prices.keys())[:20]
                if tk in self._state["equities"] and self._state["equities"][tk].value > 0
            ])

            ds = avg_return * avg_price  # dollar move per share

            total_delta_pnl += prev["delta"] * avg_return
            total_gamma_pnl += 0.5 * prev["gamma"] * (avg_return ** 2) * avg_price
            total_vega_pnl += prev["vega"] * (
                (self._state["vix_md"].value - (self._prev_vix or self._state["vix_md"].value)) / 100
            )

        desk_pnl = sum(t.book.total_value - t.initial_value for t in self._traders)
        prev_desk_pnl = 0
        if len(self._pnl_history) >= 2:
            prev_desk_pnl = self._pnl_history[-2]["desk"]
        tick_pnl = desk_pnl - prev_desk_pnl

        # Theta: approximate as time decay (~-0.1% of vega per tick)
        total_theta = sum(
            -abs(self._prev_greeks.get(t.name, {}).get("vega", 0)) * 0.001
            for t in self._traders
        )

        explained = total_delta_pnl + total_gamma_pnl + total_vega_pnl + total_theta
        unexplained = tick_pnl - explained

        self._attribution = {
            "delta_pnl": round(total_delta_pnl, 0),
            "gamma_pnl": round(total_gamma_pnl, 0),
            "vega_pnl": round(total_vega_pnl, 0),
            "theta_pnl": round(total_theta, 0),
            "unexplained": round(unexplained, 0),
            "total": round(tick_pnl, 0),
        }

    def _compute_bump_ladder(self):
        """Compute P&L impact for parallel spot bumps using Greeks."""
        from prop_desk import compute_trader_risk, aggregate_desk_risk

        bumps = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0]
        desk_risk = aggregate_desk_risk(self._traders)
        desk_delta = desk_risk["delta"]
        desk_gamma = desk_risk["gamma"]

        ladder = []
        for bump_pct in bumps:
            b = bump_pct / 100.0
            # Desk level: delta * bump + 0.5 * gamma * bump^2
            desk_impact = desk_delta * b + 0.5 * desk_gamma * (b ** 2)

            # Per trader
            per_trader = {}
            for t in self._traders:
                risk = compute_trader_risk(t)
                impact = risk["delta"] * b + 0.5 * risk["gamma"] * (b ** 2)
                per_trader[t.name] = round(impact, 0)

            ladder.append({
                "bump": bump_pct,
                "desk": round(desk_impact, 0),
                "traders": per_trader,
            })

        self._bump_ladder = ladder

    def _compute_correlation(self):
        """Compute return correlation matrix from price history."""
        tickers = []
        returns_matrix = []

        for ticker, hist in self._price_history.items():
            if len(hist) < 5:
                continue
            prices = np.array(hist)
            rets = np.diff(prices) / prices[:-1]
            if len(rets) >= 4:
                tickers.append(ticker)
                returns_matrix.append(rets[-min(len(rets), 20):])

        if len(tickers) < 3:
            return

        # Align to same length
        min_len = min(len(r) for r in returns_matrix)
        aligned = np.array([r[-min_len:] for r in returns_matrix])

        # Compute correlation
        try:
            corr = np.corrcoef(aligned)
            # Take top 12 tickers for display
            n = min(12, len(tickers))
            self._correlation = {
                "tickers": tickers[:n],
                "matrix": [[round(float(corr[i][j]), 2) for j in range(n)] for i in range(n)],
            }
        except Exception:
            pass

    def _compute_trader_stats(self):
        """Compute rolling stats per trader: Sharpe, max drawdown, win rate."""
        stats = []
        for t in self._traders:
            series = self._trader_pnl_series.get(t.name, [])
            if len(series) < 3:
                stats.append({
                    "name": t.name,
                    "sharpe": 0,
                    "max_dd": 0,
                    "win_rate": 0,
                    "total_pnl": 0,
                    "n_ticks": len(series),
                })
                continue

            pnls = np.array(series)
            # Tick-over-tick changes
            changes = np.diff(pnls)

            # Sharpe (annualized-ish: assume 2s ticks, ~450 ticks/15min)
            mean_ret = np.mean(changes)
            std_ret = np.std(changes) if len(changes) > 1 else 1
            sharpe = (mean_ret / std_ret * np.sqrt(450)) if std_ret > 0 else 0

            # Max drawdown
            cummax = np.maximum.accumulate(pnls)
            drawdowns = pnls - cummax
            max_dd = float(np.min(drawdowns))

            # Win rate (ticks where P&L improved)
            wins = np.sum(changes > 0)
            win_rate = float(wins / len(changes)) if len(changes) > 0 else 0

            stats.append({
                "name": t.name,
                "sharpe": round(float(sharpe), 2),
                "max_dd": round(max_dd, 0),
                "win_rate": round(win_rate * 100, 1),
                "total_pnl": round(float(pnls[-1]), 0),
                "n_ticks": len(series),
            })

        return stats

    # Personality-aware order profiles: preferred tickers, bias, limit pct
    TRADER_PROFILES = {
        "Joe": {
            "tickers": ["TLT", "GLD", "XOM", "CAT", "SPY", "GE", "DE"],
            "bias": "BUY",         # adding to long duration / real assets
            "limit_pct": 0.4,      # patient, uses limits
        },
        "Tracy": {
            "tickers": ["JPM", "BAC", "GS", "C", "WFC", "MS", "LQD", "HYG"],
            "bias": "BUY",         # adding to bank longs
            "limit_pct": 0.3,
        },
        "Matt": {
            "tickers": ["GOOGL", "META", "XOM", "CVX", "V", "MA", "HD", "LOW",
                         "AAPL", "MSFT", "SPY"],
            "bias": None,          # no directional bias — always pairs
            "limit_pct": 0.5,      # very patient
        },
        "Katie": {
            "tickers": ["AAPL", "MSFT", "NVDA", "LLY", "UNH", "AMZN", "COST",
                         "JNJ", "QQQ", "GOOGL"],
            "bias": "BUY",         # follows flow, currently buying
            "limit_pct": 0.2,      # hits the market
        },
        "Nero": {
            "tickers": ["BAC", "GLD", "LQD", "HYG", "SPY"],
            "bias": None,          # opportunistic both ways
            "limit_pct": 0.6,      # very patient, waits for dislocations
        },
        "Tony": {
            "tickers": ["NVDA", "AVGO", "AMD", "MSFT", "META", "CRM", "NOW",
                         "PANW", "TSLA", "NFLX", "DDOG", "NET"],
            "bias": "BUY",         # perma-bull on tech
            "limit_pct": 0.15,     # aggressive, hits the offer
        },
        "Adam": {
            "tickers": ["EEM", "EFA", "GLD", "XOM", "LMT", "SPY", "RTX"],
            "bias": "BUY",
            "limit_pct": 0.35,
        },
        "Charlie": {
            "tickers": ["NVDA", "META", "TSLA", "AAPL", "MSFT"],
            "bias": None,          # delta-neutral, direction irrelevant
            "limit_pct": 0.4,
        },
        "Izzy": {
            "tickers": ["XOM", "CVX", "COP", "SLB", "FCX", "NEM", "GLD",
                         "EOG", "MPC", "OXY"],
            "bias": "BUY",         # long commodities
            "limit_pct": 0.3,
        },
        "Claudia": {
            "tickers": ["JNJ", "PFE", "KO", "PEP", "WMT", "COST", "TLT",
                         "TSLA", "COIN", "RIVN"],
            "bias": None,          # buys defensives, sells cyclicals
            "limit_pct": 0.3,
        },
        "Noah": {
            "tickers": ["COIN", "PLTR", "SOFI", "HOOD", "AFRM", "SHOP",
                         "DDOG", "NET", "RBLX", "U"],
            "bias": "BUY",         # buys every dip in innovation
            "limit_pct": 0.1,      # slams the market button
        },
        "Skanda": {
            "tickers": ["TLT", "JPM", "SCHW", "BLK", "SPY"],
            "bias": None,          # curve trades are relative
            "limit_pct": 0.5,
        },
        "Torsten": {
            "tickers": ["AAPL", "GOOGL", "JPM", "EEM", "QQQ", "GLD",
                         "META", "AMZN"],
            "bias": "BUY",         # risk-on tilt currently
            "limit_pct": 0.25,
        },
        "Robin": {
            "tickers": ["EEM", "EFA", "MELI", "NU", "SE", "SPY"],
            "bias": "BUY",         # adding EM longs
            "limit_pct": 0.35,
        },
    }

    def _run_signals_and_orders(self):
        """Generate personality-aware orders for each trader."""
        try:
            from bank_python.trading_engine import TradingEngine

            engine = TradingEngine(
                barbara=self._db, graph=self._state["graph"], state=self._state
            )

            # Pick 4-7 traders to be active this round
            n_active = int(self._rng.integers(4, 8))
            active_traders = list(
                self._rng.choice(self._traders, size=min(n_active, len(self._traders)), replace=False)
            )

            new_orders = []
            for trader in active_traders:
                profile = self.TRADER_PROFILES.get(trader.name)
                if not profile:
                    continue

                # Pick 1-2 tickers from this trader's preferred list
                n_picks = int(self._rng.integers(1, 3))
                pool = profile["tickers"]
                picks = list(self._rng.choice(pool, size=min(n_picks, len(pool)), replace=False))

                for sym in picks:
                    price_eq = self._state["equities"].get(sym)
                    if not price_eq or price_eq.value <= 0:
                        continue

                    # Determine side from bias (with some randomness)
                    bias = profile["bias"]
                    if bias is None:
                        side = str(self._rng.choice(["BUY", "SELL"]))
                    elif self._rng.random() < 0.75:
                        side = bias
                    else:
                        side = "SELL" if bias == "BUY" else "BUY"

                    # Claudia special: sells cyclicals, buys defensives
                    if trader.name == "Claudia":
                        cyclicals = {"TSLA", "COIN", "RIVN", "NVDA", "AMD"}
                        if sym in cyclicals:
                            side = "SELL"
                        else:
                            side = "BUY"

                    book_val = max(abs(trader.book.total_value), 50_000)
                    max_notional = book_val * 0.20
                    max_qty = max(1, int(max_notional / price_eq.value))
                    qty = int(self._rng.integers(1, max(2, max_qty)))

                    is_limit = self._rng.random() < profile["limit_pct"]
                    order_type = "LIMIT" if is_limit else "MARKET"
                    limit_price = (
                        round(price_eq.value * (1 + self._rng.uniform(-0.02, 0.02)), 2)
                        if is_limit else None
                    )

                    try:
                        order = engine.submit_order(
                            trader=trader, instrument_name=sym,
                            side=side, quantity=qty, order_type=order_type,
                            limit_price=limit_price, book_value=book_val,
                        )
                        new_orders.append({
                            "trader": trader.name,
                            "instrument": sym,
                            "side": side,
                            "quantity": qty,
                            "type": order_type,
                            "status": order.status,
                            "fill_price": round(order.filled_price, 2) if order.filled_price else None,
                            "timestamp": datetime.now().isoformat(),
                        })
                    except Exception:
                        pass

            # Keep last 25 orders
            self._recent_orders = (new_orders + self._recent_orders)[:25]

        except Exception as e:
            logger.warning(f"Signal/order generation failed: {e}")

    def _run_heavy_analytics(self):
        """Run MC simulation and stress tests (called from sub-thread)."""
        try:
            from mc_engine import MCConfig, MonteCarloEngine, StressEngine
            from prop_desk import compute_greeks

            # MC simulation (50K paths, fast)
            mc_config = MCConfig(
                n_paths=50_000, horizon_days=1,
                random_seed=int(self._rng.integers(0, 2**31)),
            )
            engine = MonteCarloEngine(
                self._traders, self._state, self._db, mc_config,
                compute_greeks_fn=compute_greeks,
            )
            self._mc_results = engine.run_full_simulation()

            # Stress tests
            stress = StressEngine()
            self._stress_results = stress.run_all_scenarios(self._traders, self._state)

        except Exception as e:
            logger.warning(f"Heavy analytics failed: {e}")

    def _build_and_broadcast(self):
        """Serialize state and send to all WebSocket clients."""
        from desk_app.serializers import serialize_state

        snapshot = serialize_state(
            traders=self._traders,
            state=self._state,
            config=self._config,
            db=self._db,
            tick=self.tick_count,
            start_time=self._start_time or time.time(),
            recent_orders=self._recent_orders,
            mc_results=self._mc_results,
            stress_results=self._stress_results,
            price_history=self._price_history,
            pnl_history=self._pnl_history,
            attribution=self._attribution,
            bump_ladder=self._bump_ladder,
            correlation=self._correlation,
            trader_stats=self._compute_trader_stats(),
        )

        with self._snapshot_lock:
            self._snapshot = snapshot

        # Broadcast to all connected clients via Channels
        try:
            from channels.layers import get_channel_layer
            from asgiref.sync import async_to_sync

            channel_layer = get_channel_layer()
            if channel_layer:
                async_to_sync(channel_layer.group_send)(
                    "simulation",
                    {"type": "simulation.tick", "data": snapshot},
                )
        except Exception as e:
            logger.debug(f"Broadcast failed (no clients?): {e}")

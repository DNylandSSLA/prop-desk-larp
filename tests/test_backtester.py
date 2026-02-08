"""Tests for backtester — historical replay, strategies, tearsheets."""

import numpy as np
import pytest
from dataclasses import dataclass

from bank_python import BarbaraDB, DependencyGraph, MarketData, Book
from bank_python.backtester import (
    HistoricalDataLoader,
    BacktestContext,
    BacktestStrategy,
    MomentumBacktest,
    VolArbBacktest,
    StatArbBacktest,
    MacroBacktest,
    EquityCurve,
    Tearsheet,
    BacktestResult,
    Backtester,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _generate_price_data(tickers, n_days=100, seed=42):
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    data = {}

    for ticker in tickers:
        base = rng.uniform(50, 200)
        bars = []
        price = base
        for i in range(n_days):
            price = price * (1 + rng.normal(0.0005, 0.02))
            bars.append({
                "date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": 1_000_000,
            })
        data[ticker] = bars

    return data


# ── HistoricalDataLoader ───────────────────────────────────────────────

class TestHistoricalDataLoader:
    def test_load_from_arrays(self):
        loader = HistoricalDataLoader()
        data = _generate_price_data(["AAPL", "MSFT"], n_days=10)
        loader.load_from_arrays(data)
        assert len(loader._cache) == 2

    def test_get_dates(self):
        loader = HistoricalDataLoader()
        data = _generate_price_data(["AAPL"], n_days=10)
        loader.load_from_arrays(data)
        dates = loader.get_dates()
        assert len(dates) >= 1

    def test_get_price(self):
        loader = HistoricalDataLoader()
        data = {"AAPL": [
            {"date": "2024-01-01", "open": 99, "high": 101, "low": 98,
             "close": 100.0, "volume": 1000000},
        ]}
        loader.load_from_arrays(data)
        price = loader.get_price("AAPL", "2024-01-01")
        assert price == 100.0

    def test_get_price_missing(self):
        loader = HistoricalDataLoader()
        price = loader.get_price("AAPL", "2024-01-01")
        assert price is None

    def test_get_bar(self):
        loader = HistoricalDataLoader()
        data = {"AAPL": [
            {"date": "2024-01-01", "open": 99, "high": 101, "low": 98,
             "close": 100.0, "volume": 1000000},
        ]}
        loader.load_from_arrays(data)
        bar = loader.get_bar("AAPL", "2024-01-01")
        assert bar is not None
        assert bar["close"] == 100.0


# ── BacktestContext ────────────────────────────────────────────────────

class TestBacktestContext:
    def test_apply_updates_nodes(self):
        graph = DependencyGraph()
        md = MarketData("AAPL_BT", price=0.0)
        graph.register(md)

        ctx = BacktestContext(
            graph=graph,
            market_data_nodes={"AAPL": md},
            date="2024-01-01",
            prices={"AAPL": 150.0},
        )
        ctx.apply()
        assert md.value == 150.0

    def test_apply_skips_zero_price(self):
        graph = DependencyGraph()
        md = MarketData("AAPL_BT", price=100.0)
        graph.register(md)

        ctx = BacktestContext(
            graph=graph,
            market_data_nodes={"AAPL": md},
            date="2024-01-01",
            prices={"AAPL": 0.0},
        )
        ctx.apply()
        assert md.value == 100.0  # unchanged


# ── EquityCurve ────────────────────────────────────────────────────────

class TestEquityCurve:
    def test_initial_state(self):
        ec = EquityCurve(initial_capital=100000)
        assert ec.initial_capital == 100000

    def test_record_day(self):
        ec = EquityCurve(initial_capital=100000)
        ec.record("2024-01-01", 101000)
        rows = list(ec.table)
        assert len(rows) == 1
        assert rows[0]["portfolio_value"] == 101000

    def test_daily_return(self):
        ec = EquityCurve(initial_capital=100000)
        ec.record("2024-01-01", 101000)
        rows = list(ec.table)
        assert abs(rows[0]["daily_return"] - 0.01) < 0.001

    def test_cumulative_return(self):
        ec = EquityCurve(initial_capital=100000)
        ec.record("2024-01-01", 110000)
        rows = list(ec.table)
        assert abs(rows[0]["cumulative_return"] - 0.10) < 0.001

    def test_drawdown(self):
        ec = EquityCurve(initial_capital=100000)
        ec.record("2024-01-01", 110000)
        ec.record("2024-01-02", 100000)
        rows = list(ec.table)
        # DD = (100000 - 110000) / 110000 = -9.09%
        assert rows[1]["drawdown"] < 0
        assert abs(rows[1]["drawdown"] - (-10000 / 110000)) < 0.001

    def test_high_water_mark(self):
        ec = EquityCurve(initial_capital=100000)
        ec.record("2024-01-01", 110000)
        ec.record("2024-01-02", 105000)
        rows = list(ec.table)
        assert rows[1]["high_water_mark"] == 110000

    def test_to_arrays(self):
        ec = EquityCurve(initial_capital=100000)
        ec.record("2024-01-01", 101000)
        ec.record("2024-01-02", 102000)
        arr = ec.to_arrays()
        assert len(arr["dates"]) == 2
        assert len(arr["values"]) == 2

    def test_empty_to_arrays(self):
        ec = EquityCurve(initial_capital=100000)
        arr = ec.to_arrays()
        assert arr["dates"] == []


# ── Tearsheet ──────────────────────────────────────────────────────────

class TestTearsheet:
    def _make_equity_curve(self, n_days=252, annual_ret=0.10, annual_vol=0.15):
        """Generate a synthetic equity curve."""
        rng = np.random.default_rng(42)
        ec = EquityCurve(initial_capital=1_000_000)

        daily_ret = annual_ret / 252
        daily_vol = annual_vol / np.sqrt(252)

        value = 1_000_000
        for i in range(n_days):
            ret = rng.normal(daily_ret, daily_vol)
            value *= (1 + ret)
            ec.record(f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", value)

        return ec

    def test_compute_metrics(self):
        ec = self._make_equity_curve()
        ts = Tearsheet(ec)
        m = ts.compute()

        assert "total_return" in m
        assert "annualized_return" in m
        assert "sharpe_ratio" in m
        assert "max_drawdown" in m
        assert "win_rate" in m

    def test_sharpe_positive(self):
        ec = self._make_equity_curve(annual_ret=0.15, annual_vol=0.10)
        ts = Tearsheet(ec)
        m = ts.compute()
        # High return, low vol => positive Sharpe
        assert m["sharpe_ratio"] > 0

    def test_sortino_ratio(self):
        ec = self._make_equity_curve()
        ts = Tearsheet(ec)
        m = ts.compute()
        assert "sortino_ratio" in m

    def test_max_drawdown_negative(self):
        ec = self._make_equity_curve()
        ts = Tearsheet(ec)
        m = ts.compute()
        assert m["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self):
        ec = self._make_equity_curve()
        ts = Tearsheet(ec)
        m = ts.compute()
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_empty_curve(self):
        ec = EquityCurve(initial_capital=100000)
        ts = Tearsheet(ec)
        m = ts.compute()
        assert "error" in m

    def test_metrics_property(self):
        ec = self._make_equity_curve(n_days=50)
        ts = Tearsheet(ec)
        m = ts.metrics
        assert "total_return" in m

    def test_n_trading_days(self):
        ec = self._make_equity_curve(n_days=100)
        ts = Tearsheet(ec)
        m = ts.compute()
        assert m["n_trading_days"] == 100

    def test_profit_factor(self):
        ec = self._make_equity_curve()
        ts = Tearsheet(ec)
        m = ts.compute()
        assert m["profit_factor"] > 0


# ── Strategies ─────────────────────────────────────────────────────────

class TestMomentumBacktest:
    def test_creates(self):
        strat = MomentumBacktest(["AAPL", "MSFT"])
        assert strat.lookback == 20

    def test_on_bar_no_history(self):
        strat = MomentumBacktest(["AAPL"], lookback=5)
        # Should not error with empty context
        graph = DependencyGraph()
        md = MarketData("AAPL_BT", price=100.0)
        graph.register(md)

        from bank_python.trading_engine import TradingEngine
        engine = TradingEngine(state={})

        @dataclass
        class _T:
            name: str = "BT"
            book: Book = None
            def __post_init__(self):
                self.book = Book(self.name)

        engine._backtest_trader = _T()
        engine._backtest_capital = 100000

        ctx = BacktestContext(graph, {"AAPL": md}, "2024-01-01", {"AAPL": 100.0})
        # Should not raise
        strat.on_bar(ctx, engine, "2024-01-01")


class TestStatArbBacktest:
    def test_creates(self):
        strat = StatArbBacktest([("GOOGL", "AMZN")])
        assert strat.z_threshold == 2.0


class TestMacroBacktest:
    def test_creates(self):
        strat = MacroBacktest(["SPY"])
        assert strat.trade_qty == 100


# ── Backtester (integration) ──────────────────────────────────────────

class TestBacktester:
    def test_run_with_synthetic_data(self):
        data = _generate_price_data(["AAPL", "MSFT"], n_days=60, seed=42)
        bt = Backtester()
        bt.add_strategy(
            MomentumBacktest(["AAPL", "MSFT"], lookback=10, top_n=1),
            trader_name="TestTrader",
            initial_capital=100_000,
        )
        results = bt.run(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-03-01",
            data=data,
        )
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, BacktestResult)
        assert r.strategy_name == "MomentumBacktest"
        assert r.initial_capital == 100_000

    def test_tearsheet_computed(self):
        data = _generate_price_data(["AAPL"], n_days=60)
        bt = Backtester()
        bt.add_strategy(MacroBacktest(["AAPL"]))
        results = bt.run(["AAPL"], "2024-01-01", "2024-03-01", data=data)
        assert len(results) == 1
        m = results[0].tearsheet.metrics
        assert "sharpe_ratio" in m

    def test_equity_curve_populated(self):
        data = _generate_price_data(["AAPL"], n_days=30)
        bt = Backtester()
        bt.add_strategy(MacroBacktest(["AAPL"]), initial_capital=50_000)
        results = bt.run(["AAPL"], "2024-01-01", "2024-02-01", data=data)
        ec_rows = list(results[0].equity_curve.table)
        assert len(ec_rows) > 0

    def test_no_data(self):
        bt = Backtester()
        bt.add_strategy(MomentumBacktest(["AAPL"]))
        results = bt.run(["AAPL"], "2024-01-01", "2024-02-01", data={})
        assert results == []

    def test_multiple_strategies(self):
        data = _generate_price_data(["AAPL", "MSFT"], n_days=60)
        bt = Backtester()
        bt.add_strategy(MomentumBacktest(["AAPL", "MSFT"], lookback=10))
        bt.add_strategy(MacroBacktest(["AAPL"]))
        results = bt.run(["AAPL", "MSFT"], "2024-01-01", "2024-03-01", data=data)
        assert len(results) == 2

    def test_stat_arb_backtest(self):
        data = _generate_price_data(["GOOGL", "AMZN"], n_days=80)
        bt = Backtester()
        bt.add_strategy(StatArbBacktest([("GOOGL", "AMZN")], lookback=30))
        results = bt.run(["GOOGL", "AMZN"], "2024-01-01", "2024-04-01", data=data)
        assert len(results) == 1

    def test_barbara_persistence(self):
        db = BarbaraDB.open("test", db_path=":memory:")
        data = _generate_price_data(["AAPL"], n_days=30)
        bt = Backtester(barbara=db)
        bt.add_strategy(MacroBacktest(["AAPL"]))
        bt.run(["AAPL"], "2024-01-01", "2024-02-01", data=data)

        keys = db.keys(prefix="/Backtest/")
        assert len(keys) >= 1
        db.close()

    def test_in_memory_barbara_default(self):
        bt = Backtester()
        assert bt.barbara is not None

    def test_vol_arb_backtest(self):
        data = _generate_price_data(["AAPL"], n_days=60)
        bt = Backtester()
        bt.add_strategy(VolArbBacktest(["AAPL"]))
        results = bt.run(["AAPL"], "2024-01-01", "2024-03-01", data=data)
        assert len(results) == 1


# ── BacktestResult ─────────────────────────────────────────────────────

class TestBacktestResult:
    def test_create(self):
        ec = EquityCurve(100000)
        ec.record("2024-01-01", 101000)
        ts = Tearsheet(ec)
        ts.compute()

        r = BacktestResult(
            equity_curve=ec,
            tearsheet=ts,
            trade_count=10,
            strategy_name="Test",
            start_date="2024-01-01",
            end_date="2024-12-31",
            initial_capital=100000,
            final_value=110000,
        )
        assert r.trade_count == 10
        assert r.strategy_name == "Test"

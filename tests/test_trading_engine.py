"""Tests for trading_engine — orders, execution, signals, audit."""

import json
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pytest

from bank_python import BarbaraDB, DependencyGraph, MarketData, Book, Position, Table
from bank_python.market_data import Equity, HistoricalStore
from bank_python.trading_engine import (
    Order,
    Signal,
    ExecutionModel,
    PositionLimits,
    OrderBook,
    MomentumSignal,
    VolArbSignal,
    StatArbSignal,
    MacroSignal,
    AuditTrail,
    TradingEngine,
    _next_order_id,
)


# ── Helpers ──────────────────────────────────────────────────────────────

@dataclass
class MockTrader:
    name: str
    strategy: str = "test"
    book: Book = None

    def __post_init__(self):
        if self.book is None:
            self.book = Book(self.name)


def _make_state_with_history(tickers, n_days=60, base_prices=None):
    """Create a state dict with historical data for signal testing."""
    graph = DependencyGraph()
    spots = {}
    equities = {}
    base_prices = base_prices or {}

    for ticker in tickers:
        bp = base_prices.get(ticker, 100.0)
        md = MarketData(f"{ticker}_SPOT", price=bp)
        graph.register(md)
        spots[ticker] = md

        eq = Equity(ticker, spot_source=md)
        graph.register(eq)
        equities[ticker] = eq

    # Build history
    from bank_python.market_data import HistoricalStore, MarketDataManager
    history = HistoricalStore()
    rng = np.random.default_rng(42)

    for ticker in tickers:
        bp = base_prices.get(ticker, 100.0)
        prices = [bp]
        for i in range(n_days - 1):
            prices.append(prices[-1] * (1.0 + rng.normal(0.001, 0.02)))

        rows = []
        for i, p in enumerate(prices):
            rows.append({
                "timestamp": f"2024-01-{i+1:02d}",
                "open": p * 0.99,
                "high": p * 1.01,
                "low": p * 0.98,
                "close": p,
                "volume": 1_000_000,
            })
        history.append(ticker, rows)

    # Minimal MarketDataManager-like
    class MockMgr:
        def __init__(self):
            self.history = history
            self._tickers = {}
        @property
        def registered_tickers(self):
            return self._tickers

    mgr = MockMgr()
    for ticker in tickers:
        mgr._tickers[ticker] = spots[ticker]

    sofr_md = MarketData("SOFR", price=0.053)
    eurusd_md = MarketData("EURUSD_RATE", price=1.08)
    graph.register(sofr_md)
    graph.register(eurusd_md)

    return {
        "graph": graph,
        "mgr": mgr,
        "spots": spots,
        "equities": equities,
        "sofr_md": sofr_md,
        "eurusd_md": eurusd_md,
        "traders": [],
    }


# ── Order ───────────────────────────────────────────────────────────────

class TestOrder:
    def test_create_order(self):
        o = Order(
            order_id="ORD-000001",
            trader="Alice",
            instrument_name="AAPL",
            side="BUY",
            quantity=100,
        )
        assert o.status == "PENDING"
        assert o.filled_qty == 0.0

    def test_order_defaults(self):
        o = Order(
            order_id="ORD-000002",
            trader="Bob",
            instrument_name="MSFT",
            side="SELL",
            quantity=50,
        )
        assert o.order_type == "MARKET"
        assert o.limit_price is None
        assert o.reject_reason == ""

    def test_signal_dataclass(self):
        s = Signal("AAPL", "BUY", 100, 0.8, "test reason")
        assert s.instrument == "AAPL"
        assert s.strength == 0.8


# ── ExecutionModel ──────────────────────────────────────────────────────

class TestExecutionModel:
    def test_default_params(self):
        em = ExecutionModel()
        assert em.fee_per_share == 0.005
        assert em.commission_pct == 0.001
        assert em.slippage_bps == 5.0

    def test_buy_execution(self):
        em = ExecutionModel()
        order = Order("O1", "Alice", "AAPL", "BUY", 100)
        fill_price, fees = em.execute(order, 150.0)
        # Buy should have higher fill price (slippage up)
        assert fill_price >= 150.0
        assert fees > 0

    def test_sell_execution(self):
        em = ExecutionModel()
        order = Order("O1", "Alice", "AAPL", "SELL", 100)
        fill_price, fees = em.execute(order, 150.0)
        # Sell should have lower fill price (slippage down)
        assert fill_price <= 150.0
        assert fees > 0

    def test_slippage_proportional_to_size(self):
        em = ExecutionModel(slippage_bps=10.0)
        small_order = Order("O1", "Alice", "AAPL", "BUY", 100)
        large_order = Order("O2", "Alice", "AAPL", "BUY", 10000)

        price_small, _ = em.execute(small_order, 100.0, avg_daily_volume=100000)
        price_large, _ = em.execute(large_order, 100.0, avg_daily_volume=100000)

        # Larger order -> more slippage
        assert price_large > price_small

    def test_limit_order_caps_fill(self):
        em = ExecutionModel(slippage_bps=100.0)  # high slippage
        order = Order("O1", "Alice", "AAPL", "BUY", 1000,
                       order_type="LIMIT", limit_price=101.0)
        fill_price, _ = em.execute(order, 100.0, avg_daily_volume=10000)
        assert fill_price <= 101.0

    def test_fees_calculation(self):
        em = ExecutionModel(fee_per_share=0.01, commission_pct=0.002, slippage_bps=0.0)
        order = Order("O1", "Alice", "AAPL", "BUY", 100)
        fill_price, fees = em.execute(order, 100.0)
        # fees = 100 * 0.01 + 100 * 100 * 0.002 = 1 + 20 = 21
        assert abs(fees - 21.0) < 0.1


# ── PositionLimits ──────────────────────────────────────────────────────

class TestPositionLimits:
    def test_within_limits(self):
        pl = PositionLimits()
        order = Order("O1", "Alice", "AAPL", "BUY", 100)
        trader = MockTrader("Alice")
        ok, reason = pl.check(order, trader, 150.0, book_value=100000)
        assert ok
        assert reason == ""

    def test_notional_breach(self):
        pl = PositionLimits()
        order = Order("O1", "Alice", "AAPL", "BUY", 10000)
        trader = MockTrader("Alice")
        ok, reason = pl.check(order, trader, 100.0, book_value=100000)
        assert not ok
        assert "Notional" in reason

    def test_concentration_breach(self):
        pl = PositionLimits()
        order = Order("O1", "Alice", "AAPL", "BUY", 100)
        trader = MockTrader("Alice")
        # Small book => 100 * 100 / 200 = 5000% concentration
        ok, reason = pl.check(order, trader, 100.0, book_value=200)
        assert not ok
        assert "Concentration" in reason

    def test_loads_from_barbara(self):
        db = BarbaraDB.open("test")
        db["/Config/max_position_notional"] = 1_000_000
        pl = PositionLimits(barbara=db)
        assert pl._limits["max_position_notional"] == 1_000_000
        db.close()


# ── OrderBook ───────────────────────────────────────────────────────────

class TestOrderBook:
    def test_submit_and_retrieve(self):
        ob = OrderBook()
        order = Order("O1", "Alice", "AAPL", "BUY", 100,
                       created_at=datetime.now().isoformat())
        ob.submit(order)
        assert ob.get_order("O1") is not None
        assert ob.get_order("O1").status == "PENDING"

    def test_fill_order(self):
        ob = OrderBook()
        order = Order("O1", "Alice", "AAPL", "BUY", 100,
                       created_at=datetime.now().isoformat())
        ob.submit(order)
        ob.fill("O1", 150.5, 1.25)
        filled = ob.get_order("O1")
        assert filled.status == "FILLED"
        assert filled.filled_price == 150.5

    def test_cancel_order(self):
        ob = OrderBook()
        order = Order("O1", "Alice", "AAPL", "BUY", 100,
                       created_at=datetime.now().isoformat())
        ob.submit(order)
        ob.cancel("O1")
        assert ob.get_order("O1").status == "CANCELLED"

    def test_reject_order(self):
        ob = OrderBook()
        order = Order("O1", "Alice", "AAPL", "BUY", 100,
                       created_at=datetime.now().isoformat())
        ob.submit(order)
        ob.reject("O1", "Over limit")
        assert ob.get_order("O1").status == "REJECTED"
        assert ob.get_order("O1").reject_reason == "Over limit"

    def test_get_open_orders(self):
        ob = OrderBook()
        o1 = Order("O1", "Alice", "AAPL", "BUY", 100, created_at="now")
        o2 = Order("O2", "Bob", "MSFT", "SELL", 50, created_at="now")
        ob.submit(o1)
        ob.submit(o2)
        ob.fill("O2", 400.0, 1.0)

        open_orders = ob.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].order_id == "O1"

    def test_get_open_orders_filtered(self):
        ob = OrderBook()
        o1 = Order("O1", "Alice", "AAPL", "BUY", 100, created_at="now")
        o2 = Order("O2", "Bob", "MSFT", "SELL", 50, created_at="now")
        ob.submit(o1)
        ob.submit(o2)

        alice_orders = ob.get_open_orders(trader="Alice")
        assert len(alice_orders) == 1

    def test_get_fills(self):
        ob = OrderBook()
        o1 = Order("O1", "Alice", "AAPL", "BUY", 100, created_at="now")
        ob.submit(o1)
        ob.fill("O1", 150.0, 1.0)

        fills = ob.get_fills()
        assert len(fills) == 1

    def test_barbara_persistence(self):
        db = BarbaraDB.open("test")
        ob = OrderBook(barbara=db)
        o1 = Order("O1", "Alice", "AAPL", "BUY", 100, created_at="now")
        ob.submit(o1)

        stored = db["/Trading/orders/O1"]
        assert stored["trader"] == "Alice"
        db.close()

    def test_table_property(self):
        ob = OrderBook()
        o1 = Order("O1", "Alice", "AAPL", "BUY", 100, created_at="now")
        ob.submit(o1)
        assert len(ob.table) == 1


# ── SignalGenerators ────────────────────────────────────────────────────

class TestMomentumSignal:
    def test_generate_signals(self):
        state = _make_state_with_history(["AAPL", "MSFT"], n_days=60)
        gen = MomentumSignal(tickers=["AAPL", "MSFT"])
        signals = gen.generate_signals(state)
        assert isinstance(signals, list)
        for sig in signals:
            assert sig.side in ("BUY", "SELL")
            assert 0.0 <= sig.strength <= 1.0

    def test_no_data_no_signals(self):
        state = _make_state_with_history(["AAPL"], n_days=5)
        gen = MomentumSignal(tickers=["AAPL"], long_window=30)
        signals = gen.generate_signals(state)
        assert signals == []

    def test_no_mgr(self):
        gen = MomentumSignal(tickers=["AAPL"])
        signals = gen.generate_signals({})
        assert signals == []


class TestStatArbSignal:
    def test_generate_signals(self):
        state = _make_state_with_history(["GOOGL", "AMZN"], n_days=100)
        gen = StatArbSignal(pairs=[("GOOGL", "AMZN")])
        signals = gen.generate_signals(state)
        assert isinstance(signals, list)

    def test_no_data(self):
        state = _make_state_with_history(["GOOGL"], n_days=5)
        gen = StatArbSignal(pairs=[("GOOGL", "AMZN")])
        signals = gen.generate_signals(state)
        assert signals == []


class TestMacroSignal:
    def test_high_rates(self):
        state = {"sofr_md": MarketData("SOFR", price=0.06), "eurusd_md": MarketData("EUR", price=1.08)}
        gen = MacroSignal()
        signals = gen.generate_signals(state)
        # Should suggest buying bonds
        bond_signals = [s for s in signals if "BOND" in s.instrument]
        assert len(bond_signals) > 0

    def test_weak_eur(self):
        state = {"sofr_md": MarketData("SOFR", price=0.04), "eurusd_md": MarketData("EUR", price=1.02)}
        gen = MacroSignal()
        signals = gen.generate_signals(state)
        eur_signals = [s for s in signals if "EURUSD" in s.instrument]
        assert len(eur_signals) > 0
        assert eur_signals[0].side == "BUY"

    def test_strong_eur(self):
        state = {"sofr_md": MarketData("SOFR", price=0.04), "eurusd_md": MarketData("EUR", price=1.15)}
        gen = MacroSignal()
        signals = gen.generate_signals(state)
        eur_signals = [s for s in signals if "EURUSD" in s.instrument]
        assert len(eur_signals) > 0
        assert eur_signals[0].side == "SELL"


# ── AuditTrail ──────────────────────────────────────────────────────────

class TestAuditTrail:
    def test_log_event(self):
        at = AuditTrail()
        at.log("ORDER_SUBMIT", "Alice", "AAPL", details={"qty": 100})
        rows = list(at.table)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "ORDER_SUBMIT"

    def test_query_by_trader(self):
        at = AuditTrail()
        at.log("FILL", "Alice", "AAPL")
        at.log("FILL", "Bob", "MSFT")
        results = list(at.query(trader="Alice"))
        assert len(results) == 1

    def test_query_by_event_type(self):
        at = AuditTrail()
        at.log("ORDER_SUBMIT", "Alice", "AAPL")
        at.log("FILL", "Alice", "AAPL")
        at.log("REJECT", "Bob", "MSFT")
        results = list(at.query(event_type="FILL"))
        assert len(results) == 1

    def test_barbara_persistence(self):
        db = BarbaraDB.open("test")
        at = AuditTrail(barbara=db)
        at.log("FILL", "Alice", "AAPL", details={"price": 150.0})
        keys = db.keys(prefix="/Trading/audit/")
        assert len(keys) >= 1
        db.close()

    def test_details_json(self):
        at = AuditTrail()
        at.log("FILL", "Alice", "AAPL", details={"price": 150.0, "qty": 100})
        rows = list(at.table)
        details = json.loads(rows[0]["details"])
        assert details["price"] == 150.0


# ── TradingEngine ───────────────────────────────────────────────────────

class TestTradingEngine:
    def _make_engine(self):
        db = BarbaraDB.open("test")
        graph = DependencyGraph()
        spot = MarketData("AAPL_SPOT", price=150.0)
        graph.register(spot)
        eq = Equity("AAPL", spot_source=spot)
        graph.register(eq)

        state = {
            "graph": graph,
            "spots": {"AAPL": spot},
            "equities": {"AAPL": eq},
            "traders": [],
        }

        engine = TradingEngine(barbara=db, graph=graph, state=state)
        return engine, db

    def test_submit_market_order(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")
        trader.book = Book("Alice")
        trader.book.add_position(Position(
            engine.state["equities"]["AAPL"], 100
        ))

        order = engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="BUY",
            quantity=50,
            current_price=150.0,
            book_value=500_000,  # large book to avoid concentration breach
        )

        assert order.status == "FILLED"
        assert order.filled_price > 0
        db.close()

    def test_submit_rejected_order(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")

        # Very large order should be rejected
        order = engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="BUY",
            quantity=100000,
        )

        assert order.status == "REJECTED"
        db.close()

    def test_submit_no_price(self):
        db = BarbaraDB.open("test")
        engine = TradingEngine(barbara=db, state={})
        trader = MockTrader("Alice")

        order = engine.submit_order(
            trader=trader,
            instrument_name="UNKNOWN",
            side="BUY",
            quantity=100,
        )
        assert order.status == "REJECTED"
        db.close()

    def test_submit_with_explicit_price(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")

        order = engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="BUY",
            quantity=100,
            current_price=150.0,
            book_value=100000,
        )
        assert order.status == "FILLED"
        db.close()

    def test_audit_trail_populated(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")

        engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="BUY",
            quantity=100,
            current_price=150.0,
            book_value=100000,
        )

        events = list(engine.audit_trail.table)
        # Should have ORDER_SUBMIT + FILL
        types = [e["event_type"] for e in events]
        assert "ORDER_SUBMIT" in types
        assert "FILL" in types
        db.close()

    def test_process_signals(self):
        state = _make_state_with_history(["AAPL", "MSFT"], n_days=60)
        db = BarbaraDB.open("test")
        engine = TradingEngine(barbara=db, state=state)
        engine.register_signal_generator("momentum", MomentumSignal(tickers=["AAPL"]))

        trader = MockTrader("Alice")
        state["traders"] = [trader]

        orders = engine.process_signals("Alice", "momentum", state)
        assert isinstance(orders, list)
        db.close()

    def test_register_signal_generator(self):
        engine, db = self._make_engine()
        gen = MomentumSignal(tickers=["AAPL"])
        engine.register_signal_generator("momentum", gen)
        assert "momentum" in engine._signal_generators
        db.close()

    def test_process_unknown_strategy(self):
        engine, db = self._make_engine()
        orders = engine.process_signals("Alice", "unknown_strategy")
        assert orders == []
        db.close()

    def test_limit_order(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")

        order = engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="BUY",
            quantity=100,
            order_type="LIMIT",
            limit_price=155.0,
            current_price=150.0,
            book_value=100000,
        )
        assert order.status == "FILLED"
        db.close()

    def test_sell_order(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")

        order = engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="SELL",
            quantity=100,
            current_price=150.0,
            book_value=100000,
        )
        assert order.status == "FILLED"
        db.close()

    def test_barbara_fill_stored(self):
        engine, db = self._make_engine()
        trader = MockTrader("Alice")

        engine.submit_order(
            trader=trader,
            instrument_name="AAPL",
            side="BUY",
            quantity=100,
            current_price=150.0,
            book_value=100000,
        )

        fill_keys = db.keys(prefix="/Trading/fills/")
        assert len(fill_keys) >= 1
        db.close()

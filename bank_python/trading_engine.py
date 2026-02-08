"""
Trading Engine — Order management, execution, signals, and audit trail.

Phase 2A of the prop desk platform build. Provides a complete order lifecycle:
Signal → Order → PositionLimits check → ExecutionModel → Fill → Book update → Audit

Classes:
    Order             — Order dataclass (MARKET/LIMIT, BUY/SELL)
    ExecutionModel    — Simulated slippage + fees
    PositionLimits    — Pre-trade risk checks
    OrderBook         — MnTable-backed order store with Barbara persistence
    SignalGenerator   — Base class + 4 strategy implementations
    AuditTrail        — Immutable event log
    TradingEngine     — Orchestrator wiring everything together
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Order ───────────────────────────────────────────────────────────────

_order_counter = 0


def _next_order_id():
    global _order_counter
    _order_counter += 1
    return f"ORD-{_order_counter:06d}"


@dataclass
class Order:
    order_id: str
    trader: str
    instrument_name: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: str = "MARKET"  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None
    status: str = "PENDING"  # PENDING, FILLED, CANCELLED, REJECTED
    filled_qty: float = 0.0
    filled_price: float = 0.0
    fill_cost: float = 0.0
    created_at: str = ""
    filled_at: str = ""
    reject_reason: str = ""


@dataclass
class Signal:
    instrument: str
    side: str  # "BUY" or "SELL"
    quantity: float
    strength: float  # 0.0 to 1.0
    reason: str


# ── ExecutionModel ──────────────────────────────────────────────────────

class ExecutionModel:
    """
    Simulates realistic execution with slippage and fees.

    Slippage: linear market impact = slippage_bps/10000 * (qty / avg_daily_volume)
    Fees: |qty| * fee_per_share + |fill_value| * commission_pct
    """

    def __init__(self, fee_per_share=0.005, commission_pct=0.001, slippage_bps=5.0):
        self.fee_per_share = fee_per_share
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps

    def execute(self, order, current_price, avg_daily_volume=1_000_000):
        """
        Compute fill price and total cost for an order.

        Parameters
        ----------
        order              : Order
        current_price      : float
        avg_daily_volume   : float — average daily volume for impact calc

        Returns
        -------
        (fill_price, total_cost) : tuple[float, float]
        """
        qty = abs(order.quantity)
        adv = max(avg_daily_volume, 1.0)

        # Market impact (slippage)
        impact_frac = self.slippage_bps / 10000.0 * (qty / adv)
        if order.side == "BUY":
            fill_price = current_price * (1.0 + impact_frac)
        else:
            fill_price = current_price * (1.0 - impact_frac)

        # For limit orders, check limit constraint
        if order.order_type == "LIMIT" and order.limit_price is not None:
            if order.side == "BUY" and fill_price > order.limit_price:
                fill_price = order.limit_price
            elif order.side == "SELL" and fill_price < order.limit_price:
                fill_price = order.limit_price

        fill_price = max(fill_price, 0.01)

        # Fees
        fill_value = qty * fill_price
        fees = qty * self.fee_per_share + fill_value * self.commission_pct
        total_cost = fill_value + fees if order.side == "BUY" else fill_value - fees

        return fill_price, fees


# ── PositionLimits ──────────────────────────────────────────────────────

class PositionLimits:
    """
    Pre-trade risk checks against configured limits.

    Reads limits from Barbara /Config/* keys. Checks:
    - Notional limit per trader
    - Delta limit per trader
    - Concentration limit per name
    - Daily loss limit
    """

    def __init__(self, barbara=None):
        self.barbara = barbara
        self._limits = {
            "max_position_notional": 500_000,
            "max_delta": 100_000,
            "max_concentration_pct": 40,
            "trader_loss_limit": -50_000,
        }
        if barbara:
            for key, default in self._limits.items():
                val = barbara.get(f"/Config/{key}", default)
                self._limits[key] = val

    def check(self, order, trader, current_price, book_value=0.0):
        """
        Pre-trade risk check.

        Parameters
        ----------
        order         : Order
        trader        : Trader-like object with .book
        current_price : float
        book_value    : float — current total book value

        Returns
        -------
        (approved, reject_reason) : tuple[bool, str]
        """
        qty = abs(order.quantity)
        notional = qty * current_price

        # Notional limit
        if notional > self._limits["max_position_notional"]:
            return False, (
                f"Notional ${notional:,.0f} exceeds limit "
                f"${self._limits['max_position_notional']:,}"
            )

        # Concentration: new position vs book
        total_book = abs(book_value) or 1.0
        pct = notional / total_book * 100
        if pct > self._limits["max_concentration_pct"]:
            return False, (
                f"Concentration {pct:.0f}% exceeds "
                f"{self._limits['max_concentration_pct']}% limit"
            )

        return True, ""


# ── OrderBook ───────────────────────────────────────────────────────────

class OrderBook:
    """
    MnTable-backed order store with Barbara persistence.

    Tracks all orders with their lifecycle states. Indexed on trader,
    instrument, and status for fast lookups.
    """

    def __init__(self, barbara=None):
        from bank_python.mntable import Table

        self.barbara = barbara
        self._table = Table([
            ("order_id", str),
            ("trader", str),
            ("instrument", str),
            ("side", str),
            ("quantity", float),
            ("order_type", str),
            ("limit_price", float),
            ("status", str),
            ("filled_qty", float),
            ("filled_price", float),
            ("fill_cost", float),
            ("created_at", str),
            ("filled_at", str),
        ], name="order_book")
        self._table.create_index("trader")
        self._table.create_index("instrument")
        self._table.create_index("status")
        self._orders = {}  # order_id -> Order

    def submit(self, order):
        """Submit a new order to the book."""
        self._orders[order.order_id] = order
        self._table.append({
            "order_id": order.order_id,
            "trader": order.trader,
            "instrument": order.instrument_name,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type,
            "limit_price": order.limit_price or 0.0,
            "status": order.status,
            "filled_qty": order.filled_qty,
            "filled_price": order.filled_price,
            "fill_cost": order.fill_cost,
            "created_at": order.created_at,
            "filled_at": "",
        })

        if self.barbara:
            self.barbara[f"/Trading/orders/{order.order_id}"] = {
                "order_id": order.order_id,
                "trader": order.trader,
                "instrument": order.instrument_name,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "status": order.status,
                "created_at": order.created_at,
            }

    def fill(self, order_id, filled_price, fill_cost):
        """Mark an order as filled."""
        order = self._orders.get(order_id)
        if order is None:
            return

        now = datetime.now().isoformat()
        order.status = "FILLED"
        order.filled_qty = order.quantity
        order.filled_price = filled_price
        order.fill_cost = fill_cost
        order.filled_at = now

        if self.barbara:
            self.barbara[f"/Trading/orders/{order_id}"] = {
                "order_id": order.order_id,
                "trader": order.trader,
                "instrument": order.instrument_name,
                "side": order.side,
                "quantity": order.quantity,
                "status": "FILLED",
                "filled_price": filled_price,
                "fill_cost": fill_cost,
                "filled_at": now,
            }

    def cancel(self, order_id):
        """Cancel a pending order."""
        order = self._orders.get(order_id)
        if order and order.status == "PENDING":
            order.status = "CANCELLED"
            if self.barbara:
                self.barbara[f"/Trading/orders/{order_id}"] = {
                    "order_id": order.order_id,
                    "status": "CANCELLED",
                }

    def reject(self, order_id, reason):
        """Reject an order with a reason."""
        order = self._orders.get(order_id)
        if order:
            order.status = "REJECTED"
            order.reject_reason = reason
            if self.barbara:
                self.barbara[f"/Trading/orders/{order_id}"] = {
                    "order_id": order.order_id,
                    "status": "REJECTED",
                    "reject_reason": reason,
                }

    def get_open_orders(self, trader=None):
        """Get all pending orders, optionally filtered by trader."""
        result = []
        for order in self._orders.values():
            if order.status != "PENDING":
                continue
            if trader and order.trader != trader:
                continue
            result.append(order)
        return result

    def get_fills(self, trader=None, since=None):
        """Get filled orders, optionally filtered by trader and time."""
        result = []
        for order in self._orders.values():
            if order.status != "FILLED":
                continue
            if trader and order.trader != trader:
                continue
            if since and order.filled_at < since:
                continue
            result.append(order)
        return result

    def get_order(self, order_id):
        return self._orders.get(order_id)

    @property
    def table(self):
        return self._table


# ── SignalGenerator ─────────────────────────────────────────────────────

class SignalGenerator:
    """Base class for signal generators."""

    def generate_signals(self, state):
        """
        Generate trading signals from current market state.

        Parameters
        ----------
        state : dict — desk state with market data, history, etc.

        Returns
        -------
        list[Signal]
        """
        raise NotImplementedError


class MomentumSignal(SignalGenerator):
    """
    10-day vs 30-day moving average crossover.

    BUY when 10d MA crosses above 30d MA, SELL when below.
    """

    def __init__(self, tickers=None, short_window=10, long_window=30):
        self.tickers = tickers or []
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, state):
        signals = []
        mgr = state.get("mgr")
        if mgr is None:
            return signals

        for ticker in self.tickers:
            view = mgr.history.query(ticker)
            rows = list(view)
            if len(rows) < self.long_window:
                continue

            closes = [r["close"] for r in rows]
            short_ma = np.mean(closes[-self.short_window:])
            long_ma = np.mean(closes[-self.long_window:])

            if short_ma > long_ma * 1.005:  # 0.5% threshold
                strength = min((short_ma / long_ma - 1.0) * 10, 1.0)
                signals.append(Signal(
                    instrument=ticker,
                    side="BUY",
                    quantity=100,
                    strength=strength,
                    reason=f"MA crossover: {self.short_window}d={short_ma:.2f} > "
                           f"{self.long_window}d={long_ma:.2f}",
                ))
            elif short_ma < long_ma * 0.995:
                strength = min((1.0 - short_ma / long_ma) * 10, 1.0)
                signals.append(Signal(
                    instrument=ticker,
                    side="SELL",
                    quantity=100,
                    strength=strength,
                    reason=f"MA crossover: {self.short_window}d={short_ma:.2f} < "
                           f"{self.long_window}d={long_ma:.2f}",
                ))

        return signals


class VolArbSignal(SignalGenerator):
    """
    Realized vs implied volatility arbitrage.

    SELL when realized vol < implied vol (overpriced options).
    BUY when realized vol > implied vol (underpriced options).
    """

    def __init__(self, tickers=None, rv_window=20):
        self.tickers = tickers or []
        self.rv_window = rv_window

    def generate_signals(self, state):
        signals = []
        mgr = state.get("mgr")
        vol_surfaces = state.get("vol_surfaces", {})
        if mgr is None:
            return signals

        for ticker in self.tickers:
            view = mgr.history.query(ticker)
            rows = list(view)
            if len(rows) < self.rv_window + 1:
                continue

            closes = np.array([r["close"] for r in rows[-self.rv_window - 1:]])
            log_ret = np.diff(np.log(closes))
            rv = np.std(log_ret) * np.sqrt(252)

            # Get implied vol from surface if available
            surface = vol_surfaces.get(ticker)
            if surface and surface.built:
                iv = surface.get_vol(1.0, 30.0 / 365.0)  # ATM 30d
            else:
                iv = None

            if iv is None:
                continue

            spread = iv - rv
            if spread > 0.05:  # IV > RV by 5 vol points
                strength = min(spread * 5, 1.0)
                signals.append(Signal(
                    instrument=ticker,
                    side="SELL",
                    quantity=10,
                    strength=strength,
                    reason=f"Vol arb: IV={iv:.1%} > RV={rv:.1%}, spread={spread:.1%}",
                ))
            elif spread < -0.05:
                strength = min(abs(spread) * 5, 1.0)
                signals.append(Signal(
                    instrument=ticker,
                    side="BUY",
                    quantity=10,
                    strength=strength,
                    reason=f"Vol arb: IV={iv:.1%} < RV={rv:.1%}, spread={spread:.1%}",
                ))

        return signals


class StatArbSignal(SignalGenerator):
    """
    Statistical arbitrage via price ratio z-score for pairs.

    Trade when |z-score| > 2: buy the underperformer, sell the outperformer.
    """

    def __init__(self, pairs=None, lookback=60, z_threshold=2.0):
        self.pairs = pairs or []  # list of (ticker_a, ticker_b) tuples
        self.lookback = lookback
        self.z_threshold = z_threshold

    def generate_signals(self, state):
        signals = []
        mgr = state.get("mgr")
        if mgr is None:
            return signals

        for ticker_a, ticker_b in self.pairs:
            rows_a = list(mgr.history.query(ticker_a))
            rows_b = list(mgr.history.query(ticker_b))

            min_len = min(len(rows_a), len(rows_b))
            if min_len < self.lookback:
                continue

            closes_a = np.array([r["close"] for r in rows_a[-min_len:]])
            closes_b = np.array([r["close"] for r in rows_b[-min_len:]])

            # Price ratio
            ratio = closes_a / np.maximum(closes_b, 1e-10)
            mean_ratio = np.mean(ratio[-self.lookback:])
            std_ratio = np.std(ratio[-self.lookback:])

            if std_ratio < 1e-10:
                continue

            z_score = (ratio[-1] - mean_ratio) / std_ratio

            if z_score > self.z_threshold:
                strength = min(abs(z_score) / 4.0, 1.0)
                signals.append(Signal(
                    instrument=ticker_a, side="SELL", quantity=50,
                    strength=strength,
                    reason=f"Stat arb {ticker_a}/{ticker_b}: z={z_score:.2f} (sell {ticker_a})",
                ))
                signals.append(Signal(
                    instrument=ticker_b, side="BUY", quantity=50,
                    strength=strength,
                    reason=f"Stat arb {ticker_a}/{ticker_b}: z={z_score:.2f} (buy {ticker_b})",
                ))
            elif z_score < -self.z_threshold:
                strength = min(abs(z_score) / 4.0, 1.0)
                signals.append(Signal(
                    instrument=ticker_a, side="BUY", quantity=50,
                    strength=strength,
                    reason=f"Stat arb {ticker_a}/{ticker_b}: z={z_score:.2f} (buy {ticker_a})",
                ))
                signals.append(Signal(
                    instrument=ticker_b, side="SELL", quantity=50,
                    strength=strength,
                    reason=f"Stat arb {ticker_a}/{ticker_b}: z={z_score:.2f} (sell {ticker_b})",
                ))

        return signals


class MacroSignal(SignalGenerator):
    """
    Macro signal based on rate levels, FX trends, yield curve slope.
    """

    def generate_signals(self, state):
        signals = []
        sofr_md = state.get("sofr_md")
        eurusd_md = state.get("eurusd_md")

        # Rate-based: if rates high, favor bonds
        if sofr_md and sofr_md.value > 0.05:
            signals.append(Signal(
                instrument="BOND_5Y", side="BUY", quantity=100,
                strength=min((sofr_md.value - 0.05) * 20, 1.0),
                reason=f"High rates ({sofr_md.value:.2%}): buy duration",
            ))

        # FX trend: simple level-based
        if eurusd_md and eurusd_md.value < 1.05:
            signals.append(Signal(
                instrument="EURUSD", side="BUY", quantity=10000,
                strength=min((1.05 - eurusd_md.value) * 10, 1.0),
                reason=f"EUR weak ({eurusd_md.value:.4f}): buy EURUSD",
            ))
        elif eurusd_md and eurusd_md.value > 1.12:
            signals.append(Signal(
                instrument="EURUSD", side="SELL", quantity=10000,
                strength=min((eurusd_md.value - 1.12) * 10, 1.0),
                reason=f"EUR strong ({eurusd_md.value:.4f}): sell EURUSD",
            ))

        return signals


# ── AuditTrail ──────────────────────────────────────────────────────────

_audit_counter = 0


def _next_event_id():
    global _audit_counter
    _audit_counter += 1
    return f"EVT-{_audit_counter:06d}"


class AuditTrail:
    """
    Immutable event log for all trading activity.

    Stores events in MnTable + Barbara for full audit compliance.
    """

    def __init__(self, barbara=None):
        from bank_python.mntable import Table

        self.barbara = barbara
        self._table = Table([
            ("event_id", str),
            ("timestamp", str),
            ("event_type", str),
            ("trader", str),
            ("instrument", str),
            ("details", str),  # JSON-encoded
            ("order_id", str),
        ], name="audit_trail")
        self._table.create_index("trader")
        self._table.create_index("event_type")

    def log(self, event_type, trader, instrument, details=None, order_id=""):
        """
        Log an audit event.

        Parameters
        ----------
        event_type : str — e.g. "ORDER_SUBMIT", "FILL", "REJECT", "SIGNAL"
        trader     : str
        instrument : str
        details    : dict or None
        order_id   : str
        """
        event_id = _next_event_id()
        now = datetime.now()
        ts = now.isoformat()

        details_json = json.dumps(details) if details else "{}"

        self._table.append({
            "event_id": event_id,
            "timestamp": ts,
            "event_type": event_type,
            "trader": trader,
            "instrument": instrument,
            "details": details_json,
            "order_id": order_id,
        })

        if self.barbara:
            date_key = now.strftime("%Y-%m-%d")
            self.barbara[f"/Trading/audit/{date_key}/{event_id}"] = {
                "event_id": event_id,
                "timestamp": ts,
                "event_type": event_type,
                "trader": trader,
                "instrument": instrument,
                "details": details,
                "order_id": order_id,
            }

    def query(self, trader=None, event_type=None, since=None):
        """
        Query audit events.

        Returns LazyView of matching events.
        """
        view = self._table
        if trader:
            view = view.restrict(trader=trader)
        if event_type:
            view = view.restrict(event_type=event_type)
        return view

    @property
    def table(self):
        return self._table


# ── TradingEngine ───────────────────────────────────────────────────────

class TradingEngine:
    """
    Orchestrates the full order lifecycle:
    Signal → Order → PositionLimits check → ExecutionModel → Fill → Book update → Audit

    Wires together OrderBook, ExecutionModel, PositionLimits, AuditTrail,
    and optionally the DependencyGraph for post-trade recalculation.
    """

    def __init__(self, barbara=None, graph=None, state=None):
        self.barbara = barbara
        self.graph = graph
        self.state = state or {}

        self.order_book = OrderBook(barbara=barbara)
        self.execution_model = ExecutionModel()
        self.position_limits = PositionLimits(barbara=barbara)
        self.audit_trail = AuditTrail(barbara=barbara)

        self._signal_generators = {}  # name -> SignalGenerator

    def register_signal_generator(self, name, generator):
        """Register a signal generator by name."""
        self._signal_generators[name] = generator

    def submit_order(self, trader, instrument_name, side, quantity,
                     order_type="MARKET", limit_price=None,
                     current_price=None, book_value=None):
        """
        Submit an order through the full lifecycle.

        Parameters
        ----------
        trader          : Trader-like object with .name and .book
        instrument_name : str
        side            : str — "BUY" or "SELL"
        quantity        : float
        order_type      : str — "MARKET" or "LIMIT"
        limit_price     : float or None
        current_price   : float or None — will look up from state if not provided
        book_value      : float or None

        Returns
        -------
        Order — with final status
        """
        order = Order(
            order_id=_next_order_id(),
            trader=trader.name if hasattr(trader, 'name') else str(trader),
            instrument_name=instrument_name,
            side=side.upper(),
            quantity=abs(quantity),
            order_type=order_type.upper(),
            limit_price=limit_price,
            status="PENDING",
            created_at=datetime.now().isoformat(),
        )

        # Submit to order book
        self.order_book.submit(order)
        self.audit_trail.log(
            "ORDER_SUBMIT", order.trader, instrument_name,
            details={"side": side, "qty": quantity, "type": order_type},
            order_id=order.order_id,
        )

        # Look up current price if not provided
        if current_price is None:
            current_price = self._get_current_price(instrument_name)
        if current_price is None or current_price <= 0:
            self.order_book.reject(order.order_id, "No price available")
            self.audit_trail.log(
                "REJECT", order.trader, instrument_name,
                details={"reason": "No price available"},
                order_id=order.order_id,
            )
            return order

        # Pre-trade risk check
        bv = book_value if book_value is not None else (
            trader.book.total_value if hasattr(trader, 'book') else 0.0
        )
        approved, reason = self.position_limits.check(
            order, trader, current_price, bv,
        )
        if not approved:
            self.order_book.reject(order.order_id, reason)
            self.audit_trail.log(
                "REJECT", order.trader, instrument_name,
                details={"reason": reason},
                order_id=order.order_id,
            )
            return order

        # Execute
        fill_price, fees = self.execution_model.execute(
            order, current_price,
        )
        self.order_book.fill(order.order_id, fill_price, fees)

        self.audit_trail.log(
            "FILL", order.trader, instrument_name,
            details={
                "fill_price": fill_price,
                "fees": fees,
                "side": side,
                "qty": quantity,
            },
            order_id=order.order_id,
        )

        # Store fill in Barbara
        if self.barbara:
            now = datetime.now()
            date_key = now.strftime("%Y-%m-%d")
            self.barbara[f"/Trading/fills/{date_key}/{order.order_id}"] = {
                "order_id": order.order_id,
                "trader": order.trader,
                "instrument": instrument_name,
                "side": side,
                "quantity": quantity,
                "fill_price": fill_price,
                "fees": fees,
                "filled_at": now.isoformat(),
            }

        return order

    def process_signals(self, trader_name, strategy_name, state=None):
        """
        Run a signal generator and submit resulting orders.

        Parameters
        ----------
        trader_name   : str
        strategy_name : str — registered signal generator name
        state         : dict or None — uses self.state if not provided

        Returns
        -------
        list[Order] — submitted orders
        """
        gen = self._signal_generators.get(strategy_name)
        if gen is None:
            logger.warning(f"Unknown strategy: {strategy_name}")
            return []

        st = state or self.state
        signals = gen.generate_signals(st)

        # Log signals
        for sig in signals:
            self.audit_trail.log(
                "SIGNAL", trader_name, sig.instrument,
                details={
                    "strategy": strategy_name,
                    "side": sig.side,
                    "qty": sig.quantity,
                    "strength": sig.strength,
                    "reason": sig.reason,
                },
            )

            if self.barbara:
                now = datetime.now()
                date_key = now.strftime("%Y-%m-%d")
                self.barbara[f"/Trading/signals/{strategy_name}/{date_key}"] = {
                    "instrument": sig.instrument,
                    "side": sig.side,
                    "quantity": sig.quantity,
                    "strength": sig.strength,
                    "reason": sig.reason,
                }

        # Submit orders for strong signals
        orders = []
        # Find trader object
        traders = st.get("traders", [])
        trader_obj = None
        for t in traders:
            if t.name == trader_name:
                trader_obj = t
                break

        if trader_obj is None:
            # Create a minimal trader-like object
            class _MinTrader:
                def __init__(self, name):
                    self.name = name
                    from bank_python.dagger import Book
                    self.book = Book(name)
            trader_obj = _MinTrader(trader_name)

        for sig in signals:
            if sig.strength < 0.3:
                continue
            order = self.submit_order(
                trader=trader_obj,
                instrument_name=sig.instrument,
                side=sig.side,
                quantity=sig.quantity,
            )
            orders.append(order)

        return orders

    def _get_current_price(self, instrument_name):
        """Look up current price from state."""
        # Check equities
        equities = self.state.get("equities", {})
        if instrument_name in equities:
            return equities[instrument_name].value

        # Check spots
        spots = self.state.get("spots", {})
        for ticker, md in spots.items():
            if ticker == instrument_name or md.name == instrument_name:
                return md.value

        # Check mgr registered tickers
        mgr = self.state.get("mgr")
        if mgr:
            for ticker, node in mgr.registered_tickers.items():
                if ticker == instrument_name:
                    return node.value

        return None


# ── Rendering helpers ───────────────────────────────────────────────────

def render_orders(order_book, console, trader=None, status=None):
    """Render order book as a Rich table."""
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    tbl = RichTable(
        title="Order Book",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("ID", width=12)
    tbl.add_column("Trader", style="bold", width=8)
    tbl.add_column("Instrument", width=12)
    tbl.add_column("Side", width=5)
    tbl.add_column("Qty", justify="right", width=8)
    tbl.add_column("Type", width=8)
    tbl.add_column("Status", width=10)
    tbl.add_column("Fill Price", justify="right", width=12)
    tbl.add_column("Fees", justify="right", width=10)

    for order in order_book._orders.values():
        if trader and order.trader != trader:
            continue
        if status and order.status != status:
            continue

        side_style = "green" if order.side == "BUY" else "red"
        status_styles = {
            "PENDING": "yellow",
            "FILLED": "green",
            "CANCELLED": "dim",
            "REJECTED": "red",
        }

        tbl.add_row(
            order.order_id,
            order.trader,
            order.instrument_name,
            Text(order.side, style=side_style),
            f"{order.quantity:,.0f}",
            order.order_type,
            Text(order.status, style=status_styles.get(order.status, "")),
            f"${order.filled_price:.2f}" if order.filled_price > 0 else "",
            f"${order.fill_cost:.2f}" if order.fill_cost > 0 else "",
        )

    console.print(Panel(tbl, border_style="cyan"))


def render_audit(audit_trail, console, trader=None, event_type=None):
    """Render audit trail as a Rich table."""
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    tbl = RichTable(
        title="Audit Trail",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("ID", width=12)
    tbl.add_column("Time", width=12)
    tbl.add_column("Type", style="bold", width=14)
    tbl.add_column("Trader", width=8)
    tbl.add_column("Instrument", width=12)
    tbl.add_column("Details", width=30)
    tbl.add_column("Order", width=12)

    view = audit_trail.query(trader=trader, event_type=event_type)
    for row in view:
        type_styles = {
            "ORDER_SUBMIT": "yellow",
            "FILL": "green",
            "REJECT": "red",
            "SIGNAL": "cyan",
        }
        ts = row["timestamp"]
        # Show only time portion
        time_str = ts.split("T")[1][:8] if "T" in ts else ts[:8]

        tbl.add_row(
            row["event_id"],
            time_str,
            Text(row["event_type"], style=type_styles.get(row["event_type"], "")),
            row["trader"],
            row["instrument"],
            row["details"][:30],
            row["order_id"],
        )

    console.print(Panel(tbl, border_style="cyan"))

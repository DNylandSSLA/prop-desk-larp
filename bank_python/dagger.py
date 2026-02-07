"""
Dagger — Reactive dependency graph for instrument pricing.

Inspired by the DAG-based reactive computation engines inside investment banks.
When a leaf node (e.g. a market data point) changes, Dagger propagates "dirty"
flags through the graph and recalculates all affected derived instruments.

Key concepts:
- Instrument: base class with underliers and a compute() method
- MarketData: leaf node representing an observable market price
- Bond, CreditDefaultSwap, Option: derived instruments
- Position: instrument + quantity
- Book: collection of positions
- DependencyGraph: tracks relationships, handles dirty propagation via BFS
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import logging

logger = logging.getLogger(__name__)


class Instrument(ABC):
    """Base class for all instruments in the dependency graph."""

    def __init__(self, name):
        self.name = name
        self._value = None
        self._dirty = True

    @property
    def underliers(self):
        """Return list of Instrument objects this instrument depends on."""
        return []

    @abstractmethod
    def compute(self):
        """Recompute this instrument's value from its underliers."""

    @property
    def value(self):
        if self._dirty:
            self.compute()
            self._dirty = False
        return self._value

    def mark_dirty(self):
        self._dirty = True

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, value={self._value})"


class MarketData(Instrument):
    """Leaf node representing an observable market price or rate."""

    def __init__(self, name, price=0.0):
        super().__init__(name)
        self._value = price
        self._dirty = False

    @property
    def underliers(self):
        return []

    def compute(self):
        pass  # leaf node — value is set externally

    def set_price(self, price):
        old = self._value
        self._value = price
        self._dirty = False
        logger.info(f"MarketData {self.name}: {old} -> {price}")


class Bond(Instrument):
    """
    Simplified bond priced off a reference rate.
    PV = sum of coupon/(1+rate)^t + face/(1+rate)^maturity
    """

    def __init__(self, name, rate_source, face=100.0, coupon_rate=0.05, maturity=5):
        super().__init__(name)
        self.rate_source = rate_source
        self.face = face
        self.coupon_rate = coupon_rate
        self.maturity = maturity

    @property
    def underliers(self):
        return [self.rate_source]

    def compute(self):
        rate = self.rate_source.value
        coupon = self.face * self.coupon_rate
        pv = sum(coupon / (1 + rate) ** t for t in range(1, self.maturity + 1))
        pv += self.face / (1 + rate) ** self.maturity
        self._value = round(pv, 4)


class CreditDefaultSwap(Instrument):
    """
    Simplified CDS: spread based on credit spread and reference rate.
    Value approximation: notional * (credit_spread - base_rate) * maturity * 0.01
    """

    def __init__(self, name, credit_spread_source, rate_source, notional=10_000_000, maturity=5):
        super().__init__(name)
        self.credit_spread_source = credit_spread_source
        self.rate_source = rate_source
        self.notional = notional
        self.maturity = maturity

    @property
    def underliers(self):
        return [self.credit_spread_source, self.rate_source]

    def compute(self):
        spread = self.credit_spread_source.value
        rate = self.rate_source.value
        self._value = round(
            self.notional * (spread - rate) * self.maturity * 0.01, 2
        )


class Option(Instrument):
    """
    Simplified Black-Scholes-ish option pricing.
    Uses a very simplified model for demonstration purposes.
    """

    def __init__(self, name, spot_source, strike=100.0, volatility=0.2,
                 time_to_expiry=1.0, is_call=True):
        super().__init__(name)
        self.spot_source = spot_source
        self.strike = strike
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.is_call = is_call

    @property
    def underliers(self):
        return [self.spot_source]

    def compute(self):
        S = self.spot_source.value
        K = self.strike
        sigma = self.volatility
        T = self.time_to_expiry

        if T <= 0 or sigma <= 0:
            self._value = max(0, S - K) if self.is_call else max(0, K - S)
            return

        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)

        if self.is_call:
            self._value = round(S * nd1 - K * nd2, 4)
        else:
            self._value = round(K * (1 - nd2) - S * (1 - nd1), 4)


def _norm_cdf(x):
    """Approximation of the cumulative normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class Position:
    """An instrument held in a specific quantity."""

    def __init__(self, instrument, quantity):
        self.instrument = instrument
        self.quantity = quantity

    @property
    def market_value(self):
        return self.instrument.value * self.quantity

    def __repr__(self):
        return f"Position({self.instrument.name}, qty={self.quantity}, mv={self.market_value:.2f})"


class Book:
    """A collection of positions (a trading book)."""

    def __init__(self, name):
        self.name = name
        self.positions = []

    def add_position(self, position):
        self.positions.append(position)

    @property
    def total_value(self):
        return sum(p.market_value for p in self.positions)

    def summary(self):
        lines = [f"Book: {self.name}"]
        for p in self.positions:
            lines.append(f"  {p.instrument.name}: qty={p.quantity}, "
                         f"price={p.instrument.value:.4f}, "
                         f"mv={p.market_value:.2f}")
        lines.append(f"  Total: {self.total_value:.2f}")
        return "\n".join(lines)

    def __repr__(self):
        return f"Book({self.name!r}, positions={len(self.positions)}, total={self.total_value:.2f})"


class CycleError(Exception):
    """Raised when a cycle is detected in the dependency graph."""


class DependencyGraph:
    """
    Tracks the full dependency graph of instruments.
    Handles BFS dirty propagation and recalculation.
    """

    def __init__(self):
        self._nodes = {}  # name -> Instrument
        self._dependents = defaultdict(set)  # name -> set of dependent names

    def register(self, instrument):
        """Register an instrument and its dependency edges."""
        name = instrument.name
        self._nodes[name] = instrument
        for underlier in instrument.underliers:
            if underlier.name not in self._nodes:
                self.register(underlier)
            self._dependents[underlier.name].add(name)

        # Cycle detection
        if self._has_cycle():
            # Rollback
            del self._nodes[name]
            for underlier in instrument.underliers:
                self._dependents[underlier.name].discard(name)
            raise CycleError(f"Registering {name!r} would create a cycle")

    def _has_cycle(self):
        """DFS-based cycle detection."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in self._nodes}

        def dfs(node):
            color[node] = GRAY
            for dep in self._dependents.get(node, set()):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def recalculate(self, source):
        """
        After a source node changes, propagate dirty flags via BFS
        and recompute all affected instruments in topological order.
        """
        # BFS to find all affected nodes
        affected = set()
        queue = deque()
        for dep in self._dependents.get(source.name, set()):
            queue.append(dep)

        while queue:
            name = queue.popleft()
            if name in affected:
                continue
            affected.add(name)
            self._nodes[name].mark_dirty()
            for dep in self._dependents.get(name, set()):
                queue.append(dep)

        # Recompute in topological order (underliers before dependents)
        ordered = self._topo_sort(affected)
        for name in ordered:
            node = self._nodes[name]
            _ = node.value  # triggers compute if dirty
            logger.info(f"Recalculated {name}: {node.value}")

        return ordered

    def _topo_sort(self, names):
        """Topological sort of a subset of nodes."""
        # Build in-degree map restricted to affected set
        in_degree = {n: 0 for n in names}
        for n in names:
            inst = self._nodes[n]
            for u in inst.underliers:
                if u.name in names:
                    in_degree[n] += 1

        queue = deque(n for n, d in in_degree.items() if d == 0)
        result = []
        while queue:
            n = queue.popleft()
            result.append(n)
            for dep in self._dependents.get(n, set()):
                if dep in in_degree:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        return result

    @property
    def nodes(self):
        return dict(self._nodes)

    def __repr__(self):
        return f"DependencyGraph(nodes={len(self._nodes)})"

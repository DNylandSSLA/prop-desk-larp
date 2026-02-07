"""
Market Data Layer — yfinance integration for live market data.

Provides real equity prices, FX rates, and historical OHLCV data via Yahoo
Finance, with Barbara-backed caching and MnTable-based historical storage.
SOFR rates and credit spreads stay simulated (not available on Yahoo Finance).

Components:
- MarketDataSnapshot: dataclass holding a fetched price point
- MarketDataFetcher: wraps yfinance for batch downloads
- MarketDataCache: Barbara-backed cache with TTL
- HistoricalStore: MnTable for OHLCV time series
- Equity: Instrument subclass for equity spot prices
- FXRate: Instrument subclass for FX currency pairs
- MarketDataManager: orchestration layer tying everything together
"""

from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

import yfinance as yf

from .dagger import Instrument, MarketData, DependencyGraph
from .barbara import BarbaraDB
from .mntable import Table

logger = logging.getLogger(__name__)


# ── MarketDataSnapshot ────────────────────────────────────────────────────

@dataclass
class MarketDataSnapshot:
    """A single fetched price point from a market data source."""
    ticker: str
    price: float
    timestamp: datetime
    source: str = "yfinance"
    stale: bool = False
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        stale_tag = " [STALE]" if self.stale else ""
        return (f"MarketDataSnapshot({self.ticker}, "
                f"price={self.price:.4f}, "
                f"time={self.timestamp:%Y-%m-%d %H:%M}{stale_tag})")


# ── MarketDataFetcher ─────────────────────────────────────────────────────

class MarketDataFetcher:
    """Wraps yfinance for batch downloads with graceful error handling."""

    def __init__(self, max_retries=2, backoff_factor=1.0):
        self._max_retries = max_retries
        self._backoff_factor = backoff_factor

    def fetch_batch(self, tickers):
        """
        Fetch current prices for a list of tickers.

        Returns dict of ticker -> MarketDataSnapshot.
        Invalid tickers are logged and skipped, network errors return {}.
        """

        results = {}
        if not tickers:
            return results

        for attempt in range(self._max_retries + 1):
            try:
                data = yf.download(
                    tickers,
                    period="1d",
                    progress=False,
                    threads=False,
                )
                now = datetime.now()

                if len(tickers) == 1:
                    ticker = tickers[0]
                    if data.empty:
                        logger.warning(f"No data returned for {ticker}")
                    else:
                        close = float(data["Close"].iloc[-1])
                        meta = {}
                        for col in ("Open", "High", "Low", "Close", "Volume"):
                            if col in data.columns:
                                val = data[col].iloc[-1]
                                meta[col.lower()] = float(val) if col != "Volume" else int(val)
                        results[ticker] = MarketDataSnapshot(
                            ticker=ticker, price=close,
                            timestamp=now, metadata=meta,
                        )
                else:
                    for ticker in tickers:
                        try:
                            if ticker not in data["Close"].columns:
                                logger.warning(f"Ticker {ticker} not in results, skipping")
                                continue
                            close_val = data["Close"][ticker].iloc[-1]
                            if close_val != close_val:  # NaN check
                                logger.warning(f"NaN price for {ticker}, skipping")
                                continue
                            meta = {}
                            for col in ("Open", "High", "Low", "Close", "Volume"):
                                if col in data.columns:
                                    val = data[col][ticker].iloc[-1]
                                    if val == val:  # not NaN
                                        meta[col.lower()] = float(val) if col != "Volume" else int(val)
                            results[ticker] = MarketDataSnapshot(
                                ticker=ticker, price=float(close_val),
                                timestamp=now, metadata=meta,
                            )
                        except (KeyError, IndexError) as e:
                            logger.warning(f"Failed to parse {ticker}: {e}")

                return results

            except Exception as e:
                if attempt < self._max_retries:
                    wait = self._backoff_factor * (2 ** attempt)
                    logger.warning(f"Fetch attempt {attempt + 1} failed: {e}, "
                                   f"retrying in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    logger.error(f"All fetch attempts failed: {e}")
                    return results

        return results

    def fetch_historical(self, ticker, period="1mo"):
        """
        Fetch historical OHLCV data for a single ticker.

        Returns list of dicts with keys: timestamp, open, high, low, close, volume.
        Returns [] on error.
        """

        try:
            data = yf.download(ticker, period=period, progress=False, threads=False)
            if data.empty:
                logger.warning(f"No historical data for {ticker}")
                return []

            rows = []
            for ts, row in data.iterrows():
                rows.append({
                    "timestamp": ts.to_pydatetime().isoformat(),
                    "open": float(row["Open"].iloc[0]) if hasattr(row["Open"], "iloc") else float(row["Open"]),
                    "high": float(row["High"].iloc[0]) if hasattr(row["High"], "iloc") else float(row["High"]),
                    "low": float(row["Low"].iloc[0]) if hasattr(row["Low"], "iloc") else float(row["Low"]),
                    "close": float(row["Close"].iloc[0]) if hasattr(row["Close"], "iloc") else float(row["Close"]),
                    "volume": int(row["Volume"].iloc[0]) if hasattr(row["Volume"], "iloc") else int(row["Volume"]),
                })
            return rows

        except Exception as e:
            logger.error(f"Historical fetch failed for {ticker}: {e}")
            return []


# ── MarketDataCache ───────────────────────────────────────────────────────

class MarketDataCache:
    """Barbara-backed cache for MarketDataSnapshots with TTL."""

    def __init__(self, barbara, ttl=60.0):
        """
        barbara: BarbaraDB instance
        ttl: time-to-live in seconds (default 60)
        """
        self._db = barbara
        self._ttl = ttl

    def _key(self, ticker):
        return f"/MarketData/Cache/{ticker}/latest"

    def get(self, ticker):
        """Return cached snapshot if within TTL, else None."""
        snap = self._db.get(self._key(ticker))
        if snap is None:
            return None
        elapsed = (datetime.now() - snap.timestamp).total_seconds()
        if elapsed > self._ttl:
            snap.stale = True
            return None
        return snap

    def put(self, snapshot):
        """Store a snapshot in the cache."""
        self._db.put(
            self._key(snapshot.ticker),
            snapshot,
            metadata={"cached_at": time.time(), "price": snapshot.price},
        )

    def invalidate(self, ticker):
        """Remove a ticker from the cache."""
        key = self._key(ticker)
        if key in self._db:
            del self._db[key]


# ── HistoricalStore ───────────────────────────────────────────────────────

class HistoricalStore:
    """MnTable-backed OHLCV time series store."""

    SCHEMA = [
        ("ticker", str),
        ("timestamp", str),
        ("open", float),
        ("high", float),
        ("low", float),
        ("close", float),
        ("volume", int),
        ("source", str),
    ]

    def __init__(self):
        self._table = Table(self.SCHEMA, name="historical_ohlcv")
        self._table.create_index("ticker")
        self._table.create_index("timestamp")

    def append(self, ticker, rows, source="yfinance"):
        """Append OHLCV rows for a ticker."""
        for row in rows:
            self._table.append({
                "ticker": ticker,
                "timestamp": row["timestamp"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "source": source,
            })

    def query(self, ticker):
        """Return a LazyView of all rows for a ticker."""
        return self._table.restrict(ticker=ticker)

    def query_range(self, ticker, start, end):
        """Return a LazyView of rows for a ticker within a date range."""
        view = self._table.restrict(ticker=ticker)
        # Filter using LazyView chaining — timestamp is stored as ISO string
        # so lexicographic comparison works
        sql = (f'SELECT * FROM ({view._sql}) '
               f'WHERE "timestamp" >= ? AND "timestamp" <= ?')
        params = view._params + (start, end)
        from .mntable import LazyView
        return LazyView(self._table, sql, params)

    @property
    def table(self):
        return self._table

    def __repr__(self):
        return f"HistoricalStore(rows={len(self._table)})"


# ── Equity ────────────────────────────────────────────────────────────────

class Equity(Instrument):
    """
    Equity instrument priced from a MarketData spot source.

    Value is the spot price, optionally adjusted by dividend yield for
    forward pricing.
    """

    def __init__(self, name, spot_source, dividend_yield=0.0):
        super().__init__(name)
        self.spot_source = spot_source
        self.dividend_yield = dividend_yield

    @property
    def underliers(self):
        return [self.spot_source]

    def compute(self):
        self._value = self.spot_source.value

    @property
    def dividend_adjusted_price(self):
        """Spot price adjusted for dividend yield (simple annual adjustment)."""
        return self.value * (1 - self.dividend_yield)


# ── FXRate ────────────────────────────────────────────────────────────────

class FXRate(Instrument):
    """
    FX currency pair priced from a MarketData rate source.

    Convention: base/quote, e.g. EUR/USD = 1.08 means 1 EUR = 1.08 USD.
    """

    def __init__(self, name, rate_source, base_currency="EUR", quote_currency="USD"):
        super().__init__(name)
        self.rate_source = rate_source
        self.base_currency = base_currency
        self.quote_currency = quote_currency

    @property
    def underliers(self):
        return [self.rate_source]

    def compute(self):
        self._value = self.rate_source.value

    def convert(self, amount, from_currency=None):
        """
        Convert an amount between base and quote currencies.

        If from_currency is base (or None), multiply by rate (base -> quote).
        If from_currency is quote, divide by rate (quote -> base).
        """
        rate = self.value
        if from_currency is None or from_currency == self.base_currency:
            return amount * rate
        elif from_currency == self.quote_currency:
            return amount / rate if rate != 0 else 0.0
        else:
            raise ValueError(f"Unknown currency {from_currency} for pair "
                             f"{self.base_currency}/{self.quote_currency}")

    @property
    def inverse_rate(self):
        """The inverse rate (quote/base)."""
        return 1.0 / self.value if self.value != 0 else 0.0


# ── MarketDataManager ────────────────────────────────────────────────────

class MarketDataManager:
    """
    Orchestration layer for market data.

    Links Yahoo Finance tickers to Dagger MarketData nodes, handles fetching,
    caching, historical storage, and graph recalculation.
    """

    def __init__(self, graph, barbara, cache_ttl=60.0):
        self._graph = graph
        self._fetcher = MarketDataFetcher()
        self._cache = MarketDataCache(barbara, ttl=cache_ttl)
        self._history = HistoricalStore()
        self._registrations = {}  # ticker -> MarketData node

    @property
    def fetcher(self):
        return self._fetcher

    @property
    def cache(self):
        return self._cache

    @property
    def history(self):
        return self._history

    def register_ticker(self, ticker, market_data_node):
        """Link a Yahoo Finance ticker symbol to a Dagger MarketData node."""
        self._registrations[ticker] = market_data_node
        logger.info(f"Registered {ticker} -> {market_data_node.name}")

    def update_all(self):
        """
        Fetch all registered tickers, update cache, append history,
        and trigger graph recalculation.

        Returns dict of ticker -> new price (or None if fetch failed).
        """
        tickers = list(self._registrations.keys())
        if not tickers:
            return {}

        snapshots = self._fetcher.fetch_batch(tickers)
        results = {}

        for ticker, snap in snapshots.items():
            self._cache.put(snap)

            # Append to historical store
            self._history.append(ticker, [{
                "timestamp": snap.timestamp.isoformat(),
                "open": snap.metadata.get("open", snap.price),
                "high": snap.metadata.get("high", snap.price),
                "low": snap.metadata.get("low", snap.price),
                "close": snap.price,
                "volume": snap.metadata.get("volume", 0),
            }])

            # Update the Dagger node and recalculate
            node = self._registrations[ticker]
            node.set_price(snap.price)
            self._graph.recalculate(node)
            results[ticker] = snap.price
            logger.info(f"Updated {ticker}: {snap.price:.4f}")

        for ticker in tickers:
            if ticker not in snapshots:
                results[ticker] = None
                logger.warning(f"No update for {ticker}")

        return results

    def load_history(self, ticker, period="1mo"):
        """Fetch and store historical data for a ticker."""
        rows = self._fetcher.fetch_historical(ticker, period=period)
        if rows:
            self._history.append(ticker, rows)
            logger.info(f"Loaded {len(rows)} historical rows for {ticker}")
        return rows

    @property
    def registered_tickers(self):
        return dict(self._registrations)

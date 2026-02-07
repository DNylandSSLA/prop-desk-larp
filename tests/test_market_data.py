"""Tests for the market data layer — all yfinance calls are mocked."""

import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from bank_python import (
    BarbaraDB,
    DependencyGraph,
    MarketData,
    Bond,
    Option,
)
from bank_python.market_data import (
    MarketDataSnapshot,
    MarketDataFetcher,
    MarketDataCache,
    HistoricalStore,
    Equity,
    FXRate,
    MarketDataManager,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_download_df(tickers_prices, multi=False):
    """Build a DataFrame mimicking yfinance.download() output."""
    if not multi:
        ticker, price = list(tickers_prices.items())[0]
        return pd.DataFrame({
            "Open": [price * 0.99],
            "High": [price * 1.01],
            "Low": [price * 0.98],
            "Close": [price],
            "Volume": [1_000_000],
        })
    else:
        data = {}
        for col in ("Open", "High", "Low", "Close", "Volume"):
            col_data = {}
            for ticker, price in tickers_prices.items():
                if col == "Open":
                    col_data[ticker] = [price * 0.99]
                elif col == "High":
                    col_data[ticker] = [price * 1.01]
                elif col == "Low":
                    col_data[ticker] = [price * 0.98]
                elif col == "Close":
                    col_data[ticker] = [price]
                elif col == "Volume":
                    col_data[ticker] = [1_000_000]
            data[col] = pd.DataFrame(col_data)

        return pd.concat(data, axis=1)


def _make_historical_df(n_days=5, base_price=150.0):
    """Build a DataFrame mimicking historical yfinance.download() output."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")
    return pd.DataFrame({
        "Open": [base_price + i for i in range(n_days)],
        "High": [base_price + i + 2 for i in range(n_days)],
        "Low": [base_price + i - 2 for i in range(n_days)],
        "Close": [base_price + i + 1 for i in range(n_days)],
        "Volume": [1_000_000 + i * 100_000 for i in range(n_days)],
    }, index=dates)


# ── TestMarketDataFetcher ─────────────────────────────────────────────────

class TestMarketDataFetcher:

    @patch("bank_python.market_data.yf.download")
    def test_fetch_batch_single_ticker(self, mock_download):
        mock_download.return_value = _make_download_df({"AAPL": 175.50})

        fetcher = MarketDataFetcher()
        results = fetcher.fetch_batch(["AAPL"])

        assert "AAPL" in results
        snap = results["AAPL"]
        assert isinstance(snap, MarketDataSnapshot)
        assert snap.price == 175.50
        assert snap.ticker == "AAPL"
        assert snap.source == "yfinance"
        assert "close" in snap.metadata

    @patch("bank_python.market_data.yf.download")
    def test_fetch_batch_multiple_tickers(self, mock_download):
        mock_download.return_value = _make_download_df(
            {"AAPL": 175.50, "MSFT": 380.00}, multi=True
        )

        fetcher = MarketDataFetcher()
        results = fetcher.fetch_batch(["AAPL", "MSFT"])

        assert "AAPL" in results
        assert "MSFT" in results
        assert results["AAPL"].price == 175.50
        assert results["MSFT"].price == 380.00

    @patch("bank_python.market_data.yf.download")
    def test_fetch_batch_empty_tickers(self, mock_download):
        fetcher = MarketDataFetcher()
        results = fetcher.fetch_batch([])
        assert results == {}
        mock_download.assert_not_called()

    @patch("bank_python.market_data.yf.download")
    def test_fetch_batch_invalid_ticker(self, mock_download):
        mock_download.return_value = _make_download_df(
            {"AAPL": 175.50, "INVALID123": float("nan")}, multi=True
        )

        fetcher = MarketDataFetcher()
        results = fetcher.fetch_batch(["AAPL", "INVALID123"])

        assert "AAPL" in results
        assert "INVALID123" not in results  # NaN price skipped

    @patch("bank_python.market_data.yf.download")
    def test_fetch_batch_network_error(self, mock_download):
        mock_download.side_effect = ConnectionError("Network down")

        fetcher = MarketDataFetcher(max_retries=0)
        results = fetcher.fetch_batch(["AAPL"])

        assert results == {}

    @patch("bank_python.market_data.yf.download")
    def test_fetch_historical_success(self, mock_download):
        mock_download.return_value = _make_historical_df(n_days=5, base_price=150.0)

        fetcher = MarketDataFetcher()
        rows = fetcher.fetch_historical("AAPL", period="5d")

        assert len(rows) == 5
        assert "timestamp" in rows[0]
        assert "open" in rows[0]
        assert "close" in rows[0]
        assert rows[0]["close"] == 151.0

    @patch("bank_python.market_data.yf.download")
    def test_fetch_historical_empty(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        fetcher = MarketDataFetcher()
        rows = fetcher.fetch_historical("INVALID")

        assert rows == []


# ── TestMarketDataCache ───────────────────────────────────────────────────

class TestMarketDataCache:

    def test_cache_hit_within_ttl(self):
        db = BarbaraDB.open("test")
        cache = MarketDataCache(db, ttl=60.0)

        snap = MarketDataSnapshot(
            ticker="AAPL", price=175.50,
            timestamp=datetime.now(),
        )
        cache.put(snap)

        result = cache.get("AAPL")
        assert result is not None
        assert result.price == 175.50
        db.close()

    def test_cache_miss_after_ttl(self):
        db = BarbaraDB.open("test")
        cache = MarketDataCache(db, ttl=1.0)

        snap = MarketDataSnapshot(
            ticker="AAPL", price=175.50,
            timestamp=datetime.now() - timedelta(seconds=5),
        )
        cache.put(snap)

        result = cache.get("AAPL")
        assert result is None
        db.close()

    def test_cache_miss_nonexistent(self):
        db = BarbaraDB.open("test")
        cache = MarketDataCache(db, ttl=60.0)
        assert cache.get("NOPE") is None
        db.close()

    def test_invalidation(self):
        db = BarbaraDB.open("test")
        cache = MarketDataCache(db, ttl=60.0)

        snap = MarketDataSnapshot(
            ticker="AAPL", price=175.50,
            timestamp=datetime.now(),
        )
        cache.put(snap)
        assert cache.get("AAPL") is not None

        cache.invalidate("AAPL")
        assert cache.get("AAPL") is None
        db.close()


# ── TestHistoricalStore ───────────────────────────────────────────────────

class TestHistoricalStore:

    def test_append_and_query(self):
        store = HistoricalStore()
        rows = [
            {"timestamp": "2024-01-01T00:00:00", "open": 150.0,
             "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000},
            {"timestamp": "2024-01-02T00:00:00", "open": 151.0,
             "high": 153.0, "low": 150.0, "close": 152.0, "volume": 1100000},
        ]
        store.append("AAPL", rows)

        result = store.query("AAPL").to_list()
        assert len(result) == 2
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["close"] == 151.0

    def test_query_empty(self):
        store = HistoricalStore()
        result = store.query("NOPE").to_list()
        assert result == []

    def test_query_range(self):
        store = HistoricalStore()
        rows = [
            {"timestamp": "2024-01-01T00:00:00", "open": 150.0,
             "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000},
            {"timestamp": "2024-01-02T00:00:00", "open": 151.0,
             "high": 153.0, "low": 150.0, "close": 152.0, "volume": 1100000},
            {"timestamp": "2024-01-03T00:00:00", "open": 152.0,
             "high": 154.0, "low": 151.0, "close": 153.0, "volume": 1200000},
        ]
        store.append("AAPL", rows)

        result = store.query_range(
            "AAPL", "2024-01-01T00:00:00", "2024-01-02T00:00:00"
        ).to_list()
        assert len(result) == 2

    def test_multiple_tickers(self):
        store = HistoricalStore()
        store.append("AAPL", [
            {"timestamp": "2024-01-01T00:00:00", "open": 150.0,
             "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000000},
        ])
        store.append("MSFT", [
            {"timestamp": "2024-01-01T00:00:00", "open": 380.0,
             "high": 382.0, "low": 379.0, "close": 381.0, "volume": 500000},
        ])

        assert len(store.query("AAPL").to_list()) == 1
        assert len(store.query("MSFT").to_list()) == 1
        assert store.query("MSFT").to_list()[0]["close"] == 381.0


# ── TestEquity ────────────────────────────────────────────────────────────

class TestEquity:

    def test_value_from_spot(self):
        spot = MarketData("AAPL_SPOT", price=175.50)
        eq = Equity("AAPL", spot_source=spot)
        assert eq.value == 175.50

    def test_value_updates_with_spot(self):
        spot = MarketData("AAPL_SPOT", price=175.50)
        eq = Equity("AAPL", spot_source=spot)
        _ = eq.value  # trigger initial compute

        spot.set_price(180.00)
        eq.mark_dirty()
        assert eq.value == 180.00

    def test_dividend_adjusted_price(self):
        spot = MarketData("AAPL_SPOT", price=200.0)
        eq = Equity("AAPL", spot_source=spot, dividend_yield=0.01)
        assert eq.dividend_adjusted_price == pytest.approx(198.0)

    def test_underliers(self):
        spot = MarketData("AAPL_SPOT", price=175.50)
        eq = Equity("AAPL", spot_source=spot)
        assert eq.underliers == [spot]

    def test_registers_in_graph(self):
        spot = MarketData("AAPL_SPOT", price=175.50)
        eq = Equity("AAPL", spot_source=spot)
        graph = DependencyGraph()
        graph.register(eq)
        assert "AAPL" in graph.nodes
        assert "AAPL_SPOT" in graph.nodes


# ── TestFXRate ────────────────────────────────────────────────────────────

class TestFXRate:

    def test_value(self):
        rate = MarketData("EURUSD_RATE", price=1.08)
        fx = FXRate("EURUSD", rate_source=rate, base_currency="EUR",
                    quote_currency="USD")
        assert fx.value == 1.08

    def test_convert_base_to_quote(self):
        rate = MarketData("EURUSD_RATE", price=1.08)
        fx = FXRate("EURUSD", rate_source=rate, base_currency="EUR",
                    quote_currency="USD")
        assert fx.convert(100, from_currency="EUR") == pytest.approx(108.0)

    def test_convert_quote_to_base(self):
        rate = MarketData("EURUSD_RATE", price=1.08)
        fx = FXRate("EURUSD", rate_source=rate, base_currency="EUR",
                    quote_currency="USD")
        result = fx.convert(108, from_currency="USD")
        assert result == pytest.approx(100.0)

    def test_convert_default_is_base(self):
        rate = MarketData("EURUSD_RATE", price=1.08)
        fx = FXRate("EURUSD", rate_source=rate, base_currency="EUR",
                    quote_currency="USD")
        assert fx.convert(100) == pytest.approx(108.0)

    def test_convert_unknown_currency(self):
        rate = MarketData("EURUSD_RATE", price=1.08)
        fx = FXRate("EURUSD", rate_source=rate, base_currency="EUR",
                    quote_currency="USD")
        with pytest.raises(ValueError):
            fx.convert(100, from_currency="GBP")

    def test_inverse_rate(self):
        rate = MarketData("EURUSD_RATE", price=1.25)
        fx = FXRate("EURUSD", rate_source=rate, base_currency="EUR",
                    quote_currency="USD")
        assert fx.inverse_rate == pytest.approx(0.8)

    def test_underliers(self):
        rate = MarketData("EURUSD_RATE", price=1.08)
        fx = FXRate("EURUSD", rate_source=rate)
        assert fx.underliers == [rate]


# ── TestMarketDataManager ────────────────────────────────────────────────

class TestMarketDataManager:

    @patch("bank_python.market_data.yf.download")
    def test_update_all_triggers_recalc(self, mock_download):
        mock_download.return_value = _make_download_df({"AAPL": 175.50})

        db = BarbaraDB.open("test")
        graph = DependencyGraph()

        spot = MarketData("AAPL_SPOT", price=100.0)
        eq = Equity("AAPL_EQ", spot_source=spot)
        graph.register(eq)

        mgr = MarketDataManager(graph, db)
        mgr.register_ticker("AAPL", spot)

        results = mgr.update_all()

        assert results["AAPL"] == 175.50
        assert spot.value == 175.50
        assert eq.value == 175.50
        db.close()

    @patch("bank_python.market_data.yf.download")
    def test_cache_integration(self, mock_download):
        mock_download.return_value = _make_download_df({"AAPL": 175.50})

        db = BarbaraDB.open("test")
        graph = DependencyGraph()

        spot = MarketData("AAPL_SPOT", price=100.0)
        graph.register(spot)

        mgr = MarketDataManager(graph, db)
        mgr.register_ticker("AAPL", spot)
        mgr.update_all()

        cached = mgr.cache.get("AAPL")
        assert cached is not None
        assert cached.price == 175.50
        db.close()

    @patch("bank_python.market_data.yf.download")
    def test_history_storage(self, mock_download):
        mock_download.return_value = _make_download_df({"AAPL": 175.50})

        db = BarbaraDB.open("test")
        graph = DependencyGraph()

        spot = MarketData("AAPL_SPOT", price=100.0)
        graph.register(spot)

        mgr = MarketDataManager(graph, db)
        mgr.register_ticker("AAPL", spot)
        mgr.update_all()

        history = mgr.history.query("AAPL").to_list()
        assert len(history) == 1
        assert history[0]["close"] == 175.50
        db.close()

    @patch("bank_python.market_data.yf.download")
    def test_load_history(self, mock_download):
        mock_download.return_value = _make_historical_df(n_days=10)

        db = BarbaraDB.open("test")
        graph = DependencyGraph()

        mgr = MarketDataManager(graph, db)
        rows = mgr.load_history("AAPL", period="10d")

        assert len(rows) == 10
        stored = mgr.history.query("AAPL").to_list()
        assert len(stored) == 10
        db.close()

    def test_registered_tickers(self):
        db = BarbaraDB.open("test")
        graph = DependencyGraph()
        mgr = MarketDataManager(graph, db)

        spot = MarketData("AAPL_SPOT", price=100.0)
        mgr.register_ticker("AAPL", spot)

        assert "AAPL" in mgr.registered_tickers
        assert mgr.registered_tickers["AAPL"] is spot
        db.close()

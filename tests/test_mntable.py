"""Tests for MnTable — column-oriented table."""

import pickle
from bank_python.mntable import Table, LazyView


def make_sample_table():
    t = Table([("name", str), ("price", float), ("sector", str)])
    t.extend([
        {"name": "AAPL", "price": 150.0, "sector": "tech"},
        {"name": "GOOG", "price": 2800.0, "sector": "tech"},
        {"name": "JPM", "price": 160.0, "sector": "finance"},
        {"name": "GS", "price": 380.0, "sector": "finance"},
        {"name": "MSFT", "price": 310.0, "sector": "tech"},
    ])
    return t


class TestTableBasics:
    def test_create_and_len(self):
        t = Table([("x", int)])
        assert len(t) == 0
        t.append({"x": 42})
        assert len(t) == 1

    def test_extend_and_iterate(self):
        t = make_sample_table()
        assert len(t) == 5
        rows = list(t)
        assert rows[0]["name"] == "AAPL"
        assert rows[2]["sector"] == "finance"

    def test_columns_and_schema(self):
        t = Table([("a", int), ("b", str)])
        assert t.columns == ["a", "b"]
        assert t.schema == [("a", int), ("b", str)]


class TestLazyViews:
    def test_restrict(self):
        t = make_sample_table()
        view = t.restrict(sector="tech")
        assert isinstance(view, LazyView)
        rows = view.to_list()
        assert len(rows) == 3
        assert all(r["sector"] == "tech" for r in rows)

    def test_project(self):
        t = make_sample_table()
        view = t.project("name", "price")
        rows = view.to_list()
        assert len(rows) == 5
        assert set(rows[0].keys()) == {"name", "price"}

    def test_chained_restrict_project(self):
        t = make_sample_table()
        view = t.restrict(sector="finance").project("name")
        rows = view.to_list()
        assert len(rows) == 2
        names = {r["name"] for r in rows}
        assert names == {"JPM", "GS"}

    def test_lazy_view_len(self):
        t = make_sample_table()
        view = t.restrict(sector="tech")
        assert len(view) == 3

    def test_restrict_no_match(self):
        t = make_sample_table()
        view = t.restrict(sector="energy")
        assert len(view) == 0


class TestIndexing:
    def test_create_index(self):
        t = make_sample_table()
        t.create_index("sector")
        # Index doesn't change results, just speeds things up
        rows = t.restrict(sector="tech").to_list()
        assert len(rows) == 3


class TestAggregation:
    def test_aggregate_sum(self):
        t = make_sample_table()
        agg = t.aggregate("sector", {"price": "sum"})
        rows = {r["sector"]: r["sum_price"] for r in agg}
        assert rows["tech"] == 150.0 + 2800.0 + 310.0
        assert rows["finance"] == 160.0 + 380.0

    def test_aggregate_count(self):
        t = make_sample_table()
        agg = t.aggregate("sector", {"name": "count"})
        rows = {r["sector"]: r["count_name"] for r in agg}
        assert rows["tech"] == 3
        assert rows["finance"] == 2


class TestJoin:
    def test_inner_join(self):
        prices = Table([("ticker", str), ("price", float)])
        prices.extend([
            {"ticker": "AAPL", "price": 150.0},
            {"ticker": "GOOG", "price": 2800.0},
        ])

        info = Table([("ticker", str), ("sector", str)])
        info.extend([
            {"ticker": "AAPL", "sector": "tech"},
            {"ticker": "GOOG", "sector": "tech"},
            {"ticker": "JPM", "sector": "finance"},
        ])

        joined = prices.join(info, on="ticker")
        rows = list(joined)
        assert len(rows) == 2
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["sector"] == "tech"
        assert rows[0]["price"] == 150.0


class TestPickle:
    def test_roundtrip(self):
        """Test pickle roundtrip — pickle is used intentionally per the plan."""
        t = make_sample_table()
        data = pickle.dumps(t)
        t2 = pickle.loads(data)
        assert len(t2) == 5
        rows = list(t2)
        assert rows[0]["name"] == "AAPL"
        assert t2.columns == t.columns

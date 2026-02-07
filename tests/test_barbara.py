"""Tests for Barbara — hierarchical key-value store.

NOTE: Barbara intentionally uses pickle+zlib per the project plan,
matching the article's description of Bank Python's "zipped pickles".
"""

import pytest
from bank_python.barbara import BarbaraDB
from bank_python.mntable import Table


class TestBarbaraBasics:
    def test_open_and_repr(self):
        db = BarbaraDB.open("default")
        assert "default" in repr(db)
        db.close()

    def test_set_and_get(self):
        db = BarbaraDB.open("default")
        db["/key1"] = "hello"
        assert db["/key1"] == "hello"
        db.close()

    def test_missing_key_raises(self):
        db = BarbaraDB.open("default")
        with pytest.raises(KeyError):
            _ = db["/nonexistent"]
        db.close()

    def test_get_with_default(self):
        db = BarbaraDB.open("default")
        assert db.get("/missing", "fallback") == "fallback"
        db.close()

    def test_contains(self):
        db = BarbaraDB.open("default")
        db["/exists"] = 42
        assert "/exists" in db
        assert "/nope" not in db
        db.close()

    def test_delete(self):
        db = BarbaraDB.open("default")
        db["/todelete"] = "bye"
        assert "/todelete" in db
        del db["/todelete"]
        assert "/todelete" not in db
        db.close()

    def test_overwrite(self):
        db = BarbaraDB.open("default")
        db["/key"] = "v1"
        db["/key"] = "v2"
        assert db["/key"] == "v2"
        db.close()


class TestHierarchicalKeys:
    def test_keys_with_prefix(self):
        db = BarbaraDB.open("default")
        db["/Instruments/BOND_1"] = "bond1"
        db["/Instruments/CDS_1"] = "cds1"
        db["/MarketData/LIBOR"] = "libor"

        inst_keys = db.keys("/Instruments/")
        assert len(inst_keys) == 2
        assert "/Instruments/BOND_1" in inst_keys
        assert "/Instruments/CDS_1" in inst_keys

        md_keys = db.keys("/MarketData/")
        assert md_keys == ["/MarketData/LIBOR"]
        db.close()


class TestRings:
    def test_ring_cascade(self):
        db = BarbaraDB.open("desk;default")
        assert db.ring_names == ["desk", "default"]

        # Write to default ring directly
        db._rings[1].put("/shared", "from_default")
        # Read cascades to default
        assert db["/shared"] == "from_default"

        # Write to desk ring (first ring)
        db["/shared"] = "from_desk"
        # Now desk shadows default
        assert db["/shared"] == "from_desk"
        db.close()

    def test_writes_go_to_first_ring(self):
        db = BarbaraDB.open("primary;fallback")
        db["/test"] = "value"
        # Should be in primary ring
        assert db._rings[0].has("/test")
        # Should NOT be in fallback ring
        assert not db._rings[1].has("/test")
        db.close()


class TestMetadata:
    def test_put_with_metadata(self):
        db = BarbaraDB.open("default")
        db.put("/item", {"data": 1}, metadata={"source": "test"})
        meta = db.get_metadata("/item")
        assert meta == {"source": "test"}
        db.close()


class TestComplexValues:
    def test_store_dict(self):
        db = BarbaraDB.open("default")
        db["/complex"] = {"a": 1, "b": [2, 3], "c": {"nested": True}}
        val = db["/complex"]
        assert val["a"] == 1
        assert val["b"] == [2, 3]
        assert val["c"]["nested"] is True
        db.close()

    def test_store_table(self):
        """Tables can be stored in Barbara via pickle (intentional)."""
        db = BarbaraDB.open("default")
        t = Table([("x", int)])
        t.append({"x": 99})
        db["/tables/test"] = t

        restored = db["/tables/test"]
        assert len(restored) == 1
        assert list(restored)[0]["x"] == 99
        db.close()

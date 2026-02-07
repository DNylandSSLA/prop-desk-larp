"""
Barbara — Hierarchical key-value store with ring-based namespaces.

Inspired by the centralised data store described in Cal Paterson's article
about Bank Python. Barbara uses SQLite for persistence and pickle+zlib for
serialization, matching the article's description of "zipped pickles".

NOTE: pickle is used intentionally here. This is a learning project, not
production software handling untrusted data.

Rings provide namespace overlay logic: reads cascade through rings (falling
back to the next ring if a key isn't found), while writes always go to the
first (most specific) ring.
"""

import sqlite3
import pickle
import zlib
import time
import json


class Ring:
    """A single namespace backed by a SQLite table."""

    def __init__(self, conn, name):
        self._conn = conn
        self.name = name
        self._table = f"ring_{name}"
        self._conn.execute(
            f'CREATE TABLE IF NOT EXISTS "{self._table}" ('
            "  key TEXT PRIMARY KEY,"
            "  value BLOB NOT NULL,"
            "  metadata TEXT,"
            "  updated_at REAL"
            ")"
        )

    def get(self, key):
        cur = self._conn.execute(
            f'SELECT value FROM "{self._table}" WHERE key = ?', (key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return pickle.loads(zlib.decompress(row[0]))

    def put(self, key, value, metadata=None):
        blob = zlib.compress(pickle.dumps(value))
        meta_json = json.dumps(metadata) if metadata else None
        self._conn.execute(
            f'INSERT OR REPLACE INTO "{self._table}" '
            "(key, value, metadata, updated_at) VALUES (?, ?, ?, ?)",
            (key, blob, meta_json, time.time()),
        )
        self._conn.commit()

    def delete(self, key):
        self._conn.execute(
            f'DELETE FROM "{self._table}" WHERE key = ?', (key,)
        )
        self._conn.commit()

    def has(self, key):
        cur = self._conn.execute(
            f'SELECT 1 FROM "{self._table}" WHERE key = ?', (key,)
        )
        return cur.fetchone() is not None

    def keys(self, prefix=""):
        cur = self._conn.execute(
            f'SELECT key FROM "{self._table}" WHERE key LIKE ? ORDER BY key',
            (prefix + "%",),
        )
        return [row[0] for row in cur]

    def get_metadata(self, key):
        cur = self._conn.execute(
            f'SELECT metadata FROM "{self._table}" WHERE key = ?', (key,)
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return json.loads(row[0])


class BarbaraDB:
    """
    Hierarchical key-value store with ring-based namespace overlay.

    Usage:
        db = BarbaraDB.open("trading_desk;default")
        db["/Instruments/VODA_BOND"] = bond_obj
        bond = db["/Instruments/VODA_BOND"]
    """

    def __init__(self, rings, conn):
        self._rings = rings  # ordered list: most specific first
        self._conn = conn

    @classmethod
    def open(cls, ring_spec, db_path=":memory:"):
        """
        Open a Barbara store with the given ring specification.

        ring_spec: semicolon-separated ring names, e.g. "trading_desk;default"
                   Reads cascade left-to-right, writes go to the first ring.
        db_path: SQLite database path (default: in-memory)
        """
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        ring_names = [r.strip() for r in ring_spec.split(";")]
        rings = [Ring(conn, name) for name in ring_names]
        return cls(rings, conn)

    @property
    def ring_names(self):
        return [r.name for r in self._rings]

    @property
    def write_ring(self):
        return self._rings[0]

    def __setitem__(self, key, value):
        self.write_ring.put(key, value)

    def __getitem__(self, key):
        for ring in self._rings:
            val = ring.get(key)
            if val is not None:
                return val
        raise KeyError(key)

    def __contains__(self, key):
        for ring in self._rings:
            if ring.has(key):
                return True
        return False

    def __delitem__(self, key):
        self.write_ring.delete(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def put(self, key, value, metadata=None):
        self.write_ring.put(key, value, metadata=metadata)

    def keys(self, prefix=""):
        """Return all keys matching prefix across all rings (deduplicated)."""
        seen = set()
        result = []
        for ring in self._rings:
            for key in ring.keys(prefix):
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        result.sort()
        return result

    def get_metadata(self, key):
        for ring in self._rings:
            if ring.has(key):
                return ring.get_metadata(key)
        return None

    def close(self):
        self._conn.close()

    def __repr__(self):
        return f"BarbaraDB(rings={self.ring_names})"

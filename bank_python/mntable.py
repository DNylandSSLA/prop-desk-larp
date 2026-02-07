"""
MnTable — Column-oriented table backed by in-memory SQLite.

Inspired by the proprietary table libraries found inside investment banks,
MnTable provides typed schemas, lazy views (restrict/project), indexing,
aggregation, and joins — all backed by SQLite for zero-dependency performance.
"""

import sqlite3
import pickle
import itertools

# Map Python types to SQLite column affinities
_TYPE_MAP = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bool: "INTEGER",
    bytes: "BLOB",
}

_counter = itertools.count()


class LazyView:
    """A lazy view over a Table that defers SQL execution until iteration."""

    def __init__(self, table, sql, params=()):
        self._table = table
        self._sql = sql
        self._params = params

    def __iter__(self):
        cur = self._table._conn.execute(self._sql, self._params)
        cols = [desc[0] for desc in cur.description]
        for row in cur:
            yield dict(zip(cols, row))

    def __len__(self):
        # Wrap in a COUNT query
        count_sql = f"SELECT COUNT(*) FROM ({self._sql})"
        cur = self._table._conn.execute(count_sql, self._params)
        return cur.fetchone()[0]

    def to_list(self):
        return list(self)

    def restrict(self, **kwargs):
        where_parts = []
        params = list(self._params)
        for col, val in kwargs.items():
            where_parts.append(f'"{col}" = ?')
            params.append(val)
        new_sql = f"SELECT * FROM ({self._sql}) WHERE " + " AND ".join(where_parts)
        return LazyView(self._table, new_sql, tuple(params))

    def project(self, *cols):
        col_list = ", ".join(f'"{c}"' for c in cols)
        new_sql = f"SELECT {col_list} FROM ({self._sql})"
        return LazyView(self._table, new_sql, self._params)

    def __repr__(self):
        rows = self.to_list()
        if not rows:
            return "LazyView(empty)"
        return f"LazyView({len(rows)} rows)"


class Table:
    """
    Column-oriented table backed by in-memory SQLite.

    Usage:
        t = Table([("name", str), ("price", float)])
        t.extend([{"name": "AAPL", "price": 150.0}])
        for row in t.restrict(name="AAPL"):
            print(row)
    """

    def __init__(self, schema, name=None):
        """
        schema: list of (column_name, python_type) tuples
        name: optional table name (auto-generated if not provided)
        """
        self._schema = schema
        self._name = name or f"tbl_{next(_counter)}"
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")

        col_defs = []
        for col_name, col_type in schema:
            sql_type = _TYPE_MAP.get(col_type, "TEXT")
            col_defs.append(f'"{col_name}" {sql_type}')

        create_sql = f'CREATE TABLE "{self._name}" ({", ".join(col_defs)})'
        self._conn.execute(create_sql)

    @property
    def columns(self):
        return [name for name, _ in self._schema]

    @property
    def schema(self):
        return list(self._schema)

    def append(self, row):
        """Append a single row (dict)."""
        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(f'"{c}"' for c in cols)
        sql = f'INSERT INTO "{self._name}" ({col_names}) VALUES ({placeholders})'
        self._conn.execute(sql, [row[c] for c in cols])

    def extend(self, rows):
        """Append multiple rows (list of dicts)."""
        rows = list(rows)
        if not rows:
            return
        cols = list(rows[0].keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(f'"{c}"' for c in cols)
        sql = f'INSERT INTO "{self._name}" ({col_names}) VALUES ({placeholders})'
        self._conn.executemany(sql, [[r[c] for c in cols] for r in rows])

    def restrict(self, **kwargs):
        """Return a LazyView filtered by column=value conditions."""
        where_parts = []
        params = []
        for col, val in kwargs.items():
            where_parts.append(f'"{col}" = ?')
            params.append(val)
        if where_parts:
            sql = f'SELECT * FROM "{self._name}" WHERE ' + " AND ".join(where_parts)
        else:
            sql = f'SELECT * FROM "{self._name}"'
        return LazyView(self, sql, tuple(params))

    def project(self, *cols):
        """Return a LazyView with only the specified columns."""
        col_list = ", ".join(f'"{c}"' for c in cols)
        sql = f'SELECT {col_list} FROM "{self._name}"'
        return LazyView(self, sql)

    def create_index(self, col):
        """Create a B-tree index on the given column."""
        idx_name = f"idx_{self._name}_{col}"
        self._conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{self._name}" ("{col}")')

    def aggregate(self, group_by, agg_dict):
        """
        Aggregate the table.

        group_by: column name to group by (or list of column names)
        agg_dict: {column_name: "sum"|"avg"|"min"|"max"|"count"}

        Returns a new Table with the aggregated results.
        """
        if isinstance(group_by, str):
            group_by = [group_by]

        select_parts = [f'"{g}"' for g in group_by]
        result_schema = [(g, str) for g in group_by]  # group-by cols

        for col, func in agg_dict.items():
            func = func.upper()
            alias = f"{func.lower()}_{col}"
            select_parts.append(f'{func}("{col}") AS "{alias}"')
            if func in ("COUNT",):
                result_schema.append((alias, int))
            else:
                result_schema.append((alias, float))

        group_clause = ", ".join(f'"{g}"' for g in group_by)
        sql = f'SELECT {", ".join(select_parts)} FROM "{self._name}" GROUP BY {group_clause}'

        result = Table(result_schema)
        cur = self._conn.execute(sql)
        cols = [desc[0] for desc in cur.description]
        for row in cur:
            result.append(dict(zip(cols, row)))
        return result

    def join(self, other, on):
        """
        Inner join with another Table on a shared column.

        Returns a new Table with combined columns.
        """
        # Build combined schema (avoid duplicate join column)
        left_cols = self.columns
        right_cols = [c for c in other.columns if c != on]

        result_schema = list(self._schema)
        for col_name, col_type in other._schema:
            if col_name != on:
                result_schema.append((col_name, col_type))

        result = Table(result_schema)

        # Copy other's data into a temp table in this connection
        temp_name = f"_join_tmp_{next(_counter)}"
        other_col_defs = []
        for col_name, col_type in other._schema:
            sql_type = _TYPE_MAP.get(col_type, "TEXT")
            other_col_defs.append(f'"{col_name}" {sql_type}')
        self._conn.execute(
            f'CREATE TEMP TABLE "{temp_name}" ({", ".join(other_col_defs)})'
        )

        other_cols = other.columns
        placeholders = ", ".join("?" for _ in other_cols)
        col_names = ", ".join(f'"{c}"' for c in other_cols)
        other_rows = list(other._conn.execute(f'SELECT * FROM "{other._name}"'))
        self._conn.executemany(
            f'INSERT INTO "{temp_name}" ({col_names}) VALUES ({placeholders})',
            other_rows,
        )

        # Build join query
        left_select = [f'a."{c}"' for c in left_cols]
        right_select = [f'b."{c}"' for c in right_cols]
        select_clause = ", ".join(left_select + right_select)

        join_sql = (
            f'SELECT {select_clause} FROM "{self._name}" a '
            f'INNER JOIN "{temp_name}" b ON a."{on}" = b."{on}"'
        )

        cur = self._conn.execute(join_sql)
        all_cols = left_cols + right_cols
        for row in cur.fetchall():
            result.append(dict(zip(all_cols, row)))

        self._conn.execute(f'DROP TABLE "{temp_name}"')
        return result

    def __iter__(self):
        cur = self._conn.execute(f'SELECT * FROM "{self._name}"')
        cols = self.columns
        for row in cur:
            yield dict(zip(cols, row))

    def __len__(self):
        cur = self._conn.execute(f'SELECT COUNT(*) FROM "{self._name}"')
        return cur.fetchone()[0]

    def __repr__(self):
        return f"Table(name={self._name!r}, columns={self.columns}, rows={len(self)})"

    def __getstate__(self):
        """Pickle support: dump all rows + schema."""
        rows = list(self)
        return {
            "schema": self._schema,
            "name": self._name,
            "rows": rows,
        }

    def __setstate__(self, state):
        """Pickle support: rebuild table from saved state."""
        self._schema = state["schema"]
        self._name = state["name"]
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")

        col_defs = []
        for col_name, col_type in self._schema:
            sql_type = _TYPE_MAP.get(col_type, "TEXT")
            col_defs.append(f'"{col_name}" {sql_type}')
        self._conn.execute(
            f'CREATE TABLE "{self._name}" ({", ".join(col_defs)})'
        )

        if state["rows"]:
            self.extend(state["rows"])

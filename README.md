# Mini Bank Python

A simplified implementation of the four core components found inside investment bank proprietary Python ecosystems, inspired by [Cal Paterson's article](https://calpaterson.com/bank-python.html) about the strange world of Bank Python.

This is a learning/fun project — not production banking software. The goal is to capture the interesting architectural ideas (data-first, reactive computation, centralised store) using normal Python conventions.

**Pure Python, no external dependencies** — just `sqlite3`, `pickle`, `zlib`, `threading`, and standard library modules.

## Components

### MnTable (`bank_python/mntable.py`)

Column-oriented table backed by in-memory SQLite.

```python
from bank_python import Table

t = Table([("counterparty", str), ("quantity", float), ("price", float)])
t.extend([
    {"counterparty": "GS", "quantity": 1000, "price": 104.33},
    {"counterparty": "JPM", "quantity": 500, "price": 104.33},
    {"counterparty": "GS", "quantity": 300, "price": 12.95},
])

# Lazy views — SQL only runs on iteration
gs_trades = t.restrict(counterparty="GS").project("quantity", "price")
for row in gs_trades:
    print(row)

# Aggregation and joins
agg = t.aggregate("counterparty", {"quantity": "sum"})
```

### Barbara (`bank_python/barbara.py`)

Hierarchical key-value store with ring-based namespace overlay. Rings provide layered namespaces — reads cascade through rings, writes go to the first.

```python
from bank_python import BarbaraDB

db = BarbaraDB.open("trading_desk;default")
db["/Instruments/VODA_BOND"] = bond_obj
db["/Config/currency"] = "USD"

# Hierarchical browsing
keys = db.keys("/Instruments/")

# Ring overlay — desk-level values shadow defaults
```

### Dagger (`bank_python/dagger.py`)

Reactive dependency graph for instrument pricing. When a leaf node changes, dirty flags propagate through the graph via BFS and all affected instruments are recalculated.

```python
from bank_python import MarketData, Bond, DependencyGraph

libor = MarketData("LIBOR", price=0.05)
bond = Bond("VODA_BOND", rate_source=libor, face=100, coupon_rate=0.06, maturity=5)

graph = DependencyGraph()
graph.register(bond)  # automatically registers underliers

# Shock LIBOR — bond revalues automatically
libor.set_price(0.07)
graph.recalculate(libor)
print(bond.value)  # updated price
```

Built-in instrument types: `Bond`, `CreditDefaultSwap`, `Option`. Plus `Position` and `Book` for portfolio management.

### Walpole (`bank_python/walpole.py`)

Job orchestrator with scheduling, dependencies, and retry logic.

```python
from bank_python import JobRunner, JobConfig, JobMode

runner = JobRunner(barbara=db)  # optional Barbara integration

runner.add_job(JobConfig(
    name="market_data",
    callable=fetch_prices,
    mode=JobMode.PERIODIC,
    interval=5.0,
))
runner.add_job(JobConfig(
    name="revalue",
    callable=revalue_book,
    mode=JobMode.PERIODIC,
    interval=10.0,
    depends_on=["market_data"],
))

runner.start()
# ... jobs run in background threads ...
runner.stop()
```

Job modes: one-shot (`RUN_ONCE`), `PERIODIC`, and `SERVICE` (restart on crash).

## Running

```bash
# End-to-end demo
python demo.py

# Tests (50 tests)
python -m pytest tests/ -v
```

The demo walks through a complete scenario: creating market data and instruments, building a trade blotter, shocking LIBOR and watching reactive recalculation cascade, then running periodic Walpole jobs for 10 seconds.

## Project Structure

```
bank_python/
    __init__.py          # Package init + convenience imports
    barbara.py           # Key-value store (hierarchical, ring-based)
    dagger.py            # Reactive dependency graph engine
    walpole.py           # Job runner / scheduler
    mntable.py           # Column-oriented table library
tests/
    test_barbara.py
    test_dagger.py
    test_walpole.py
    test_mntable.py
demo.py                  # End-to-end demo tying all four together
```

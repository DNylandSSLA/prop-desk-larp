#!/usr/bin/env python3
"""
End-to-end demo of Mini Bank Python.

Walks through a complete scenario tying all four components together:
1. Open Barbara with a trading desk ring
2. Create market data and derived instruments in Dagger
3. Store everything in Barbara
4. Build a trade blotter in MnTable, demonstrate lazy queries
5. Construct positions and a trading book
6. Shock SOFR, watch reactive recalculation cascade
7. Start Walpole with periodic jobs
8. Run for 10 seconds, then show final state
"""

import time
import logging
import random

from bank_python import (
    BarbaraDB, Table,
    MarketData, Bond, CreditDefaultSwap, Option,
    Position, Book, DependencyGraph,
    JobRunner, JobConfig, JobMode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def main():
    print("=" * 70)
    print("  MINI BANK PYTHON — End-to-End Demo")
    print("=" * 70)

    # ── 1. Open Barbara ──────────────────────────────────────────────────
    print("\n── 1. Barbara: Opening store with rings ──")
    db = BarbaraDB.open("trading_desk;default")
    print(f"  {db}")

    # Store some reference data in the default ring
    db.put("/Config/base_currency", "USD")
    db.put("/Config/desk_name", "Rates Trading")
    print(f"  Stored config: currency={db['/Config/base_currency']}, "
          f"desk={db['/Config/desk_name']}")

    # ── 2. Dagger: Create market data and instruments ────────────────────
    print("\n── 2. Dagger: Creating market data and instruments ──")
    graph = DependencyGraph()

    # Market data (leaf nodes)
    sofr = MarketData("SOFR_3M", price=0.05)
    voda_price = MarketData("VODA_SPOT", price=215.0)
    voda_spread = MarketData("VODA_CREDIT_SPREAD", price=0.02)

    # Derived instruments
    bond = Bond("VODA_BOND", rate_source=sofr, face=100, coupon_rate=0.06, maturity=5)
    cds = CreditDefaultSwap("VODA_CDS", credit_spread_source=voda_spread,
                            rate_source=sofr, notional=10_000_000, maturity=5)
    option = Option("VODA_CALL", spot_source=voda_price, strike=220.0,
                    volatility=0.25, time_to_expiry=0.5)

    # Register all in the graph
    for inst in [sofr, voda_price, voda_spread, bond, cds, option]:
        graph.register(inst)

    print(f"  {graph}")
    print(f"  SOFR_3M       = {sofr.value:.4f}")
    print(f"  VODA_SPOT     = {voda_price.value:.2f}")
    print(f"  VODA_BOND     = {bond.value:.4f}")
    print(f"  VODA_CDS      = {cds.value:.2f}")
    print(f"  VODA_CALL     = {option.value:.4f}")

    # ── 3. Store instruments in Barbara ──────────────────────────────────
    print("\n── 3. Barbara: Storing instruments ──")
    db["/Instruments/VODA_BOND"] = bond
    db["/Instruments/VODA_CDS"] = cds
    db["/Instruments/VODA_CALL"] = option
    db["/MarketData/SOFR_3M"] = sofr
    db["/MarketData/VODA_SPOT"] = voda_price
    print(f"  Keys under /Instruments/: {db.keys('/Instruments/')}")
    print(f"  Keys under /MarketData/:  {db.keys('/MarketData/')}")

    # ── 4. MnTable: Build a trade blotter ────────────────────────────────
    print("\n── 4. MnTable: Trade blotter with lazy queries ──")
    blotter = Table([
        ("trade_id", int),
        ("instrument", str),
        ("side", str),
        ("quantity", float),
        ("price", float),
        ("counterparty", str),
    ], name="blotter")

    blotter.extend([
        {"trade_id": 1, "instrument": "VODA_BOND", "side": "BUY",
         "quantity": 1000, "price": bond.value, "counterparty": "GS"},
        {"trade_id": 2, "instrument": "VODA_CDS", "side": "BUY",
         "quantity": 1, "price": cds.value, "counterparty": "JPM"},
        {"trade_id": 3, "instrument": "VODA_CALL", "side": "BUY",
         "quantity": 500, "price": option.value, "counterparty": "MS"},
        {"trade_id": 4, "instrument": "VODA_BOND", "side": "SELL",
         "quantity": 200, "price": bond.value, "counterparty": "BARC"},
        {"trade_id": 5, "instrument": "VODA_CALL", "side": "BUY",
         "quantity": 300, "price": option.value, "counterparty": "GS"},
    ])
    blotter.create_index("instrument")
    blotter.create_index("counterparty")

    print(f"  {blotter}")

    # Lazy query: all VODA_BOND trades
    bond_trades = blotter.restrict(instrument="VODA_BOND")
    print(f"\n  VODA_BOND trades (lazy): {bond_trades}")
    for row in bond_trades:
        print(f"    {row}")

    # Lazy chain: GS trades, projected to instrument + quantity
    gs_summary = blotter.restrict(counterparty="GS").project("instrument", "quantity")
    print(f"\n  GS trades (instrument, quantity):")
    for row in gs_summary:
        print(f"    {row}")

    # Aggregation: total quantity by instrument
    agg = blotter.aggregate("instrument", {"quantity": "sum"})
    print(f"\n  Total quantity by instrument:")
    for row in agg:
        print(f"    {row}")

    # Store blotter in Barbara
    db["/Blotters/trades"] = blotter

    # ── 5. Positions and Book ────────────────────────────────────────────
    print("\n── 5. Dagger: Positions and trading book ──")
    book = Book("Rates Desk")
    book.add_position(Position(bond, quantity=800))   # net from trades
    book.add_position(Position(cds, quantity=1))
    book.add_position(Position(option, quantity=800))
    print(book.summary())

    db["/Books/rates_desk"] = book

    # ── 6. SOFR Shock ────────────────────────────────────────────────────
    print("\n── 6. Dagger: Shocking SOFR from 5% to 7% ──")
    print(f"  Before shock:")
    print(f"    SOFR   = {sofr.value:.4f}")
    print(f"    Bond   = {bond.value:.4f}")
    print(f"    CDS    = {cds.value:.2f}")
    print(f"    Book   = {book.total_value:.2f}")

    sofr.set_price(0.07)
    recalced = graph.recalculate(sofr)

    print(f"\n  After shock (recalculated {recalced}):")
    print(f"    SOFR   = {sofr.value:.4f}")
    print(f"    Bond   = {bond.value:.4f}")
    print(f"    CDS    = {cds.value:.2f}")
    print(f"    Book   = {book.total_value:.2f}")

    # ── 7. Walpole: Periodic jobs ────────────────────────────────────────
    print("\n── 7. Walpole: Starting periodic jobs (10 second run) ──")

    tick_count = {"n": 0}

    def update_market_data():
        """Simulate market data ticking."""
        tick_count["n"] += 1
        new_rate = 0.05 + random.uniform(-0.005, 0.005)
        sofr.set_price(new_rate)
        new_spot = 215 + random.uniform(-5, 5)
        voda_price.set_price(new_spot)
        return {"sofr": new_rate, "voda_spot": new_spot, "tick": tick_count["n"]}

    def revalue_book():
        """Revalue the book after market data changes."""
        graph.recalculate(sofr)
        graph.recalculate(voda_price)
        total = book.total_value
        return {"book_value": total, "bond": bond.value, "option": option.value}

    def generate_report():
        """Generate a summary report."""
        report = (
            f"Tick #{tick_count['n']}: "
            f"SOFR={sofr.value:.4f}, "
            f"VODA={voda_price.value:.2f}, "
            f"Bond={bond.value:.4f}, "
            f"Book={book.total_value:.2f}"
        )
        log.info(f"REPORT: {report}")
        return report

    runner = JobRunner(barbara=db)

    runner.add_job(JobConfig(
        name="market_data_update",
        callable=update_market_data,
        mode=JobMode.PERIODIC,
        interval=2.0,
        barbara_key="/Jobs/market_data_latest",
    ))

    runner.add_job(JobConfig(
        name="revalue_book",
        callable=revalue_book,
        mode=JobMode.PERIODIC,
        interval=2.5,
        depends_on=["market_data_update"],
        barbara_key="/Jobs/book_valuation",
    ))

    runner.add_job(JobConfig(
        name="report_generator",
        callable=generate_report,
        mode=JobMode.PERIODIC,
        interval=3.0,
        depends_on=["market_data_update"],
        barbara_key="/Jobs/latest_report",
    ))

    runner.start()
    time.sleep(10)
    runner.stop()

    # ── 8. Final state ───────────────────────────────────────────────────
    print("\n── 8. Final State ──")
    print(f"\n  {runner.status_report()}")

    latest_md = db.get("/Jobs/market_data_latest")
    latest_val = db.get("/Jobs/book_valuation")
    latest_report = db.get("/Jobs/latest_report")

    print(f"\n  Latest market data: {latest_md}")
    print(f"  Latest valuation:  {latest_val}")
    print(f"  Latest report:     {latest_report}")

    print(f"\n  All Barbara keys:")
    for key in db.keys():
        print(f"    {key}")

    print(f"\n  Final book state:")
    print(f"  {book.summary()}")

    db.close()
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
End-to-end demo of Mini Bank Python with live market data.

Walks through a complete scenario using real yfinance data for equities and FX,
while keeping SOFR and credit spreads simulated:
1. Open Barbara with a trading desk ring
2. Create MarketData nodes for real tickers (AAPL, MSFT, TSLA) + EURUSD FX
3. Simulated: SOFR_3M, VODA_CREDIT_SPREAD
4. Create instruments: Equity, FXRate, Option, Bond, CDS
5. MarketDataManager: register tickers, initial fetch, print live prices
6. Load 30-day history for AAPL, query with MnTable lazy views
7. Build trade blotter with real prices, positions, book
8. Shock SOFR, watch reactive recalculation cascade
9. Walpole: periodic market data fetch (30s interval), revalue, report
10. Run 30 seconds, show final state with real prices
"""

import time
import logging
import random

from bank_python import (
    BarbaraDB, Table,
    MarketData, Bond, CreditDefaultSwap, Option,
    Position, Book, DependencyGraph,
    JobRunner, JobConfig, JobMode,
    Equity, FXRate, MarketDataManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def main():
    print("=" * 70)
    print("  MINI BANK PYTHON — Live Market Data Demo")
    print("=" * 70)

    # ── 1. Open Barbara ──────────────────────────────────────────────────
    print("\n── 1. Barbara: Opening store with rings ──")
    db = BarbaraDB.open("trading_desk;default")
    print(f"  {db}")

    db.put("/Config/base_currency", "USD")
    db.put("/Config/desk_name", "Equity & Rates Trading")
    print(f"  Stored config: currency={db['/Config/base_currency']}, "
          f"desk={db['/Config/desk_name']}")

    # ── 2. Dagger: Create market data nodes ────────────────────────────
    print("\n── 2. Dagger: Creating market data nodes ──")
    graph = DependencyGraph()

    # Real tickers — prices will be fetched from Yahoo Finance
    aapl_spot = MarketData("AAPL_SPOT", price=0.0)
    msft_spot = MarketData("MSFT_SPOT", price=0.0)
    tsla_spot = MarketData("TSLA_SPOT", price=0.0)
    eurusd_rate = MarketData("EURUSD_RATE", price=0.0)

    # Simulated (not available on Yahoo Finance)
    sofr = MarketData("SOFR_3M", price=0.05)
    credit_spread = MarketData("VODA_CREDIT_SPREAD", price=0.02)

    # ── 3. Create instruments ──────────────────────────────────────────
    print("\n── 3. Creating instruments (Equity, FXRate, Option, Bond, CDS) ──")

    aapl_eq = Equity("AAPL", spot_source=aapl_spot)
    msft_eq = Equity("MSFT", spot_source=msft_spot, dividend_yield=0.008)
    eurusd = FXRate("EURUSD", rate_source=eurusd_rate,
                    base_currency="EUR", quote_currency="USD")

    # Option on AAPL (will use live spot)
    aapl_call = Option("AAPL_CALL", spot_source=aapl_spot, strike=220.0,
                       volatility=0.25, time_to_expiry=0.5)

    # Bond and CDS on simulated rates
    bond = Bond("CORP_BOND", rate_source=sofr, face=100, coupon_rate=0.06, maturity=5)
    cds = CreditDefaultSwap("CORP_CDS", credit_spread_source=credit_spread,
                            rate_source=sofr, notional=10_000_000, maturity=5)

    # Register all in the graph
    for inst in [aapl_spot, msft_spot, tsla_spot, eurusd_rate,
                 sofr, credit_spread,
                 aapl_eq, msft_eq, eurusd, aapl_call, bond, cds]:
        graph.register(inst)

    print(f"  {graph}")

    # ── 4. MarketDataManager: fetch live prices ────────────────────────
    print("\n── 4. MarketDataManager: Fetching live prices from Yahoo Finance ──")

    mgr = MarketDataManager(graph, db, cache_ttl=30.0)
    mgr.register_ticker("AAPL", aapl_spot)
    mgr.register_ticker("MSFT", msft_spot)
    mgr.register_ticker("TSLA", tsla_spot)
    mgr.register_ticker("EURUSD=X", eurusd_rate)

    print("  Fetching batch...")
    prices = mgr.update_all()

    for ticker, price in prices.items():
        status = f"${price:.2f}" if price else "FAILED"
        print(f"  {ticker:12s} = {status}")

    print(f"\n  Equity values:")
    print(f"    AAPL  = ${aapl_eq.value:.2f}")
    print(f"    MSFT  = ${msft_eq.value:.2f} (div-adj: ${msft_eq.dividend_adjusted_price:.2f})")
    print(f"    TSLA  = ${tsla_spot.value:.2f}")
    print(f"  FX:")
    print(f"    EUR/USD = {eurusd.value:.4f}")
    print(f"    100 EUR = ${eurusd.convert(100, 'EUR'):.2f}")
    print(f"    100 USD = €{eurusd.convert(100, 'USD'):.2f}")
    print(f"  Simulated:")
    print(f"    SOFR_3M            = {sofr.value:.4f}")
    print(f"    VODA_CREDIT_SPREAD = {credit_spread.value:.4f}")
    print(f"  Derived:")
    print(f"    AAPL_CALL (K=220) = ${aapl_call.value:.4f}")
    print(f"    CORP_BOND         = ${bond.value:.4f}")
    print(f"    CORP_CDS          = ${cds.value:.2f}")

    # ── 5. Store instruments in Barbara ────────────────────────────────
    print("\n── 5. Barbara: Storing instruments ──")
    db["/Instruments/AAPL"] = aapl_eq
    db["/Instruments/MSFT"] = msft_eq
    db["/Instruments/EURUSD"] = eurusd
    db["/Instruments/AAPL_CALL"] = aapl_call
    db["/Instruments/CORP_BOND"] = bond
    db["/Instruments/CORP_CDS"] = cds
    db["/MarketData/SOFR_3M"] = sofr
    print(f"  Keys under /Instruments/: {db.keys('/Instruments/')}")

    # ── 6. Historical data ─────────────────────────────────────────────
    print("\n── 6. Loading 30-day AAPL history ──")
    history = mgr.load_history("AAPL", period="1mo")
    if history:
        print(f"  Loaded {len(history)} daily rows")
        view = mgr.history.query("AAPL")
        print(f"  Historical query: {view}")
        rows = view.to_list()
        if len(rows) >= 2:
            first = rows[0]
            last = rows[-1]
            print(f"  First: {first['timestamp'][:10]} close=${first['close']:.2f}")
            print(f"  Last:  {last['timestamp'][:10]} close=${last['close']:.2f}")
    else:
        print("  (No historical data — market may be closed or network unavailable)")

    # ── 7. MnTable: Build a trade blotter ──────────────────────────────
    print("\n── 7. MnTable: Trade blotter with live prices ──")
    blotter = Table([
        ("trade_id", int),
        ("instrument", str),
        ("side", str),
        ("quantity", float),
        ("price", float),
        ("counterparty", str),
    ], name="blotter")

    blotter.extend([
        {"trade_id": 1, "instrument": "AAPL", "side": "BUY",
         "quantity": 500, "price": aapl_eq.value, "counterparty": "GS"},
        {"trade_id": 2, "instrument": "MSFT", "side": "BUY",
         "quantity": 300, "price": msft_eq.value, "counterparty": "JPM"},
        {"trade_id": 3, "instrument": "AAPL_CALL", "side": "BUY",
         "quantity": 1000, "price": aapl_call.value, "counterparty": "MS"},
        {"trade_id": 4, "instrument": "CORP_BOND", "side": "BUY",
         "quantity": 1000, "price": bond.value, "counterparty": "BARC"},
        {"trade_id": 5, "instrument": "CORP_CDS", "side": "BUY",
         "quantity": 1, "price": cds.value, "counterparty": "GS"},
    ])
    blotter.create_index("instrument")
    blotter.create_index("counterparty")
    print(f"  {blotter}")

    # Lazy queries
    gs_summary = blotter.restrict(counterparty="GS").project("instrument", "quantity", "price")
    print(f"\n  GS trades:")
    for row in gs_summary:
        print(f"    {row}")

    agg = blotter.aggregate("instrument", {"quantity": "sum"})
    print(f"\n  Total quantity by instrument:")
    for row in agg:
        print(f"    {row}")

    db["/Blotters/trades"] = blotter

    # ── 8. Positions and Book ──────────────────────────────────────────
    print("\n── 8. Positions and trading book ──")
    book = Book("Equity & Rates Desk")
    book.add_position(Position(aapl_eq, quantity=500))
    book.add_position(Position(msft_eq, quantity=300))
    book.add_position(Position(aapl_call, quantity=1000))
    book.add_position(Position(bond, quantity=1000))
    book.add_position(Position(cds, quantity=1))
    print(book.summary())

    db["/Books/eq_rates_desk"] = book

    # ── 9. SOFR Shock ──────────────────────────────────────────────────
    print("\n── 9. Shocking SOFR from 5% to 7% ──")
    print(f"  Before shock:")
    print(f"    SOFR      = {sofr.value:.4f}")
    print(f"    Bond      = {bond.value:.4f}")
    print(f"    CDS       = {cds.value:.2f}")
    print(f"    Book      = {book.total_value:.2f}")

    sofr.set_price(0.07)
    recalced = graph.recalculate(sofr)

    print(f"\n  After shock (recalculated {recalced}):")
    print(f"    SOFR      = {sofr.value:.4f}")
    print(f"    Bond      = {bond.value:.4f}")
    print(f"    CDS       = {cds.value:.2f}")
    print(f"    Book      = {book.total_value:.2f}")

    # ── 10. Walpole: Periodic jobs ─────────────────────────────────────
    print("\n── 10. Walpole: Starting periodic jobs (30 second run) ──")

    tick_count = {"n": 0}

    def update_market_data():
        """Fetch live market data from Yahoo Finance."""
        tick_count["n"] += 1
        result = mgr.update_all()
        # Also jiggle SOFR (simulated)
        new_rate = 0.05 + random.uniform(-0.005, 0.005)
        sofr.set_price(new_rate)
        graph.recalculate(sofr)
        return {
            "tick": tick_count["n"],
            "prices": {k: f"${v:.2f}" if v else "N/A" for k, v in result.items()},
            "sofr": new_rate,
        }

    def revalue_book():
        """Revalue the book after market data changes."""
        total = book.total_value
        return {
            "book_value": total,
            "aapl": aapl_eq.value,
            "msft": msft_eq.value,
            "bond": bond.value,
            "option": aapl_call.value,
        }

    def generate_report():
        """Generate a summary report."""
        report = (
            f"Tick #{tick_count['n']}: "
            f"AAPL=${aapl_eq.value:.2f}, "
            f"MSFT=${msft_eq.value:.2f}, "
            f"EUR/USD={eurusd.value:.4f}, "
            f"Bond={bond.value:.4f}, "
            f"Book=${book.total_value:.2f}"
        )
        log.info(f"REPORT: {report}")
        return report

    runner = JobRunner(barbara=db)

    runner.add_job(JobConfig(
        name="market_data_update",
        callable=update_market_data,
        mode=JobMode.PERIODIC,
        interval=10.0,
        barbara_key="/Jobs/market_data_latest",
    ))

    runner.add_job(JobConfig(
        name="revalue_book",
        callable=revalue_book,
        mode=JobMode.PERIODIC,
        interval=12.0,
        depends_on=["market_data_update"],
        barbara_key="/Jobs/book_valuation",
    ))

    runner.add_job(JobConfig(
        name="report_generator",
        callable=generate_report,
        mode=JobMode.PERIODIC,
        interval=15.0,
        depends_on=["market_data_update"],
        barbara_key="/Jobs/latest_report",
    ))

    runner.start()
    time.sleep(30)
    runner.stop()

    # ── 11. Final state ───────────────────────────────────────────────
    print("\n── 11. Final State ──")
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

    # Historical data summary
    aapl_history = mgr.history.query("AAPL").to_list()
    print(f"\n  AAPL historical rows stored: {len(aapl_history)}")

    db.close()
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

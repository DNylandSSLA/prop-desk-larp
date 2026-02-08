#!/usr/bin/env python3
"""
Showcase — Continuous demo cycling through every feature of the prop desk platform.

Runs through all capabilities in a loop with live market data:
  Monte Carlo (GBM → Heston → Merton), stress tests, trading signals,
  order execution, portfolio optimization (MV, RP, Black-Litterman),
  efficient frontier, rebalancing, backtesting, and tearsheets.

Usage:
    python showcase.py              # default 8s pause between scenes
    python showcase.py --pause 5    # custom pause
    python showcase.py --once       # single pass, no loop
"""

import argparse
import json
import random
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from prop_desk import (
    open_db, ensure_config, get_config, setup_desk, snapshot_initial_values,
    compute_greeks, compute_trader_risk, aggregate_desk_risk,
    check_risk_limits, EQUITY_TICKERS, FX_TICKERS,
    cmd_mc, cmd_stress, cmd_risk, cmd_pnl,
    cmd_signals, cmd_order, cmd_orders, cmd_audit,
    cmd_optimize, cmd_frontier, cmd_rebalance,
    cmd_backtest, cmd_tearsheet,
    one_shot_report,
)
from bank_python.integrations import HAS_PYPFOPT, HAS_RISKFOLIO, HAS_QUANTLIB

console = Console(width=140)

BANNER = r"""
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║                                                                            ║
 ║    ██████╗ ██████╗  ██████╗ ██████╗     ██████╗ ███████╗███████╗██╗  ██╗   ║
 ║    ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗    ██╔══██╗██╔════╝██╔════╝██║ ██╔╝   ║
 ║    ██████╔╝██████╔╝██║   ██║██████╔╝    ██║  ██║█████╗  ███████╗█████╔╝    ║
 ║    ██╔═══╝ ██╔══██╗██║   ██║██╔═══╝     ██║  ██║██╔══╝  ╚════██║██╔═██╗    ║
 ║    ██║     ██║  ██║╚██████╔╝██║         ██████╔╝███████╗███████║██║  ██╗   ║
 ║    ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝         ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝   ║
 ║                                                                            ║
 ║         Full Platform Showcase — Live Data · Risk · Trading · Alpha        ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
"""


def scene_header(title, subtitle="", scene_num=None, total=None):
    """Print a scene transition header."""
    console.print()
    if scene_num and total:
        tag = f"[dim]({scene_num}/{total})[/dim] "
    else:
        tag = ""
    console.print(Rule(style="cyan"))
    console.print(
        Panel(
            f"[bold cyan]{tag}{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()


def refresh_market_data(state):
    """Quick market data refresh between scenes."""
    try:
        state["mgr"].update_all()
        # Jiggle credit spread
        spread_md = state["spread_md"]
        new_spread = max(0.005, spread_md.value + random.uniform(-0.001, 0.001))
        spread_md.set_price(new_spread)
        state["graph"].recalculate(spread_md)
    except Exception:
        pass


def run_showcase(pause=8, once=False, plain=False):
    """Main showcase loop."""
    global console
    if plain:
        console = Console(width=160, no_color=True, highlight=False)
        pause = 0  # no waiting in plain mode
        once = True
    console.print(BANNER, style="bold cyan")
    console.print()

    # ── Infrastructure setup ──
    console.print("[bold yellow]Initializing prop desk infrastructure...[/bold yellow]")
    db = open_db()
    ensure_config(db)
    config = get_config(db)

    state = setup_desk(db)
    traders = state["traders"]
    mgr = state["mgr"]

    console.print("[dim]Fetching live market data from Yahoo Finance...[/dim]")
    mgr.update_all()
    snapshot_initial_values(traders)

    console.print(
        f"[green]Desk online.[/green] "
        f"[dim]{len(traders)} traders, "
        f"{sum(len(t.book.positions) for t in traders)} positions, "
        f"{len(EQUITY_TICKERS)} equities tracked[/dim]\n"
    )
    if not plain:
        time.sleep(2)

    scenes = [
        # (title, subtitle, callable)
        (
            "DESK OVERVIEW",
            f"14 traders · {len(EQUITY_TICKERS)} equities/ETFs · options, bonds, FX, CDS · live prices",
            lambda: one_shot_report(db, state, config, console),
        ),
        (
            "MONTE CARLO — Geometric Brownian Motion",
            "50,000 paths · 1-day horizon · Cholesky-correlated returns",
            lambda: cmd_mc(db, state, console, n_paths=50_000, model="gbm"),
        ),
        (
            "MONTE CARLO — Heston Stochastic Volatility",
            "50,000 paths · mean-reverting variance · vol-of-vol · leverage effect",
            lambda: cmd_mc(db, state, console, n_paths=50_000, model="heston"),
        ),
        (
            "MONTE CARLO — Merton Jump-Diffusion",
            "50,000 paths · Poisson jumps · fat tails · crash risk",
            lambda: cmd_mc(db, state, console, n_paths=50_000, model="merton"),
        ),
        (
            "STRESS TEST SCENARIOS",
            "2008 GFC · COVID crash · rate shock · vol spike · USD rally",
            lambda: cmd_stress(db, state, console),
        ),
        (
            "TRADING SIGNALS",
            "Momentum MA crossover · stat arb z-score · macro regime",
            lambda: cmd_signals(db, state, console),
        ),
        (
            "ORDER EXECUTION",
            "Live order → risk check → slippage model → fill → audit",
            lambda: _run_order_scene(db, state),
        ),
        (
            "PORTFOLIO OPTIMIZATION — Mean-Variance",
            "Projected gradient descent on simplex · live covariance from yfinance",
            lambda: cmd_optimize(db, state, console, method="mv"),
        ),
        (
            "PORTFOLIO OPTIMIZATION — Risk Parity",
            "Equal risk contribution · inverse marginal risk weighting",
            lambda: cmd_optimize(db, state, console, method="rp"),
        ),
        (
            "PORTFOLIO OPTIMIZATION — Black-Litterman",
            "Market equilibrium + Bayesian view updating",
            lambda: cmd_optimize(db, state, console, method="bl"),
        ),
        (
            "PORTFOLIO OPTIMIZATION — HRP (PyPortfolioOpt)",
            "Hierarchical Risk Parity · dendrogram clustering · no covariance inversion",
            lambda: cmd_optimize(db, state, console, method="hrp", backend="pypfopt"),
        ) if HAS_PYPFOPT else None,
        (
            "PORTFOLIO OPTIMIZATION — CVaR Minimization (Riskfolio-Lib)",
            "Minimize Conditional Value-at-Risk · tail-risk focus · 5% worst outcomes",
            lambda: cmd_optimize(db, state, console, method="cvar", backend="riskfolio"),
        ) if HAS_RISKFOLIO else None,
        (
            "QUANTLIB PRICING COMPARISON",
            "Hand-rolled vs QuantLib · options Greeks · bond duration/convexity",
            lambda: _run_quantlib_comparison(state),
        ) if HAS_QUANTLIB else None,
        (
            "EFFICIENT FRONTIER",
            "40-point risk/return sweep · ASCII visualization",
            lambda: cmd_frontier(db, state, console, n_points=40),
        ),
        (
            "PORTFOLIO REBALANCE",
            "Current positions → optimal weights → trade list (dry run)",
            lambda: cmd_rebalance(db, state, console, method="mv", dry_run=True),
        ),
        (
            "BACKTEST — Momentum Strategy",
            "Top-N trailing returns · 2024-01-01 to 2024-12-31 · $1M capital",
            lambda: cmd_backtest(db, state, console, "momentum",
                                 "2024-01-01", "2024-12-31"),
        ),
        (
            "BACKTEST — All Strategies Head-to-Head",
            "Momentum vs Vol Arb vs Stat Arb vs Macro · same period · tearsheets",
            lambda: cmd_backtest(db, state, console, "all",
                                 "2024-01-01", "2024-12-31"),
        ),
        (
            "TEARSHEET",
            "Performance metrics from last backtest · Sharpe, Sortino, max DD",
            lambda: cmd_tearsheet(db, state, console),
        ),
    ]

    # Filter out None scenes (libraries not installed)
    scenes = [s for s in scenes if s is not None]

    loop_count = 0
    try:
        while True:
            loop_count += 1
            start_loop = time.time()

            if loop_count > 1:
                console.print()
                console.print(Rule(style="yellow"))
                console.print(
                    Panel(
                        f"[bold yellow]LOOP {loop_count}[/bold yellow]  "
                        f"[dim]Refreshing market data...[/dim]",
                        border_style="yellow",
                    )
                )
                refresh_market_data(state)
                snapshot_initial_values(traders)
                console.print("[green]Market data refreshed.[/green]\n")
                time.sleep(2)

            for i, (title, subtitle, fn) in enumerate(scenes, 1):
                scene_header(title, subtitle, scene_num=i, total=len(scenes))

                try:
                    fn()
                except Exception as e:
                    console.print(f"[red]Scene error: {e}[/red]")

                elapsed = time.time() - start_loop
                console.print(
                    f"\n[dim]Scene {i}/{len(scenes)} complete · "
                    f"Session: {elapsed:.0f}s · "
                    f"Loop {loop_count} · "
                    f"{datetime.now().strftime('%H:%M:%S')}[/dim]"
                )

                if i < len(scenes) or not once:
                    # Countdown to next scene
                    for remaining in range(pause, 0, -1):
                        console.print(
                            f"\r[dim]Next scene in {remaining}s... "
                            f"(Ctrl+C to stop)[/dim]",
                            end="",
                        )
                        time.sleep(1)
                    console.print()

                # Refresh data every few scenes
                if i % 4 == 0:
                    refresh_market_data(state)

            if once:
                break

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_loop if 'start_loop' in dir() else 0
        console.print()
        console.print(Rule(style="cyan"))
        console.print(
            Panel(
                f"[bold cyan]Showcase complete.[/bold cyan]\n"
                f"[dim]Loops: {loop_count} · "
                f"Scenes: {loop_count * len(scenes)} · "
                f"{datetime.now().strftime('%H:%M:%S')}[/dim]",
                border_style="cyan",
            )
        )
        db.close()


def _run_order_scene(db, state):
    """Submit a few orders, show the book, show the audit trail."""
    # Clear stale audit/order keys so we only show this run's activity
    for prefix in ["/Trading/audit/", "/Trading/orders/"]:
        for key in db.keys(prefix=prefix):
            try:
                del db[key]
            except Exception:
                pass

    # Pick random tickers for variety — sample from a broad range
    tickers_to_trade = random.sample(EQUITY_TICKERS[:30], 5)

    trader_orders = [
        ("Alice", tickers_to_trade[0], "BUY", 50),
        ("Tony",  tickers_to_trade[1], "BUY", 75),
        ("Nero",  tickers_to_trade[2], "SELL", 40),
        ("Zara",  tickers_to_trade[3], "BUY", 120),
        ("Diana", tickers_to_trade[4], "BUY", 100),
    ]

    for trader_name, sym, side, qty in trader_orders:
        console.print(f"[dim]Submitting: {trader_name} {side} {qty} {sym}...[/dim]")
        try:
            cmd_order(db, state, console, trader_name, sym, side, qty)
        except Exception as e:
            console.print(f"[yellow]Order issue: {e}[/yellow]")
        console.print()

    console.print(Rule("Order Book", style="dim"))
    cmd_orders(db, state, console)

    console.print()
    console.print(Rule("Audit Trail (last 10)", style="dim"))
    cmd_audit(db, state, console)


def _run_quantlib_comparison(state):
    """Side-by-side comparison of hand-rolled vs QuantLib pricing."""
    from rich.table import Table as RichTable
    from rich.text import Text

    if not HAS_QUANTLIB:
        console.print("[yellow]QuantLib not installed — skipping comparison[/yellow]")
        return

    from bank_python.integrations.quantlib_pricing import QuantLibOption, QuantLibBond

    # Options comparison
    tbl = RichTable(title="Option Pricing: Hand-Rolled vs QuantLib", expand=True,
                    show_header=True, header_style="bold cyan")
    tbl.add_column("Option", style="bold", width=14)
    tbl.add_column("Spot", justify="right", width=10)
    tbl.add_column("Hand-Rolled", justify="right", width=12)
    tbl.add_column("QuantLib", justify="right", width=12)
    tbl.add_column("Diff", justify="right", width=10)
    tbl.add_column("QL Delta", justify="right", width=10)
    tbl.add_column("QL Gamma", justify="right", width=10)
    tbl.add_column("QL Vega", justify="right", width=10)

    opt_keys = [("aapl_call", "AAPL"), ("msft_put", "MSFT"),
                ("nvda_call", "NVDA"), ("tsla_call", "TSLA"), ("meta_call", "META")]
    for opt_key, ticker in opt_keys:
        hand_opt = state.get(opt_key)
        if hand_opt is None or hand_opt.spot_source.value <= 0:
            continue

        ql_opt = QuantLibOption(
            f"QL_{hand_opt.name}", spot_source=hand_opt.spot_source,
            strike=hand_opt.strike, volatility=hand_opt.volatility,
            time_to_expiry=hand_opt.time_to_expiry, is_call=hand_opt.is_call,
            risk_free_rate=0.0,  # match hand-rolled BS which uses r=0
        )
        _ = ql_opt.value  # trigger compute
        greeks = ql_opt.greeks()

        hr_price = hand_opt.value
        ql_price = ql_opt.value
        diff_pct = abs(hr_price - ql_price) / max(abs(hr_price), 1e-6) * 100

        diff_style = "green" if diff_pct < 1 else ("yellow" if diff_pct < 5 else "red")
        tbl.add_row(
            hand_opt.name,
            f"${hand_opt.spot_source.value:.2f}",
            f"${hr_price:.4f}",
            f"${ql_price:.4f}",
            Text(f"{diff_pct:.2f}%", style=diff_style),
            f"{greeks['delta']:.4f}",
            f"{greeks['gamma']:.6f}",
            f"{greeks['vega']:.4f}",
        )

    console.print(Panel(tbl, border_style="cyan"))

    # Bonds comparison
    btbl = RichTable(title="Bond Pricing: Hand-Rolled vs QuantLib", expand=True,
                     show_header=True, header_style="bold cyan")
    btbl.add_column("Bond", style="bold", width=12)
    btbl.add_column("Rate", justify="right", width=8)
    btbl.add_column("Hand-Rolled", justify="right", width=12)
    btbl.add_column("QuantLib", justify="right", width=12)
    btbl.add_column("Diff", justify="right", width=10)
    btbl.add_column("Duration", justify="right", width=10)
    btbl.add_column("Convexity", justify="right", width=10)

    for bond_key in ("bond_5y", "bond_10y", "bond_2y"):
        hand_bond = state.get(bond_key)
        if hand_bond is None:
            continue

        ql_bond = QuantLibBond(
            f"QL_{hand_bond.name}", rate_source=hand_bond.rate_source,
            face=hand_bond.face, coupon_rate=hand_bond.coupon_rate,
            maturity=hand_bond.maturity,
        )
        _ = ql_bond.value

        hr_price = hand_bond.value
        ql_price = ql_bond.value
        diff_pct = abs(hr_price - ql_price) / max(abs(hr_price), 1e-6) * 100

        diff_style = "green" if diff_pct < 1 else ("yellow" if diff_pct < 5 else "red")
        btbl.add_row(
            hand_bond.name,
            f"{hand_bond.rate_source.value:.4f}",
            f"${hr_price:.2f}",
            f"${ql_price:.2f}",
            Text(f"{diff_pct:.2f}%", style=diff_style),
            f"{ql_bond.duration:.4f}",
            f"{ql_bond.convexity:.4f}",
        )

    console.print(Panel(btbl, border_style="cyan"))


def _jiggle_market(state, rng, intensity=1.0):
    """
    Simulate intraday price moves for all assets.

    Each equity/FX spot gets a random ±0.3-1.5% move (scaled by intensity).
    VIX mean-reverts toward 20 with noise. SOFR drifts ±2bp. Spread widens/tightens.
    Option implied vols are rescaled based on VIX level.
    """
    graph = state["graph"]
    spots = state["spots"]
    vix_md = state["vix_md"]
    sofr_md = state["sofr_md"]
    spread_md = state["spread_md"]

    # Equity + ETF spot moves — correlated via a common market factor
    market_factor = rng.normal(0, 0.005 * intensity)  # common shock
    for ticker, md in spots.items():
        if md.value <= 0:
            continue
        idio = rng.normal(0, 0.008 * intensity)  # idiosyncratic
        move = market_factor + idio
        md.set_price(md.value * (1 + move))
    # Batch recalc (spots → equities → options recalculate)
    for md in spots.values():
        if md.value > 0:
            graph.recalculate(md)

    # FX moves — independent random walks
    for key in ("eurusd_md", "gbpusd_md", "usdjpy_md", "audusd_md",
                "usdcad_md", "usdchf_md"):
        md = state[key]
        move = rng.normal(0, 0.003 * intensity)
        md.set_price(md.value * (1 + move))
        graph.recalculate(md)

    # VIX — mean-reverting (Ornstein-Uhlenbeck toward 20)
    vix_mean = 20.0
    vix_kappa = 0.15 * intensity  # mean-reversion speed
    vix_vol = 2.0 * intensity
    new_vix = vix_md.value + vix_kappa * (vix_mean - vix_md.value) + rng.normal(0, vix_vol)
    new_vix = max(10.0, min(80.0, new_vix))
    vix_md.set_price(new_vix)
    graph.recalculate(vix_md)

    # Rescale option implied vols based on VIX
    vix_ratio = new_vix / 20.0  # baseline VIX = 20
    for opt_key in ("aapl_call", "msft_put", "nvda_call", "nvda_put",
                    "tsla_call", "meta_call"):
        opt = state.get(opt_key)
        if opt:
            # Base vols from initial setup, scaled by VIX regime
            base_vols = {"aapl_call": 0.25, "msft_put": 0.22, "nvda_call": 0.35,
                         "nvda_put": 0.35, "tsla_call": 0.40, "meta_call": 0.28}
            base = base_vols.get(opt_key, 0.25)
            opt.volatility = base * (0.5 + 0.5 * vix_ratio)  # blend

    # SOFR — small drift ±2bp
    sofr_shock = rng.normal(0, 0.0002 * intensity)
    sofr_md.set_price(max(0.01, sofr_md.value + sofr_shock))
    graph.recalculate(sofr_md)

    # Credit spread — mean-reverting around 2% with jumps
    spread_mean = 0.02
    spread_shock = 0.1 * (spread_mean - spread_md.value) + rng.normal(0, 0.002 * intensity)
    if rng.random() < 0.05:  # 5% chance of a spread blowout
        spread_shock += rng.choice([-1, 1]) * rng.uniform(0.005, 0.015)
    spread_md.set_price(max(0.003, spread_md.value + spread_shock))
    graph.recalculate(spread_md)


def run_dense(iterations=10):
    """
    Machine-readable dense output mode.
    Fetches data once, then simulates N iterations of intraday dynamics:
    price moves, VIX regime shifts, rate drift, MC risk, stress tests,
    signal generation, and order execution — all with real randomness.
    """
    warnings.filterwarnings("ignore")
    import numpy as np

    from mc_engine import MCConfig, MonteCarloEngine, StressEngine, CovarianceBuilder
    from bank_python.trading_engine import (
        MomentumSignal, StatArbSignal, MacroSignal, VolArbSignal, TradingEngine,
    )
    from bank_python.optimizer import (
        MeanVarianceOptimizer, RiskParityOptimizer, BlackLittermanModel,
    )
    from bank_python.backtester import (
        Backtester, MomentumBacktest, VolArbBacktest,
        StatArbBacktest, MacroBacktest,
    )

    def emit(obj):
        print(json.dumps(obj, separators=(",", ":")))

    # Master RNG for reproducible but varied randomness
    master_rng = np.random.default_rng(int(time.time()) % 2**31)

    # ── Setup (once) ──
    t0 = time.time()
    db = open_db()
    ensure_config(db)
    config = get_config(db)
    state = setup_desk(db)
    traders = state["traders"]
    mgr = state["mgr"]
    mgr.update_all()
    snapshot_initial_values(traders)

    # Load history for signals + covariance (before first snapshot)
    signal_tickers = EQUITY_TICKERS[:12] + ["GOOGL", "AMZN"]
    signal_tickers = list(dict.fromkeys(signal_tickers))  # dedupe
    for ticker in signal_tickers:
        mgr.load_history(ticker, "1y")

    # Collect initial prices
    def snapshot_prices():
        prices = {}
        for ticker in EQUITY_TICKERS:
            eq = state["equities"].get(ticker)
            if eq and eq.value:
                prices[ticker] = round(eq.value, 2)
        for label, key in [("EURUSD", "eurusd_md"), ("GBPUSD", "gbpusd_md"),
                           ("USDJPY", "usdjpy_md"), ("AUDUSD", "audusd_md"),
                           ("USDCAD", "usdcad_md"), ("USDCHF", "usdchf_md"),
                           ("VIX", "vix_md"), ("SOFR", "sofr_md"),
                           ("SPREAD", "spread_md")]:
            prices[label] = round(state[key].value, 4)
        return prices

    prices = snapshot_prices()

    emit({
        "scene": "setup",
        "traders": len(traders),
        "positions": sum(len(t.book.positions) for t in traders),
        "tickers": len(prices),
        "setup_sec": round(time.time() - t0, 2),
    })

    # ── Initial desk snapshot ──
    def emit_desk_snapshot(tag="desk"):
        desk_risk = aggregate_desk_risk(traders)
        trader_summary = []
        for t in traders:
            r = compute_trader_risk(t)
            trader_summary.append({
                "name": t.name, "strategy": t.strategy,
                "mv": round(t.book.total_value, 0),
                "delta": round(r["delta"], 0),
                "gamma": round(r["gamma"], 1),
                "vega": round(r["vega"], 0),
                "n_pos": len(t.book.positions),
            })
        alerts = check_risk_limits(traders, config, db)
        p = snapshot_prices()
        emit({
            "scene": tag,
            "desk_mv": round(sum(t.book.total_value for t in traders), 0),
            "desk_delta": round(desk_risk["delta"], 0),
            "desk_gamma": round(desk_risk["gamma"], 1),
            "desk_vega": round(desk_risk["vega"], 0),
            "desk_pnl": round(sum(t.book.total_value - t.initial_value for t in traders), 0),
            "traders": trader_summary,
            "n_alerts": len(alerts),
            "alert_summary": {s: sum(1 for a in alerts if a["severity"] == s)
                              for s in ("CRITICAL", "HIGH", "MEDIUM")},
            "vix": round(state["vix_md"].value, 2),
            "sofr": round(state["sofr_md"].value, 4),
            "spread": round(state["spread_md"].value, 4),
            "prices_sample": {k: p[k] for k in list(p)[:12]},
        })

    emit_desk_snapshot("desk")

    # ── Stress test (initial) ──
    def emit_stress():
        stress_engine = StressEngine()
        stress_results = stress_engine.run_all_scenarios(traders, state)
        stress_out = {}
        for scenario, res in stress_results.items():
            per_trader = {}
            for t in traders:
                per_trader[t.name] = round(
                    res.get("traders", {}).get(t.name, {}).get("pnl", 0), 0
                )
            per_trader["DESK"] = round(res.get("desk_pnl", 0), 0)
            stress_out[scenario] = per_trader
        emit({"scene": "stress", "scenarios": stress_out})

    emit_stress()

    # ── Signals (initial) ──
    def emit_signals():
        sig_gens = {
            "momentum": MomentumSignal(tickers=EQUITY_TICKERS[:8]),
            "stat_arb": StatArbSignal(pairs=[
                ("GOOGL", "AMZN"), ("AAPL", "MSFT"), ("NVDA", "AMD"),
                ("JPM", "GS"), ("XOM", "CVX"),
            ]),
            "macro": MacroSignal(),
        }
        all_signals = []
        for name, gen in sig_gens.items():
            for sig in gen.generate_signals(state):
                all_signals.append({
                    "strategy": name, "instrument": sig.instrument,
                    "side": sig.side, "qty": sig.quantity,
                    "strength": round(sig.strength, 3),
                    "reason": sig.reason[:60],
                })
        emit({"scene": "signals", "n_signals": len(all_signals), "signals": all_signals})

    emit_signals()

    # ── Covariance + Optimization (once — these use historical data) ──
    tickers_opt = [t for t in EQUITY_TICKERS[:8] if t in state["equities"]]
    cov_builder = CovarianceBuilder()
    cov_data = cov_builder.build(mgr, tickers_opt)

    for method_name, opt_cls in [("mv", MeanVarianceOptimizer),
                                  ("rp", RiskParityOptimizer),
                                  ("bl", BlackLittermanModel)]:
        try:
            opt = opt_cls(cov_data)
            result = opt.optimize() if method_name == "rp" else opt.optimize(long_only=True)
            emit({
                "scene": f"optimize_{method_name}",
                "backend": "native",
                "weights": {t: round(float(w), 4) for t, w in zip(result.tickers, result.weights)},
                "ret": round(result.expected_return, 4),
                "vol": round(result.expected_vol, 4),
                "sharpe": round(result.sharpe_ratio, 3),
            })
        except Exception as e:
            emit({"scene": f"optimize_{method_name}", "error": str(e)})

    # ── Integration-backend optimizers (if available) ──
    from bank_python.optimizer import create_optimizer
    integration_methods = []
    if HAS_PYPFOPT:
        integration_methods += [
            ("hrp", "pypfopt"), ("max_sharpe", "pypfopt"), ("min_volatility", "pypfopt"),
        ]
    if HAS_RISKFOLIO:
        integration_methods += [("cvar", "riskfolio"), ("risk_budget", "riskfolio")]

    for method_name, backend in integration_methods:
        try:
            opt = create_optimizer(cov_data, method=method_name, backend=backend)
            if method_name == "min_volatility":
                result = opt.optimize(objective="min_volatility")
            else:
                result = opt.optimize()
            emit({
                "scene": f"optimize_{method_name}",
                "backend": backend,
                "weights": {t: round(float(w), 4) for t, w in zip(result.tickers, result.weights)},
                "ret": round(result.expected_return, 4),
                "vol": round(result.expected_vol, 4),
                "sharpe": round(result.sharpe_ratio, 3),
            })
        except Exception as e:
            emit({"scene": f"optimize_{method_name}", "backend": backend, "error": str(e)})

    # ── QuantLib pricing comparison (if available) ──
    if HAS_QUANTLIB:
        from bank_python.integrations.quantlib_pricing import QuantLibOption, QuantLibBond
        ql_comparisons = []
        for opt_key in ("aapl_call", "msft_put", "nvda_call", "tsla_call", "meta_call"):
            hand_opt = state.get(opt_key)
            if hand_opt is None or hand_opt.spot_source.value <= 0:
                continue
            ql_opt = QuantLibOption(
                f"QL_{hand_opt.name}", spot_source=hand_opt.spot_source,
                strike=hand_opt.strike, volatility=hand_opt.volatility,
                time_to_expiry=hand_opt.time_to_expiry, is_call=hand_opt.is_call,
                risk_free_rate=0.0,  # match hand-rolled BS which uses r=0
            )
            _ = ql_opt.value
            greeks = ql_opt.greeks()
            ql_comparisons.append({
                "name": hand_opt.name, "hand_rolled": round(hand_opt.value, 4),
                "quantlib": round(ql_opt.value, 4),
                "delta": round(greeks["delta"], 4),
                "gamma": round(greeks["gamma"], 6),
                "vega": round(greeks["vega"], 4),
            })
        for bond_key in ("bond_5y", "bond_10y", "bond_2y"):
            hand_bond = state.get(bond_key)
            if hand_bond is None:
                continue
            ql_bond = QuantLibBond(
                f"QL_{hand_bond.name}", rate_source=hand_bond.rate_source,
                face=hand_bond.face, coupon_rate=hand_bond.coupon_rate,
                maturity=hand_bond.maturity,
            )
            _ = ql_bond.value
            ql_comparisons.append({
                "name": hand_bond.name, "hand_rolled": round(hand_bond.value, 4),
                "quantlib": round(ql_bond.value, 4),
                "duration": round(ql_bond.duration, 4),
                "convexity": round(ql_bond.convexity, 4),
            })
        emit({"scene": "quantlib_comparison", "instruments": ql_comparisons})

    # ── Backtest — all strategies (once) ──
    bt_strategies = {
        "momentum": lambda: MomentumBacktest(EQUITY_TICKERS[:8], lookback=20, top_n=3),
        "vol_arb": lambda: VolArbBacktest(EQUITY_TICKERS[:4]),
        "stat_arb": lambda: StatArbBacktest([("AAPL", "MSFT"), ("TSLA", "NVDA"),
                                              ("GOOGL", "AMZN"), ("JPM", "GS")]),
        "macro": lambda: MacroBacktest(EQUITY_TICKERS[:8]),
    }
    bt = Backtester(barbara=db)
    for name, factory in bt_strategies.items():
        bt.add_strategy(factory(), trader_name=name.title(), initial_capital=1_000_000)
    bt_results = bt.run(EQUITY_TICKERS[:8], "2024-01-01", "2024-12-31")
    for r in (bt_results or []):
        metrics = r.tearsheet.compute()
        emit({
            "scene": "backtest",
            "strategy": r.strategy_name,
            "return": round(metrics.get("total_return", 0), 4),
            "sharpe": round(metrics.get("sharpe_ratio", 0), 3),
            "max_dd": round(metrics.get("max_drawdown", 0), 4),
            "trades": r.trade_count,
            "final_value": round(r.final_value, 0),
        })

    # ═══════════════════════════════════════════════════════════════════════
    # ITERATION LOOP — each iteration simulates one "intraday period"
    # with price moves, risk recalcs, signal regeneration, and trading.
    # ═══════════════════════════════════════════════════════════════════════
    for i in range(iterations):
        iter_rng = np.random.default_rng(master_rng.integers(0, 2**31))

        # ── Simulate intraday price moves ──
        # Intensity ramps: calm markets early, volatile later
        intensity = 0.8 + 0.4 * (i / max(iterations - 1, 1))
        if iter_rng.random() < 0.15:  # 15% chance of a "volatile day"
            intensity *= 2.5
        _jiggle_market(state, iter_rng, intensity=intensity)

        # ── Desk snapshot (every iteration — shows P&L evolution) ──
        # NOTE: Do NOT call snapshot_initial_values() here.
        # PnL accumulates from the initial setup so we can track
        # how the desk value evolves across iterations.
        emit_desk_snapshot(f"desk_i{i}")

        # ── MC simulation — all 3 models ──
        for mi, model in enumerate(("gbm", "heston", "merton")):
            mc_config = MCConfig(
                n_paths=50_000, horizon_days=1,
                random_seed=int(iter_rng.integers(0, 2**31)),
            )
            engine = MonteCarloEngine(
                traders, state, db, mc_config,
                compute_greeks_fn=compute_greeks, model=model,
            )
            results = engine.run_full_simulation()
            dr = results.get("desk_risk", {})
            pnl = results.get("pnl_desk")
            emit({
                "scene": "mc",
                "iter": i,
                "model": model,
                "var95": round(dr.get(0.95, (0, 0))[0], 0),
                "cvar95": round(dr.get(0.95, (0, 0))[1], 0),
                "var99": round(dr.get(0.99, (0, 0))[0], 0),
                "cvar99": round(dr.get(0.99, (0, 0))[1], 0),
                "mean_pnl": round(float(np.mean(pnl)), 0) if pnl is not None else None,
                "std_pnl": round(float(np.std(pnl)), 0) if pnl is not None else None,
                "p_loss": round(float(np.mean(pnl < 0) * 100), 1) if pnl is not None else None,
                "worst": round(float(np.min(pnl)), 0) if pnl is not None else None,
                "best": round(float(np.max(pnl)), 0) if pnl is not None else None,
                "elapsed": round(results.get("elapsed_seconds", 0), 2),
            })

        # ── Stress test (re-run with updated prices) ──
        if i % 3 == 0:  # every 3rd iteration
            emit_stress()

        # ── Regenerate signals (prices changed → new signals) ──
        if i % 2 == 0:  # every 2nd iteration
            # Append current jiggled prices to history so MAs reflect intraday moves
            ts = datetime.now().isoformat()
            for ticker in signal_tickers:
                eq = state["equities"].get(ticker)
                if eq and eq.value > 0:
                    mgr.history.append(ticker, [{
                        "timestamp": ts,
                        "open": eq.value, "high": eq.value,
                        "low": eq.value, "close": eq.value,
                        "volume": 0,
                    }])
            emit_signals()

        # ── Order execution — 4-5 orders per iteration ──
        n_orders = iter_rng.integers(3, 6)
        tickers_pool = EQUITY_TICKERS[:50]
        tickers_to_trade = list(iter_rng.choice(tickers_pool, size=n_orders, replace=False))
        trade_engine = TradingEngine(barbara=db, graph=state["graph"], state=state)
        order_results = []
        for j in range(n_orders):
            trader_obj = iter_rng.choice(traders)
            sym = tickers_to_trade[j]
            side = iter_rng.choice(["BUY", "SELL"])
            # Size orders relative to trader's book to avoid concentration rejects
            book_val = max(abs(trader_obj.book.total_value), 50_000)
            price = state["equities"].get(sym, None)
            if price is None or price.value <= 0:
                continue
            max_notional = book_val * 0.25  # stay under 40% concentration
            max_qty = max(1, int(max_notional / price.value))
            qty = int(iter_rng.integers(1, max(2, max_qty)))
            # Mix MARKET and LIMIT orders
            is_limit = iter_rng.random() < 0.3
            order_type = "LIMIT" if is_limit else "MARKET"
            limit_price = round(price.value * (1 + iter_rng.uniform(-0.02, 0.02)), 2) if is_limit else None
            try:
                order = trade_engine.submit_order(
                    trader=trader_obj, instrument_name=sym.upper(),
                    side=side, quantity=qty, order_type=order_type,
                    limit_price=limit_price,
                    book_value=book_val,
                )
                order_results.append({
                    "trader": trader_obj.name, "sym": sym, "side": side,
                    "qty": qty, "type": order_type, "status": order.status,
                    "fill": round(order.filled_price, 2) if order.filled_price else None,
                    "limit": limit_price,
                })
            except Exception as e:
                order_results.append({
                    "trader": trader_obj.name, "sym": sym,
                    "error": str(e)[:40],
                })
        emit({"scene": "orders", "iter": i, "orders": order_results})

    elapsed = round(time.time() - t0, 2)
    emit({"scene": "done", "iterations": iterations, "total_sec": elapsed})
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prop Desk Platform Showcase")
    parser.add_argument("--pause", type=int, default=8,
                        help="Seconds between scenes (default: 8)")
    parser.add_argument("--once", action="store_true",
                        help="Single pass, no loop")
    parser.add_argument("--plain", action="store_true",
                        help="Plain text output (no color, no pauses, single pass)")
    parser.add_argument("--dense", type=int, default=0, metavar="N",
                        help="Machine-readable JSON lines mode, N MC iterations")
    args = parser.parse_args()
    if args.dense:
        run_dense(iterations=args.dense)
    else:
        run_showcase(pause=args.pause, once=args.once, plain=args.plain)

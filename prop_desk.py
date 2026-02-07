#!/usr/bin/env python3
"""
Prop Trading Desk Tracker — Multi-trader, multi-strategy risk dashboard.

Showcases the full bank_python stack (Barbara, Dagger, MnTable, Walpole)
with 4 traders running different strategies, live market data, Greeks-based
risk monitoring, and a Rich Live dashboard.

Usage:
    python prop_desk.py                           # one-shot desk report
    python prop_desk.py --live                    # live 5-panel dashboard
    python prop_desk.py trade TRADER SYM QTY      # enter an equity trade
    python prop_desk.py risk                      # per-trader risk + Greeks
    python prop_desk.py pnl                       # P&L breakdown
    python prop_desk.py config [--set KEY VALUE]  # show/edit risk limits
    python prop_desk.py mc [--paths N] [--horizon D]  # Monte Carlo analysis
    python prop_desk.py stress                    # stress scenarios only
"""

import argparse
import json
import logging
import math
import random
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.text import Text

from bank_python import (
    BarbaraDB,
    Book,
    Bond,
    DependencyGraph,
    Equity,
    FXRate,
    JobConfig,
    JobMode,
    JobRunner,
    JobStatus,
    MarketData,
    MarketDataManager,
    Option,
    Position,
    Table,
)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("prop_desk")

# ── Constants ────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "personal_doNOTputongithub"
DB_PATH = DATA_DIR / "prop_desk.db"

EQUITY_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "SPY"]
FX_TICKERS = ["EURUSD=X", "GBPUSD=X"]
VIX_TICKER = "^VIX"
ALL_TICKERS = EQUITY_TICKERS + FX_TICKERS + [VIX_TICKER]

SOFR_URL = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/1.json"

SPARK_CHARS = "▁▂▃▄▅▆▇█"

DEFAULT_CONFIG = {
    "desk_loss_limit": -200_000,
    "trader_loss_limit": -50_000,
    "max_delta": 100_000,
    "max_vega": 50_000,
    "max_concentration_pct": 40,
    "max_position_notional": 500_000,
}

RISK_RULES = [
    ("DAILY_LOSS", "trader P&L < trader_loss_limit", "CRITICAL"),
    ("DESK_LOSS", "desk P&L < desk_loss_limit", "CRITICAL"),
    ("DELTA_LIMIT", "abs(trader delta) > max_delta", "HIGH"),
    ("VEGA_LIMIT", "abs(trader vega) > max_vega", "HIGH"),
    ("CONCENTRATION", "single name > max_concentration_pct% of book", "MEDIUM"),
    ("POSITION_SIZE", "single name notional > max_position_notional", "MEDIUM"),
]


# ── Trader dataclass ─────────────────────────────────────────────────────

@dataclass
class Trader:
    name: str
    strategy: str
    book: Book
    initial_value: float = 0.0


# ── SOFR fetch ───────────────────────────────────────────────────────────

_sofr_cache = {"value": 0.053, "fetched_at": 0.0}


def fetch_sofr():
    """Fetch latest SOFR from NY Fed API. Returns rate as decimal (e.g. 0.053)."""
    now = time.time()
    if now - _sofr_cache["fetched_at"] < 3600:
        return _sofr_cache["value"]
    try:
        req = urllib.request.Request(SOFR_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            rate_pct = float(data["refRates"][0]["percentRate"])
            rate = rate_pct / 100.0
            _sofr_cache["value"] = rate
            _sofr_cache["fetched_at"] = now
            return rate
    except Exception as e:
        log.warning(f"SOFR fetch failed: {e}, using cached {_sofr_cache['value']}")
        return _sofr_cache["value"]


# ── Greeks computation ───────────────────────────────────────────────────

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def compute_greeks(position):
    """
    Compute delta, gamma, vega for a position.

    Options use Black-Scholes Greeks from the Option's parameters.
    Equities/FX: delta = quantity, gamma = 0, vega = 0.
    Bonds: delta = -duration * value * quantity (simplified).
    """
    inst = position.instrument
    qty = position.quantity

    if isinstance(inst, Option):
        S = inst.spot_source.value
        K = inst.strike
        sigma = inst.volatility
        T = inst.time_to_expiry

        if T <= 0 or sigma <= 0 or S <= 0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0}

        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))

        nd1 = _norm_cdf(d1)
        npd1 = _norm_pdf(d1)

        if inst.is_call:
            delta_per = nd1
        else:
            delta_per = nd1 - 1.0

        gamma_per = npd1 / (S * sigma * math.sqrt(T))
        vega_per = S * npd1 * math.sqrt(T) / 100.0  # per 1% vol move

        # Scale by quantity and contract multiplier (100 shares per contract)
        mult = 100.0
        return {
            "delta": delta_per * qty * mult,
            "gamma": gamma_per * qty * mult,
            "vega": vega_per * qty * mult,
        }

    elif isinstance(inst, Bond):
        # Simplified: delta ~ -modified_duration * PV * qty
        rate = inst.rate_source.value
        dur = sum(t / (1 + rate) ** t for t in range(1, inst.maturity + 1))
        dur /= max(inst.value, 0.01)
        return {
            "delta": -dur * inst.value * qty,
            "gamma": 0.0,
            "vega": 0.0,
        }

    else:
        # Equity, FX, etc: delta = notional value
        return {
            "delta": inst.value * qty,
            "gamma": 0.0,
            "vega": 0.0,
        }


def compute_trader_risk(trader):
    """Aggregate Greeks across all positions for a trader."""
    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for pos in trader.book.positions:
        g = compute_greeks(pos)
        for k in totals:
            totals[k] += g[k]
    return totals


def aggregate_desk_risk(traders):
    """Sum Greeks across all traders."""
    desk = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for t in traders:
        risk = compute_trader_risk(t)
        for k in desk:
            desk[k] += risk[k]
    return desk


# ── Risk limit checking ─────────────────────────────────────────────────

def check_risk_limits(traders, config, db):
    """Check all risk rules, return list of alert dicts, store in Barbara."""
    alerts = []
    now = datetime.now()
    desk_pnl = sum(t.book.total_value - t.initial_value for t in traders)

    # DESK_LOSS
    if desk_pnl < config["desk_loss_limit"]:
        alerts.append({
            "severity": "CRITICAL",
            "rule": "DESK_LOSS",
            "message": f"Desk P&L ${desk_pnl:+,.0f} < ${config['desk_loss_limit']:,}",
            "timestamp": now.isoformat(),
        })

    for t in traders:
        pnl = t.book.total_value - t.initial_value
        risk = compute_trader_risk(t)

        # DAILY_LOSS
        if pnl < config["trader_loss_limit"]:
            alerts.append({
                "severity": "CRITICAL",
                "rule": "DAILY_LOSS",
                "message": f"{t.name} P&L ${pnl:+,.0f} < ${config['trader_loss_limit']:,}",
                "timestamp": now.isoformat(),
            })

        # DELTA_LIMIT
        if abs(risk["delta"]) > config["max_delta"]:
            alerts.append({
                "severity": "HIGH",
                "rule": "DELTA_LIMIT",
                "message": f"{t.name} |delta| {abs(risk['delta']):,.0f} > {config['max_delta']:,}",
                "timestamp": now.isoformat(),
            })

        # VEGA_LIMIT
        if abs(risk["vega"]) > config["max_vega"]:
            alerts.append({
                "severity": "HIGH",
                "rule": "VEGA_LIMIT",
                "message": f"{t.name} |vega| {abs(risk['vega']):,.0f} > {config['max_vega']:,}",
                "timestamp": now.isoformat(),
            })

        # CONCENTRATION and POSITION_SIZE
        total_book = abs(t.book.total_value) or 1.0
        for pos in t.book.positions:
            mv = abs(pos.market_value)
            pct = mv / total_book * 100
            if pct > config["max_concentration_pct"]:
                alerts.append({
                    "severity": "MEDIUM",
                    "rule": "CONCENTRATION",
                    "message": f"{t.name} {pos.instrument.name} {pct:.0f}% > {config['max_concentration_pct']}%",
                    "timestamp": now.isoformat(),
                })
            if mv > config["max_position_notional"]:
                alerts.append({
                    "severity": "MEDIUM",
                    "rule": "POSITION_SIZE",
                    "message": f"{t.name} {pos.instrument.name} ${mv:,.0f} > ${config['max_position_notional']:,}",
                    "timestamp": now.isoformat(),
                })

    # Store alerts in Barbara
    date_key = now.strftime("%Y-%m-%d")
    for i, alert in enumerate(alerts):
        bk = f"/Risk/alerts/{date_key}/{now.strftime('%H%M%S')}_{i}"
        db[bk] = alert

    return alerts


# ── Sparkline helper ─────────────────────────────────────────────────────

def sparkline(values, width=20):
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        SPARK_CHARS[min(len(SPARK_CHARS) - 1, int((v - lo) / span * (len(SPARK_CHARS) - 1)))]
        for v in vals
    )


# ── Barbara setup & desk initialization ─────────────────────────────────

def open_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return BarbaraDB.open("prop_desk;default", db_path=str(DB_PATH))


def ensure_config(db):
    for key, default in DEFAULT_CONFIG.items():
        bk = f"/Config/{key}"
        if bk not in db:
            db[bk] = default


def get_config(db):
    cfg = {}
    for key in DEFAULT_CONFIG:
        cfg[key] = db.get(f"/Config/{key}", DEFAULT_CONFIG[key])
    return cfg


def setup_desk(db):
    """
    Create market data nodes, instruments, traders with seed positions,
    wire up MarketDataManager and DependencyGraph. Returns all state.
    """
    graph = DependencyGraph()

    # -- MarketData leaf nodes --
    spots = {}
    for ticker in EQUITY_TICKERS:
        md = MarketData(f"{ticker}_SPOT", price=0.0)
        spots[ticker] = md

    eurusd_md = MarketData("EURUSD_RATE", price=1.08)
    gbpusd_md = MarketData("GBPUSD_RATE", price=1.27)
    vix_md = MarketData("VIX", price=18.0)
    sofr_md = MarketData("SOFR", price=fetch_sofr())
    spread_md = MarketData("CREDIT_SPREAD", price=0.02)

    # -- Instruments --
    equities = {}
    for ticker in EQUITY_TICKERS:
        eq = Equity(ticker, spot_source=spots[ticker])
        equities[ticker] = eq

    eurusd = FXRate("EURUSD", rate_source=eurusd_md, base_currency="EUR", quote_currency="USD")
    gbpusd = FXRate("GBPUSD", rate_source=gbpusd_md, base_currency="GBP", quote_currency="USD")

    # Options for Bob (vol_arb)
    aapl_call = Option("AAPL_C230", spot_source=spots["AAPL"], strike=230.0,
                       volatility=0.25, time_to_expiry=0.5, is_call=True)
    msft_put = Option("MSFT_P400", spot_source=spots["MSFT"], strike=400.0,
                      volatility=0.22, time_to_expiry=0.5, is_call=False)

    # Bond for Diana (macro)
    bond_5y = Bond("BOND_5Y", rate_source=sofr_md, face=100.0, coupon_rate=0.06, maturity=5)

    # Register all in graph
    all_instruments = (
        list(spots.values())
        + [eurusd_md, gbpusd_md, vix_md, sofr_md, spread_md]
        + list(equities.values())
        + [eurusd, gbpusd, aapl_call, msft_put, bond_5y]
    )
    for inst in all_instruments:
        graph.register(inst)

    # -- MarketDataManager --
    mgr = MarketDataManager(graph, db, cache_ttl=30.0)
    for ticker in EQUITY_TICKERS:
        mgr.register_ticker(ticker, spots[ticker])
    mgr.register_ticker("EURUSD=X", eurusd_md)
    mgr.register_ticker("GBPUSD=X", gbpusd_md)
    mgr.register_ticker("^VIX", vix_md)

    # -- Traders with seed positions --
    # Alice: momentum
    alice_book = Book("Alice")
    alice_book.add_position(Position(equities["AAPL"], 200))
    alice_book.add_position(Position(equities["NVDA"], 100))
    alice_book.add_position(Position(equities["TSLA"], 50))

    # Bob: vol_arb (short options)
    bob_book = Book("Bob")
    bob_book.add_position(Position(aapl_call, -10))
    bob_book.add_position(Position(msft_put, -5))

    # Charlie: stat_arb
    charlie_book = Book("Charlie")
    charlie_book.add_position(Position(equities["GOOGL"], 100))
    charlie_book.add_position(Position(equities["AMZN"], -80))
    charlie_book.add_position(Position(equities["META"], 60))

    # Diana: macro
    diana_book = Book("Diana")
    diana_book.add_position(Position(eurusd, 50_000))
    diana_book.add_position(Position(bond_5y, 500))
    diana_book.add_position(Position(equities["SPY"], 100))

    traders = [
        Trader("Alice", "momentum", alice_book),
        Trader("Bob", "vol_arb", bob_book),
        Trader("Charlie", "stat_arb", charlie_book),
        Trader("Diana", "macro", diana_book),
    ]

    # Store trader profiles in Barbara
    for t in traders:
        db[f"/Desk/traders/{t.name}"] = {"name": t.name, "strategy": t.strategy}

    # -- Trade blotter (MnTable) --
    blotter = Table([
        ("trade_id", int),
        ("trader", str),
        ("instrument", str),
        ("quantity", float),
        ("price", float),
        ("timestamp", str),
    ], name="trade_blotter")
    blotter.create_index("trader")
    blotter.create_index("instrument")

    # Seed blotter entries
    trade_id = 0
    now_str = datetime.now().isoformat()
    for t in traders:
        for pos in t.book.positions:
            trade_id += 1
            blotter.append({
                "trade_id": trade_id,
                "trader": t.name,
                "instrument": pos.instrument.name,
                "quantity": float(pos.quantity),
                "price": 0.0,  # updated after first market data fetch
                "timestamp": now_str,
            })

    return {
        "graph": graph,
        "mgr": mgr,
        "traders": traders,
        "blotter": blotter,
        "spots": spots,
        "equities": equities,
        "sofr_md": sofr_md,
        "spread_md": spread_md,
        "vix_md": vix_md,
        "eurusd_md": eurusd_md,
        "gbpusd_md": gbpusd_md,
        "aapl_call": aapl_call,
        "msft_put": msft_put,
        "bond_5y": bond_5y,
        "eurusd": eurusd,
        "gbpusd": gbpusd,
    }


def snapshot_initial_values(traders):
    """Capture initial book values for P&L calculation."""
    for t in traders:
        t.initial_value = t.book.total_value


# ── CLI commands ─────────────────────────────────────────────────────────

def cmd_trade(db, state, trader_name, sym, qty):
    """Add an equity trade to a trader's book."""
    traders = state["traders"]
    equities = state["equities"]
    blotter = state["blotter"]
    graph = state["graph"]

    trader = None
    for t in traders:
        if t.name.lower() == trader_name.lower():
            trader = t
            break
    if trader is None:
        print(f"Unknown trader: {trader_name}. Options: {[t.name for t in traders]}")
        return

    sym_upper = sym.upper()
    if sym_upper not in equities:
        print(f"Unknown symbol: {sym}. Options: {list(equities.keys())}")
        return

    eq = equities[sym_upper]
    pos = Position(eq, int(qty))
    trader.book.add_position(pos)

    # Record in blotter
    tid = len(blotter) + 1
    blotter.append({
        "trade_id": tid,
        "trader": trader.name,
        "instrument": sym_upper,
        "quantity": float(qty),
        "price": eq.value,
        "timestamp": datetime.now().isoformat(),
    })

    # Store in Barbara
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    db[f"/Desk/trades/{trader.name}/{ts}"] = {
        "instrument": sym_upper,
        "quantity": int(qty),
        "price": eq.value,
    }

    print(f"Trade recorded: {trader.name} {sym_upper} {qty:+d} @ ${eq.value:.2f}")


def cmd_risk(state, config):
    """Display per-trader risk and Greeks."""
    console = Console(width=100)
    traders = state["traders"]

    tbl = RichTable(title="Per-Trader Risk & Greeks", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Trader", style="bold", width=10)
    tbl.add_column("Strategy", width=10)
    tbl.add_column("Delta", justify="right", width=12)
    tbl.add_column("Gamma", justify="right", width=10)
    tbl.add_column("Vega", justify="right", width=10)
    tbl.add_column("Book MV", justify="right", width=14)
    tbl.add_column("P&L", justify="right", width=12)

    desk_risk = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for t in traders:
        risk = compute_trader_risk(t)
        pnl = t.book.total_value - t.initial_value
        for k in desk_risk:
            desk_risk[k] += risk[k]

        # Color delta/vega if breached
        delta_txt = f"{risk['delta']:+,.0f}"
        if abs(risk["delta"]) > config["max_delta"]:
            delta_txt = Text(delta_txt, style="red bold")
        vega_txt = f"{risk['vega']:+,.0f}"
        if abs(risk["vega"]) > config["max_vega"]:
            vega_txt = Text(vega_txt, style="red bold")

        pnl_style = "green" if pnl >= 0 else "red"
        tbl.add_row(
            t.name, t.strategy,
            delta_txt, f"{risk['gamma']:+,.1f}", vega_txt,
            f"${t.book.total_value:,.0f}",
            Text(f"${pnl:+,.0f}", style=pnl_style),
        )

    # Desk total
    tbl.add_row("", "", "", "", "", "", "")
    desk_pnl = sum(t.book.total_value - t.initial_value for t in traders)
    pnl_style = "green" if desk_pnl >= 0 else "red"
    tbl.add_row(
        Text("DESK", style="bold"), "",
        f"{desk_risk['delta']:+,.0f}",
        f"{desk_risk['gamma']:+,.1f}",
        f"{desk_risk['vega']:+,.0f}",
        f"${sum(t.book.total_value for t in traders):,.0f}",
        Text(f"${desk_pnl:+,.0f}", style=f"bold {pnl_style}"),
    )

    console.print(Panel(tbl, border_style="cyan"))


def cmd_pnl(state):
    """Display P&L breakdown."""
    console = Console(width=100)
    traders = state["traders"]

    tbl = RichTable(title="P&L Breakdown", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Trader", style="bold", width=10)
    tbl.add_column("Strategy", width=10)
    tbl.add_column("Initial MV", justify="right", width=14)
    tbl.add_column("Current MV", justify="right", width=14)
    tbl.add_column("P&L ($)", justify="right", width=14)
    tbl.add_column("P&L (%)", justify="right", width=10)

    total_init = 0.0
    total_curr = 0.0
    for t in traders:
        curr = t.book.total_value
        init = t.initial_value
        pnl = curr - init
        pct = (pnl / abs(init) * 100) if init != 0 else 0.0
        total_init += init
        total_curr += curr

        pnl_style = "green" if pnl >= 0 else "red"
        tbl.add_row(
            t.name, t.strategy,
            f"${init:,.0f}", f"${curr:,.0f}",
            Text(f"${pnl:+,.0f}", style=pnl_style),
            Text(f"{pct:+.2f}%", style=pnl_style),
        )

    tbl.add_row("", "", "", "", "", "")
    desk_pnl = total_curr - total_init
    desk_pct = (desk_pnl / abs(total_init) * 100) if total_init != 0 else 0.0
    pnl_style = "green" if desk_pnl >= 0 else "red"
    tbl.add_row(
        Text("DESK", style="bold"), "",
        f"${total_init:,.0f}", f"${total_curr:,.0f}",
        Text(f"${desk_pnl:+,.0f}", style=f"bold {pnl_style}"),
        Text(f"{desk_pct:+.2f}%", style=f"bold {pnl_style}"),
    )

    console.print(Panel(tbl, border_style="cyan"))

    # Per-position detail
    for t in traders:
        ptbl = RichTable(title=f"{t.name} Positions", expand=True,
                         show_header=True, header_style="dim")
        ptbl.add_column("Instrument", width=12)
        ptbl.add_column("Qty", justify="right", width=8)
        ptbl.add_column("Price", justify="right", width=10)
        ptbl.add_column("MV", justify="right", width=12)

        for pos in t.book.positions:
            mv = pos.market_value
            mv_style = "green" if mv >= 0 else "red"
            ptbl.add_row(
                pos.instrument.name,
                f"{pos.quantity:,}",
                f"${pos.instrument.value:.2f}",
                Text(f"${mv:,.0f}", style=mv_style),
            )
        console.print(ptbl)


def cmd_config(db, console, set_pair=None):
    """Show or edit risk limit configuration."""
    if set_pair:
        key, value = set_pair
        if key not in DEFAULT_CONFIG:
            console.print(f"[red]Unknown key: {key}[/red]")
            console.print(f"Valid keys: {', '.join(DEFAULT_CONFIG.keys())}")
            return
        typed_val = type(DEFAULT_CONFIG[key])(value)
        db[f"/Config/{key}"] = typed_val
        console.print(f"[green]Set {key} = {typed_val}[/green]")
        return

    config = get_config(db)
    tbl = RichTable(title="Prop Desk Risk Limits", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Key", style="bold", width=26)
    tbl.add_column("Value", justify="right", width=14)
    tbl.add_column("Default", justify="right", width=14, style="dim")

    for key, default in DEFAULT_CONFIG.items():
        val = config[key]
        style = "" if val == default else "yellow"
        if isinstance(val, int) and abs(val) >= 1000:
            tbl.add_row(key, Text(f"${val:,}", style=style), f"${default:,}")
        else:
            tbl.add_row(key, Text(str(val), style=style), str(default))

    console.print(Panel(tbl, border_style="cyan"))
    console.print("[dim]Edit: python prop_desk.py config --set KEY VALUE[/dim]")


# ── One-shot report ──────────────────────────────────────────────────────

def one_shot_report(db, state, config, console):
    """Print a full desk report: positions, risk, P&L, alerts."""
    traders = state["traders"]

    console.print(Panel(
        "[bold]PROP TRADING DESK[/bold]\n[dim]Multi-strategy risk dashboard[/dim]",
        style="cyan", expand=False,
    ))

    # Market data summary
    mgr = state["mgr"]
    tbl = RichTable(title="Market Data", expand=True, show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Ticker", style="bold", width=10)
    tbl.add_column("Price", justify="right", width=12)

    for ticker in ALL_TICKERS:
        node = mgr.registered_tickers.get(ticker)
        if node and node.value:
            if ticker.endswith("=X"):
                tbl.add_row(ticker.replace("=X", ""), f"{node.value:.4f}")
            elif ticker.startswith("^"):
                tbl.add_row(ticker, f"{node.value:.2f}")
            else:
                tbl.add_row(ticker, f"${node.value:.2f}")
        else:
            tbl.add_row(ticker, "[dim]—[/dim]")

    tbl.add_row("SOFR", f"{state['sofr_md'].value:.4f}")
    tbl.add_row("Spread", f"{state['spread_md'].value:.4f}")
    console.print(Panel(tbl, border_style="blue"))

    # Positions by trader
    for t in traders:
        ptbl = RichTable(title=f"{t.name} ({t.strategy})", expand=True,
                         show_header=True, header_style="dim")
        ptbl.add_column("Instrument", width=14)
        ptbl.add_column("Qty", justify="right", width=8)
        ptbl.add_column("Price", justify="right", width=10)
        ptbl.add_column("MV", justify="right", width=14)

        for pos in t.book.positions:
            mv = pos.market_value
            mv_style = "green" if mv >= 0 else "red"
            ptbl.add_row(
                pos.instrument.name,
                f"{pos.quantity:,}",
                f"${pos.instrument.value:.2f}",
                Text(f"${mv:,.0f}", style=mv_style),
            )

        pnl = t.book.total_value - t.initial_value
        pnl_style = "green" if pnl >= 0 else "red"
        ptbl.add_row(
            Text("TOTAL", style="bold"), "", "",
            Text(f"${t.book.total_value:,.0f}", style="bold"),
        )
        ptbl.add_row(
            Text("P&L", style="bold"), "", "",
            Text(f"${pnl:+,.0f}", style=f"bold {pnl_style}"),
        )
        console.print(ptbl)

    # Risk summary
    cmd_risk(state, config)

    # Alerts
    alerts = check_risk_limits(traders, config, db)
    if alerts:
        atbl = RichTable(title="Risk Alerts", expand=True, show_header=False)
        atbl.add_column("Sev", width=10)
        atbl.add_column("Rule", width=16)
        atbl.add_column("Message", width=50)
        for a in alerts:
            sev_style = "red bold" if a["severity"] == "CRITICAL" else (
                "yellow bold" if a["severity"] == "HIGH" else "yellow")
            atbl.add_row(
                Text(f"! {a['severity']}", style=sev_style),
                a["rule"], a["message"],
            )
        console.print(Panel(atbl, border_style="red"))
    else:
        console.print(Panel("[green]No active risk alerts[/green]", border_style="green"))


# ── Live dashboard panels ────────────────────────────────────────────────

def build_market_panel(state, prev_prices):
    """Top-left: Market Data with change arrows."""
    mgr = state["mgr"]
    tbl = RichTable(title="Market Data", expand=True, show_header=True,
                    header_style="bold cyan", padding=(0, 1))
    tbl.add_column("Ticker", style="bold", width=8)
    tbl.add_column("Price", justify="right", width=10)
    tbl.add_column("Chg", justify="right", width=8)

    for ticker in ALL_TICKERS:
        node = mgr.registered_tickers.get(ticker)
        if not node or node.value == 0:
            tbl.add_row(ticker[:8], "—", "")
            continue

        price = node.value
        prev = prev_prices.get(ticker)
        label = ticker.replace("=X", "").replace("^", "")

        if ticker.endswith("=X"):
            price_str = f"{price:.4f}"
        else:
            price_str = f"${price:.2f}"

        if prev and prev != 0:
            pct = (price - prev) / prev * 100
            if pct >= 0:
                chg = Text(f"▲{pct:+.1f}%", style="green")
            else:
                chg = Text(f"▼{pct:.1f}%", style="red")
        else:
            chg = Text("")

        tbl.add_row(label, price_str, chg)

    tbl.add_row("SOFR", f"{state['sofr_md'].value:.4f}", Text(""))
    tbl.add_row("VIX", f"{state['vix_md'].value:.2f}", Text(""))

    return Panel(tbl, border_style="blue", title="Market Data")


def build_pnl_panel(traders):
    """Top-center: Desk P&L summary."""
    tbl = RichTable(title="Desk P&L", expand=True, show_header=True,
                    header_style="bold cyan", padding=(0, 1))
    tbl.add_column("Trader", style="bold", width=10)
    tbl.add_column("P&L", justify="right", width=14)

    desk_pnl = 0.0
    for t in traders:
        pnl = t.book.total_value - t.initial_value
        desk_pnl += pnl
        pnl_style = "green" if pnl >= 0 else "red"
        tbl.add_row(t.name, Text(f"${pnl:+,.0f}", style=pnl_style))

    tbl.add_row("", "")
    pnl_style = "green" if desk_pnl >= 0 else "red"
    tbl.add_row(
        Text("DESK", style="bold"),
        Text(f"${desk_pnl:+,.0f}", style=f"bold {pnl_style}"),
    )

    return Panel(tbl, border_style="blue", title="Desk P&L")


def build_risk_panel(traders, config, mc_results=None):
    """Top-right: Risk dashboard with Greeks and MC VaR (if available)."""
    desk_risk = aggregate_desk_risk(traders)

    tbl = RichTable(title="Risk Dashboard", expand=True, show_header=True,
                    header_style="bold cyan", padding=(0, 1))
    tbl.add_column("Metric", style="bold", width=10)
    tbl.add_column("Value", justify="right", width=12)
    tbl.add_column("Limit", justify="right", width=12, style="dim")

    # Desk delta
    d_style = "red" if abs(desk_risk["delta"]) > config["max_delta"] * len(traders) else ""
    tbl.add_row("Desk Delta", Text(f"{desk_risk['delta']:+,.0f}", style=d_style), "")
    tbl.add_row("Desk Gamma", f"{desk_risk['gamma']:+,.1f}", "")

    v_style = "red" if abs(desk_risk["vega"]) > config["max_vega"] * len(traders) else ""
    tbl.add_row("Desk Vega", Text(f"{desk_risk['vega']:+,.0f}", style=v_style), "")

    # MC VaR/CVaR if available
    if mc_results:
        tbl.add_row("", "", "")
        dr = mc_results.get("desk_risk", {})
        for conf in sorted(dr.keys()):
            var, cvar = dr[conf]
            tbl.add_row(f"VaR {conf:.0%}", Text(f"${var:,.0f}", style="yellow"), "")
            tbl.add_row(f"CVaR {conf:.0%}", f"${cvar:,.0f}", "")
        pnl = mc_results.get("pnl_desk")
        if pnl is not None:
            import numpy as np
            prob_loss = np.mean(pnl < 0) * 100
            tbl.add_row("P(Loss)", f"{prob_loss:.1f}%", "")

    tbl.add_row("", "", "")
    # Per-trader delta summary
    for t in traders:
        risk = compute_trader_risk(t)
        d_s = "red" if abs(risk["delta"]) > config["max_delta"] else ""
        tbl.add_row(
            f"{t.name} Δ",
            Text(f"{risk['delta']:+,.0f}", style=d_s),
            f"{config['max_delta']:,}",
        )

    return Panel(tbl, border_style="blue", title="Risk Dashboard")


def build_positions_panel(traders):
    """Bottom-left: All positions across traders."""
    tbl = RichTable(title="Positions", expand=True, show_header=True,
                    header_style="bold cyan", padding=(0, 1))
    tbl.add_column("Trader", style="bold", width=8)
    tbl.add_column("Inst", width=10)
    tbl.add_column("Qty", justify="right", width=7)
    tbl.add_column("MV", justify="right", width=12)

    for t in traders:
        for pos in t.book.positions:
            mv = pos.market_value
            mv_style = "green" if mv >= 0 else "red"
            qty_str = f"{pos.quantity:,}"
            tbl.add_row(
                t.name, pos.instrument.name, qty_str,
                Text(f"${mv:,.0f}", style=mv_style),
            )

    return Panel(tbl, border_style="blue", title="Positions")


def build_alerts_status_panel(alerts, runner, job_names, start_time, db):
    """Bottom-right: Alerts + Walpole job status."""
    tbl = RichTable(title="Alerts & Status", expand=True, show_header=False,
                    padding=(0, 1))
    tbl.add_column("Info", width=50)

    status_styles = {
        JobStatus.SUCCEEDED: ("●", "green"),
        JobStatus.RUNNING: ("●", "yellow"),
        JobStatus.FAILED: ("●", "red"),
        JobStatus.PENDING: ("○", "dim"),
        JobStatus.STOPPED: ("●", "dim"),
    }

    # Alerts section
    if alerts:
        for a in alerts[:6]:
            sev = a["severity"]
            sev_style = "red bold" if sev == "CRITICAL" else (
                "yellow bold" if sev == "HIGH" else "yellow")
            tbl.add_row(Text(f"! {sev}  {a['message']}", style=sev_style))
    else:
        tbl.add_row(Text("No active alerts", style="green"))

    tbl.add_row(Text(""))

    # Job status
    for name in job_names:
        status = runner.get_status(name)
        if status is None:
            status = JobStatus.PENDING
        dot, style = status_styles.get(status, ("?", "dim"))
        tbl.add_row(Text(f"{name:12s} {dot} {status.value.upper()}", style=style))

    uptime = int(time.time() - start_time)
    n_keys = len(db.keys(prefix="/"))
    tbl.add_row(Text(f"Uptime: {uptime}s  DB: {n_keys} keys", style="dim"))

    border = "red" if alerts and any(a["severity"] == "CRITICAL" for a in alerts) else (
        "yellow" if alerts else "green")
    return Panel(tbl, border_style=border, title="Alerts & Status")


# ── Live dashboard ───────────────────────────────────────────────────────

def run_live_dashboard(db, state, config, console):
    """Run the 5-panel Rich Live dashboard with Walpole background jobs."""
    traders = state["traders"]
    mgr = state["mgr"]
    graph = state["graph"]
    sofr_md = state["sofr_md"]
    spread_md = state["spread_md"]

    db_lock = threading.Lock()
    prev_prices = {}
    alerts_state = {"current": []}
    sofr_fetch_time = {"last": 0.0}
    mc_state = {"results": None, "lock": threading.Lock()}

    # -- Walpole job functions --
    def job_mkt_data():
        """Fetch live prices from yfinance, SOFR hourly, jiggle spreads."""
        # Save previous for change arrows
        for ticker in mgr.registered_tickers:
            node = mgr.registered_tickers[ticker]
            if node.value:
                prev_prices[ticker] = node.value

        with db_lock:
            mgr.update_all()

        # SOFR (hourly)
        now = time.time()
        if now - sofr_fetch_time["last"] > 3600:
            rate = fetch_sofr()
            sofr_md.set_price(rate)
            graph.recalculate(sofr_md)
            sofr_fetch_time["last"] = now
            with db_lock:
                db["/MarketData/sofr/latest"] = rate

        # Jiggle credit spread (random walk)
        old_spread = spread_md.value
        new_spread = max(0.005, old_spread + random.uniform(-0.001, 0.001))
        spread_md.set_price(new_spread)
        graph.recalculate(spread_md)

        return {"status": "ok"}

    def job_revalue():
        """Recompute all trader books."""
        # Touching .value on each position triggers lazy recompute
        for t in traders:
            _ = t.book.total_value
        return {"desk_value": sum(t.book.total_value for t in traders)}

    def job_risk_check():
        """Check risk limits and store alerts."""
        with db_lock:
            alerts_state["current"] = check_risk_limits(traders, config, db)
        return alerts_state["current"]

    def job_snapshot():
        """Store full risk snapshot to Barbara."""
        desk_risk = aggregate_desk_risk(traders)
        snap = {
            "timestamp": datetime.now().isoformat(),
            "desk_risk": desk_risk,
            "desk_pnl": sum(t.book.total_value - t.initial_value for t in traders),
            "traders": {
                t.name: {
                    "pnl": t.book.total_value - t.initial_value,
                    "book_value": t.book.total_value,
                    "risk": compute_trader_risk(t),
                }
                for t in traders
            },
        }
        now = datetime.now()
        with db_lock:
            db[f"/Risk/snapshots/{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"] = snap
        return snap

    def job_mc_sim():
        """Run lightweight MC simulation for live VaR updates."""
        try:
            from mc_engine import MCConfig, MonteCarloEngine
            mc_config = MCConfig(n_paths=50_000, horizon_days=1)
            engine = MonteCarloEngine(traders, state, db, mc_config)
            results = engine.run_full_simulation()
            with mc_state["lock"]:
                mc_state["results"] = results
            return {"status": "ok", "var_95": results["desk_risk"].get(0.95, (0, 0))[0]}
        except Exception as e:
            log.warning(f"MC sim failed: {e}")
            return {"status": "error", "error": str(e)}

    runner = JobRunner(barbara=db)
    runner.add_job(JobConfig(
        name="mkt_data", callable=job_mkt_data,
        mode=JobMode.PERIODIC, interval=15.0,
    ))
    runner.add_job(JobConfig(
        name="revalue", callable=job_revalue,
        mode=JobMode.PERIODIC, interval=20.0,
        depends_on=["mkt_data"],
    ))
    runner.add_job(JobConfig(
        name="risk_check", callable=job_risk_check,
        mode=JobMode.PERIODIC, interval=30.0,
        depends_on=["revalue"],
    ))
    runner.add_job(JobConfig(
        name="snapshot", callable=job_snapshot,
        mode=JobMode.PERIODIC, interval=300.0,
    ))
    runner.add_job(JobConfig(
        name="mc_sim", callable=job_mc_sim,
        mode=JobMode.PERIODIC, interval=120.0,
        depends_on=["revalue"],
    ))

    job_names = ["mkt_data", "revalue", "risk_check", "snapshot", "mc_sim"]

    console.print("[bold cyan]Starting live dashboard (Ctrl+C to stop)...[/bold cyan]\n")
    runner.start()
    start_time = time.time()

    def build_dashboard():
        layout = Layout()
        layout.split_column(
            Layout(name="top", ratio=1),
            Layout(name="bottom", ratio=1),
        )
        with mc_state["lock"]:
            current_mc = mc_state["results"]
        layout["top"].split_row(
            Layout(build_market_panel(state, prev_prices), name="market"),
            Layout(build_pnl_panel(traders), name="pnl"),
            Layout(build_risk_panel(traders, config, mc_results=current_mc), name="risk"),
        )
        layout["bottom"].split_row(
            Layout(build_positions_panel(traders), name="positions"),
            Layout(build_alerts_status_panel(
                alerts_state["current"], runner, job_names, start_time, db,
            ), name="alerts"),
        )
        return layout

    try:
        with Live(build_dashboard(), console=console, refresh_per_second=0.5, screen=True) as live:
            while True:
                time.sleep(2)
                live.update(build_dashboard())
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        console.print("\n[bold cyan]Dashboard stopped.[/bold cyan]")
        uptime = int(time.time() - start_time)
        console.print(f"  Runtime: {uptime}s")
        desk_pnl = sum(t.book.total_value - t.initial_value for t in traders)
        pnl_style = "green" if desk_pnl >= 0 else "red"
        console.print(f"  Desk P&L: [{pnl_style}]${desk_pnl:+,.0f}[/{pnl_style}]")


# ── Monte Carlo CLI commands ─────────────────────────────────────────────

def cmd_mc(db, state, console, n_paths=100_000, horizon=1):
    """Run full Monte Carlo simulation and display results."""
    from mc_engine import MCConfig, MonteCarloEngine, MCVisualizer

    config = MCConfig(n_paths=n_paths, horizon_days=horizon)
    console.print(f"[dim]Running Monte Carlo simulation ({n_paths:,} paths, "
                  f"{horizon}d horizon)...[/dim]")

    engine = MonteCarloEngine(
        state["traders"], state, db, config,
        compute_greeks_fn=compute_greeks,
    )
    results = engine.run_full_simulation()

    viz = MCVisualizer(console)
    viz.render_full_report(results)
    console.print(f"[dim]Completed in {results['elapsed_seconds']:.1f}s[/dim]")


def cmd_stress(db, state, console):
    """Run stress scenarios and display results."""
    from mc_engine import StressEngine, MCVisualizer

    console.print("[dim]Running stress scenarios...[/dim]")
    engine = StressEngine()
    results = engine.run_all_scenarios(state["traders"], state)

    viz = MCVisualizer(console)
    viz.render_stress_table(results, state["traders"])


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Prop Trading Desk Tracker",
    )
    parser.add_argument("--live", action="store_true",
                        help="Run live 5-panel dashboard")

    sub = parser.add_subparsers(dest="command")

    # trade
    trade_p = sub.add_parser("trade", help="Enter an equity trade")
    trade_p.add_argument("trader", help="Trader name (Alice/Bob/Charlie/Diana)")
    trade_p.add_argument("sym", help="Equity symbol (e.g. AAPL)")
    trade_p.add_argument("qty", type=int, help="Quantity (positive=buy, negative=sell)")

    # risk
    sub.add_parser("risk", help="Show per-trader risk and Greeks")

    # pnl
    sub.add_parser("pnl", help="Show P&L breakdown")

    # config
    cfg_p = sub.add_parser("config", help="Show/edit risk limit config")
    cfg_p.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"),
                       help="Set a config value")

    # mc
    mc_p = sub.add_parser("mc", help="Run Monte Carlo risk analysis")
    mc_p.add_argument("--paths", type=int, default=100_000,
                      help="Number of simulation paths (default: 100000)")
    mc_p.add_argument("--horizon", type=int, default=1,
                      help="Horizon in trading days (default: 1)")

    # stress
    sub.add_parser("stress", help="Run stress scenarios")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    console = Console(width=110)

    db = open_db()
    ensure_config(db)
    config = get_config(db)

    console.print("[dim]Setting up desk infrastructure...[/dim]")
    state = setup_desk(db)
    traders = state["traders"]
    mgr = state["mgr"]

    # Initial market data fetch
    console.print("[dim]Fetching live market data...[/dim]")
    mgr.update_all()

    # Snapshot initial values for P&L after first fetch
    snapshot_initial_values(traders)

    try:
        if args.command == "trade":
            cmd_trade(db, state, args.trader, args.sym, args.qty)

        elif args.command == "risk":
            cmd_risk(state, config)

        elif args.command == "pnl":
            cmd_pnl(state)

        elif args.command == "config":
            cmd_config(db, console, set_pair=args.set)

        elif args.command == "mc":
            cmd_mc(db, state, console, n_paths=args.paths, horizon=args.horizon)

        elif args.command == "stress":
            cmd_stress(db, state, console)

        elif args.live:
            run_live_dashboard(db, state, config, console)

        else:
            # Default: one-shot report
            one_shot_report(db, state, config, console)
    finally:
        db.close()


if __name__ == "__main__":
    main()

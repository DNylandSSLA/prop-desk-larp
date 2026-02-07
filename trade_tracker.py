#!/usr/bin/env python3
"""
Trade Tracker — Persistent trade tracking with Barbara storage.

Tracks a Talebian options strategy in a Roth IRA, classifying trades into
insurance (deep OTM crash puts on broad indices) vs conviction (everything
else). Stores history, checks rules, shows metric trends with sparklines.

Usage:
    python trade_tracker.py                          # one-shot report
    python trade_tracker.py --live                   # live dashboard
    python trade_tracker.py import [--dir PATH]      # import Fidelity CSVs
    python trade_tracker.py log SYM QTY PRICE ...    # manually log a trade
    python trade_tracker.py snapshot                  # force a metric snapshot
    python trade_tracker.py config                    # show/edit config
"""

import argparse
import hashlib
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live

from trade_analysis import (
    parse_option_symbol, parse_dollar, parse_date,
    parse_realized, parse_transactions, parse_current_options,
    compute_convexity, compute_concentration, compute_bleed,
    compute_open_monitor,
)
from bank_python import BarbaraDB, JobRunner, JobConfig, JobMode, JobStatus

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path(__file__).parent / "personal_doNOTputongithub"
DB_PATH = DATA_DIR / "trade_tracker.db"

SPARK_CHARS = "▁▂▃▄▅▆▇█"

INSURANCE_UNDERLYINGS = {"SPY", "QQQ", "IWM", "DIA", "XHB", "KRE", "HYG", "XLF", "LQD"}

DEFAULT_CONFIG = {
    "insurance_budget": 4000,
    "conviction_budget": 7000,
    "monthly_bleed_limit": 700,
    "max_trades_losing": 4,
    "min_convexity_ratio": 2.0,
}


# ── Helpers ──────────────────────────────────────────────────────────────

def sparkline(values, width=8):
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi != lo else 1.0
    return "".join(
        SPARK_CHARS[min(len(SPARK_CHARS) - 1, int((v - lo) / span * (len(SPARK_CHARS) - 1)))]
        for v in vals
    )


def trade_hash(trade):
    key = f"{trade['sym']}|{trade['cost']}|{trade['proceeds']}|{trade['gl']}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def classify_trade(trade):
    """Classify a trade as insurance or conviction."""
    opt = trade.get('opt')
    if opt and opt['type'] == 'put' and opt['underlying'] in INSURANCE_UNDERLYINGS:
        return 'insurance'
    return 'conviction'


def progress_bar(current, total, width=15):
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = min(current / total, 1.0)
    filled = int(ratio * width)
    return "[" + "=" * filled + "-" * (width - filled) + "]"


# ── Barbara Setup ────────────────────────────────────────────────────────

def open_db():
    return BarbaraDB.open("tracker;default", db_path=str(DB_PATH))


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


# ── Import Pipeline ──────────────────────────────────────────────────────

def import_trades(db, data_dir=None, console=None):
    if console is None:
        console = Console()
    src = Path(data_dir) if data_dir else DATA_DIR

    # Find CSVs in directory
    realized_csv = None
    history_csv = None
    positions_csv = None
    for f in sorted(src.iterdir()):
        if not f.suffix.lower() == '.csv':
            continue
        name = f.name.lower()
        if 'history' in name or 'accounts_history' in name:
            history_csv = f
        elif '(1)' in f.name and 'portfolio' in name:
            realized_csv = f
        elif 'portfolio' in name:
            positions_csv = f

    stats = {'realized': 0, 'txns': 0, 'open': 0, 'skipped': 0}

    # Import realized trades
    if realized_csv:
        trades = parse_realized(csv_path=str(realized_csv))
        for t in trades:
            h = trade_hash(t)
            opt = t.get('opt')
            underlying = opt['underlying'] if opt else t['sym'].strip()
            bk = f"/Trades/realized/{underlying}/{h}"
            if bk not in db:
                t['bucket'] = classify_trade(t)
                t['imported_at'] = datetime.now().isoformat()
                db[bk] = t
                stats['realized'] += 1
            else:
                stats['skipped'] += 1
        console.print(f"  Realized: {stats['realized']} imported, {stats['skipped']} dupes skipped")
    else:
        console.print("  [yellow]No realized CSV found[/yellow]")

    # Import transactions
    if history_csv:
        txns = parse_transactions(csv_path=str(history_csv))
        for i, txn in enumerate(txns):
            bk = f"/Trades/txns/{txn['date'].strftime('%Y%m%d')}/{i}"
            if bk not in db:
                # datetime isn't picklable across versions reliably, store iso
                txn_store = {**txn, 'date': txn['date'].isoformat()}
                db[bk] = txn_store
                stats['txns'] += 1
        console.print(f"  Transactions: {stats['txns']} imported")
    else:
        console.print("  [yellow]No transaction history CSV found[/yellow]")

    # Import open positions
    if positions_csv:
        options = parse_current_options(csv_path=str(positions_csv))
        for o in options:
            bk = f"/Trades/open/{o['sym'].strip()}"
            o['bucket'] = classify_trade(o)
            o['imported_at'] = datetime.now().isoformat()
            db[bk] = o
            stats['open'] += 1
        console.print(f"  Open positions: {stats['open']} imported")
    else:
        console.print("  [yellow]No positions CSV found[/yellow]")

    return stats


# ── Manual Trade Entry ───────────────────────────────────────────────────

def log_trade(db, sym, qty, price, action='BUY_OPEN', bucket=None, note=''):
    opt = parse_option_symbol(sym)
    if bucket is None:
        bucket = classify_trade({'opt': opt, 'sym': sym})
    cost = abs(float(qty) * float(price) * 100)  # options are per-contract
    trade = {
        'sym': sym,
        'qty': int(qty),
        'price': float(price),
        'action': action,
        'bucket': bucket,
        'cost': cost,
        'opt': opt,
        'is_option': opt is not None,
        'note': note,
        'logged_at': datetime.now().isoformat(),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    bk = f"/Trades/manual/{ts}"
    db[bk] = trade
    return trade


# ── Metrics Engine ───────────────────────────────────────────────────────

def load_realized_trades(db):
    keys = db.keys(prefix="/Trades/realized/")
    return [db[k] for k in keys]


def load_open_positions(db):
    keys = db.keys(prefix="/Trades/open/")
    return [db[k] for k in keys]


def load_transactions(db):
    keys = db.keys(prefix="/Trades/txns/")
    txns = []
    for k in keys:
        t = db[k]
        if isinstance(t.get('date'), str):
            t['date'] = datetime.fromisoformat(t['date'])
        txns.append(t)
    return txns


def compute_bucket_allocation(db, trades, config):
    buckets = {'insurance': 0.0, 'conviction': 0.0}
    for t in trades:
        b = t.get('bucket', classify_trade(t))
        cost = abs(t.get('cost', 0))
        buckets[b] = buckets.get(b, 0) + cost

    # Also count manual entries
    manual_keys = db.keys(prefix="/Trades/manual/")
    for k in manual_keys:
        t = db[k]
        b = t.get('bucket', 'conviction')
        buckets[b] = buckets.get(b, 0) + abs(t.get('cost', 0))

    return {
        'insurance_deployed': buckets['insurance'],
        'insurance_budget': config['insurance_budget'],
        'insurance_pct': (buckets['insurance'] / config['insurance_budget'] * 100)
                         if config['insurance_budget'] > 0 else 0,
        'conviction_deployed': buckets['conviction'],
        'conviction_budget': config['conviction_budget'],
        'conviction_pct': (buckets['conviction'] / config['conviction_budget'] * 100)
                          if config['conviction_budget'] > 0 else 0,
    }


def compute_decorrelation(trades):
    """Score trades as correlated (equity calls) or decorrelated (puts, vol, FX, commodity)."""
    if not trades:
        return {'decorrelated_pct': 0, 'n_decorrelated': 0, 'n_total': 0}
    decorrelated = 0
    total = 0
    for t in trades:
        opt = t.get('opt')
        if not opt:
            continue
        total += 1
        # Decorrelated: puts, VIX/vol products, FX, commodity
        if opt['type'] == 'put':
            decorrelated += 1
        elif opt['underlying'] in ('UVXY', 'VXX', 'VIXY', 'SVXY', 'SQQQ', 'SH', 'SDS'):
            decorrelated += 1
    pct = (decorrelated / total * 100) if total > 0 else 0
    return {'decorrelated_pct': pct, 'n_decorrelated': decorrelated, 'n_total': total}


def compute_all_metrics(db, config):
    trades = load_realized_trades(db)
    txns = load_transactions(db)
    open_pos = load_open_positions(db)

    cx = compute_convexity(trades) if trades else {}
    conc = compute_concentration(trades) if trades else []
    bleed = compute_bleed(trades, txns) if trades else []
    open_mon = compute_open_monitor(open_pos) if open_pos else []
    bucket = compute_bucket_allocation(db, trades, config)
    decorr = compute_decorrelation(trades)

    return {
        'convexity': cx,
        'concentration': conc,
        'bleed': bleed,
        'open_monitor': open_mon,
        'bucket': bucket,
        'decorrelation': decorr,
        'n_realized': len(trades),
        'n_txns': len(txns),
        'n_open': len(open_pos),
        'timestamp': datetime.now().isoformat(),
    }


# ── Snapshots ────────────────────────────────────────────────────────────

def take_snapshot(db, config):
    metrics = compute_all_metrics(db, config)
    now = datetime.now()
    date_key = now.strftime('%Y-%m-%d')
    time_key = now.strftime('%H%M%S')
    bk = f"/Snapshots/{date_key}/{time_key}"
    db[bk] = metrics
    db["/Snapshots/latest"] = metrics
    return metrics


def load_snapshot_history(db, limit=30):
    keys = db.keys(prefix="/Snapshots/")
    # Filter out /Snapshots/latest
    keys = [k for k in keys if k != "/Snapshots/latest"]
    keys.sort()
    keys = keys[-limit:]
    return [db[k] for k in keys]


def extract_sparkline_series(snapshots, path):
    """Extract a numeric series from snapshot history for sparkline display."""
    values = []
    for s in snapshots:
        val = s
        for part in path:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        if val is not None and isinstance(val, (int, float)):
            values.append(val)
    return values


# ── Rule Checking ────────────────────────────────────────────────────────

def check_rules(db, metrics, config):
    alerts = []
    now = datetime.now()

    # OVERTRADE: >N trades on a losing name
    conc = metrics.get('concentration', [])
    max_losing = config['max_trades_losing']
    for c in conc:
        if c['n'] > max_losing and c['roi'] < 0:
            alerts.append({
                'severity': 'HIGH',
                'rule': 'OVERTRADE',
                'message': f">{max_losing} trades on {c['name']} with {c['roi']:+.0f}% ROI",
                'timestamp': now.isoformat(),
            })

    # BLEED_OVER: monthly bleed exceeds limit
    bleed = metrics.get('bleed', [])
    if bleed:
        current_month = now.strftime('%Y-%m')
        for m in bleed:
            if m['month'] == current_month and abs(m['net']) > config['monthly_bleed_limit']:
                alerts.append({
                    'severity': 'HIGH',
                    'rule': 'BLEED_OVER',
                    'message': f"Monthly bleed ${m['net']:+,.0f} exceeds ${config['monthly_bleed_limit']} limit",
                    'timestamp': now.isoformat(),
                })

    # BUCKET_OVERSPEND
    bucket = metrics.get('bucket', {})
    for bname in ('insurance', 'conviction'):
        deployed = bucket.get(f'{bname}_deployed', 0)
        budget = bucket.get(f'{bname}_budget', 1)
        if deployed > budget:
            alerts.append({
                'severity': 'MEDIUM',
                'rule': 'BUCKET_OVERSPEND',
                'message': f"{bname.title()} ${deployed:,.0f} exceeds ${budget:,.0f} budget",
                'timestamp': now.isoformat(),
            })

    # CONVEXITY_DROP
    cx = metrics.get('convexity', {})
    cr = cx.get('convexity_ratio', 0)
    if cr > 0 and cr < config['min_convexity_ratio']:
        alerts.append({
            'severity': 'HIGH',
            'rule': 'CONVEXITY_DROP',
            'message': f"Convexity {cr:.2f}x < {config['min_convexity_ratio']}x minimum",
            'timestamp': now.isoformat(),
        })

    # CORRELATION_HIGH
    decorr = metrics.get('decorrelation', {})
    dpct = decorr.get('decorrelated_pct', 100)
    if dpct < 50:
        alerts.append({
            'severity': 'MEDIUM',
            'rule': 'CORRELATION_HIGH',
            'message': f"Only {dpct:.0f}% decorrelated trades (target >50%)",
            'timestamp': now.isoformat(),
        })

    # EXPIRY_CRITICAL
    for p in metrics.get('open_monitor', []):
        if p['urgency'] == 'CRITICAL':
            alerts.append({
                'severity': 'HIGH',
                'rule': 'EXPIRY_CRITICAL',
                'message': f"{p['desc'][:25]} expires in {p['dte']}d",
                'timestamp': now.isoformat(),
            })

    # Store alerts
    date_key = now.strftime('%Y-%m-%d')
    for i, alert in enumerate(alerts):
        bk = f"/Alerts/{date_key}/{now.strftime('%H%M%S')}_{i}"
        db[bk] = alert

    return alerts


# ── One-Shot Report ──────────────────────────────────────────────────────

def display_report(console, metrics, alerts, snapshots):
    # ── Bucket Allocation ────────────────────────────────────────────
    bucket = metrics.get('bucket', {})
    tbl = RichTable(title="BUCKET ALLOCATION", expand=True,
                    title_style="bold white", show_header=False)
    tbl.add_column("Label", width=14)
    tbl.add_column("Amount", justify="right", width=18)
    tbl.add_column("Bar", width=20)
    tbl.add_column("Pct", justify="right", width=6)

    ins_d = bucket.get('insurance_deployed', 0)
    ins_b = bucket.get('insurance_budget', 1)
    ins_p = bucket.get('insurance_pct', 0)
    tbl.add_row("Insurance",
                f"${ins_d:,.0f} / ${ins_b:,.0f}",
                progress_bar(ins_d, ins_b),
                f"{ins_p:.0f}%")

    con_d = bucket.get('conviction_deployed', 0)
    con_b = bucket.get('conviction_budget', 1)
    con_p = bucket.get('conviction_pct', 0)
    tbl.add_row("Conviction",
                f"${con_d:,.0f} / ${con_b:,.0f}",
                progress_bar(con_d, con_b),
                f"{con_p:.0f}%")

    tbl.add_row("Cash Reserve", "$25K (PDT)", "", "")
    tbl.add_row("Deployable", f"~${(ins_b + con_b - ins_d - con_d):,.0f}", "", "")
    console.print(Panel(tbl, border_style="cyan"))

    # ── Convexity ────────────────────────────────────────────────────
    cx = metrics.get('convexity', {})
    if cx:
        cr = cx.get('convexity_ratio', 0)
        tr = cx.get('tail_ratio', 0)
        ev = cx.get('expected_value', 0)
        total_gl = cx.get('total_gl', 0)
        n_trades = cx.get('n_trades', 0)

        cr_grade = "POOR" if cr < 1.5 else "OK" if cr < 3 else "GOOD"
        cr_style = "red" if cr < 1.5 else "yellow" if cr < 3 else "green"
        tr_grade = "POOR" if tr < 2 else "OK" if tr < 4 else "GOOD"
        tr_style = "red" if tr < 2 else "yellow" if tr < 4 else "green"
        ev_style = "green" if ev > 0 else "red"

        cr_spark = sparkline(extract_sparkline_series(snapshots, ['convexity', 'convexity_ratio']))
        tr_spark = sparkline(extract_sparkline_series(snapshots, ['convexity', 'tail_ratio']))
        ev_spark = sparkline(extract_sparkline_series(snapshots, ['convexity', 'expected_value']))

        tbl = RichTable(title="CONVEXITY", expand=True,
                        title_style="bold white", show_header=False)
        tbl.add_column("Metric", width=12)
        tbl.add_column("Value", justify="right", width=10)
        tbl.add_column("Grade", width=8)
        tbl.add_column("Trend", width=10)
        tbl.add_column("Metric2", width=12)
        tbl.add_column("Value2", justify="right", width=14)

        tbl.add_row("Ratio", f"{cr:.2f}x",
                     Text(cr_grade, style=cr_style), cr_spark,
                     "Tail", f"{tr:.2f}x  {Text(tr_grade, style=tr_style)}  {tr_spark}")
        tbl.add_row("E[V]", f"${ev:+,.0f}",
                     Text("POS" if ev > 0 else "NEG", style=ev_style), ev_spark,
                     "Net P&L", f"${total_gl:+,.0f} ({n_trades} trades)")
        console.print(Panel(tbl, border_style="cyan"))

    # ── Bleed ────────────────────────────────────────────────────────
    bleed = metrics.get('bleed', [])
    if bleed:
        avg_net = sum(m['net'] for m in bleed) / len(bleed) if bleed else 0
        current_month = datetime.now().strftime('%Y-%m')
        this_month_net = 0
        for m in bleed:
            if m['month'] == current_month:
                this_month_net = m['net']

        bleed_vals = [m['net'] for m in bleed]
        bleed_spark = sparkline(bleed_vals)
        bleed_hist_spark = sparkline(extract_sparkline_series(
            snapshots, ['bleed_avg']))

        tbl = RichTable(title="BLEED", expand=True,
                        title_style="bold white", show_header=False)
        tbl.add_column("", width=25)
        tbl.add_column("", width=12)
        tbl.add_column("", width=25)

        tbl.add_row(
            f"Avg: ${avg_net:+,.0f}/mo  {bleed_spark}",
            f"This mo: ${this_month_net:+,.0f}",
            f"Limit: ${metrics.get('bucket', {}).get('insurance_budget', 700):,}",
        )
        console.print(Panel(tbl, border_style="cyan"))

    # ── Decorrelation ────────────────────────────────────────────────
    decorr = metrics.get('decorrelation', {})
    dpct = decorr.get('decorrelated_pct', 0)
    tbl = RichTable(title="DECORRELATION", expand=True,
                    title_style="bold white", show_header=False)
    tbl.add_column("", width=60)

    pct_style = "green" if dpct >= 50 else "yellow" if dpct >= 30 else "red"
    tbl.add_row(Text(
        f"{dpct:.0f}% decorrelated  {progress_bar(dpct, 100)}  (target >50%)",
        style=pct_style,
    ))
    console.print(Panel(tbl, border_style="cyan"))

    # ── Concentration ────────────────────────────────────────────────
    conc = metrics.get('concentration', [])
    flagged = [c for c in conc if c.get('flag')]
    if flagged:
        tbl = RichTable(title="CONCENTRATION (flagged names)", expand=True,
                        title_style="bold white", show_header=True,
                        header_style="bold cyan")
        tbl.add_column("Name", width=8)
        tbl.add_column("Trades", justify="right", width=7)
        tbl.add_column("ROI", justify="right", width=8)
        tbl.add_column("Flag", width=14)

        for c in flagged[:10]:
            flag_style = "red bold" if c['flag'] == 'BLEEDING' else "yellow"
            roi_style = "green" if c['roi'] >= 0 else "red"
            tbl.add_row(
                c['name'], str(c['n']),
                Text(f"{c['roi']:+.0f}%", style=roi_style),
                Text(c['flag'], style=flag_style),
            )
        console.print(Panel(tbl, border_style="cyan"))

    # ── Alerts ───────────────────────────────────────────────────────
    if alerts:
        tbl = RichTable(title="ALERTS", expand=True,
                        title_style="bold white", show_header=False)
        tbl.add_column("Sev", width=6)
        tbl.add_column("Message", width=60)

        for a in alerts:
            sev = a['severity']
            sev_style = "red bold" if sev == 'HIGH' else "yellow"
            tbl.add_row(
                Text(f"! {sev}", style=sev_style),
                a['message'],
            )
        console.print(Panel(tbl, border_style="red" if any(
            a['severity'] == 'HIGH' for a in alerts) else "yellow"))
    else:
        console.print(Panel("[green]No active alerts[/green]", border_style="green"))


# ── Live Dashboard ───────────────────────────────────────────────────────

def build_metrics_panel(metrics, snapshots, config):
    tbl = RichTable(title="Metrics & Allocation", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Metric", style="bold", width=18)
    tbl.add_column("Value", justify="right", width=12)
    tbl.add_column("Trend", width=10)

    cx = metrics.get('convexity', {})
    bucket = metrics.get('bucket', {})
    decorr = metrics.get('decorrelation', {})

    cr = cx.get('convexity_ratio', 0)
    cr_spark = sparkline(extract_sparkline_series(snapshots, ['convexity', 'convexity_ratio']))
    cr_style = "red" if cr < 1.5 else "yellow" if cr < 3 else "green"
    tbl.add_row("Convexity", Text(f"{cr:.2f}x", style=cr_style), cr_spark)

    tr = cx.get('tail_ratio', 0)
    tr_spark = sparkline(extract_sparkline_series(snapshots, ['convexity', 'tail_ratio']))
    tbl.add_row("Tail Ratio", f"{tr:.2f}x", tr_spark)

    ev = cx.get('expected_value', 0)
    ev_style = "green" if ev > 0 else "red"
    tbl.add_row("E[V]", Text(f"${ev:+,.0f}", style=ev_style), "")

    tbl.add_row("", "", "")
    ins_d = bucket.get('insurance_deployed', 0)
    ins_b = bucket.get('insurance_budget', 1)
    tbl.add_row("Insurance", f"${ins_d:,.0f}/${ins_b:,.0f}",
                progress_bar(ins_d, ins_b, width=10))

    con_d = bucket.get('conviction_deployed', 0)
    con_b = bucket.get('conviction_budget', 1)
    tbl.add_row("Conviction", f"${con_d:,.0f}/${con_b:,.0f}",
                progress_bar(con_d, con_b, width=10))

    dpct = decorr.get('decorrelated_pct', 0)
    pct_style = "green" if dpct >= 50 else "red"
    tbl.add_row("Decorrelation", Text(f"{dpct:.0f}%", style=pct_style), "")

    return Panel(tbl, border_style="blue")


def build_positions_panel(metrics):
    positions = metrics.get('open_monitor', [])
    tbl = RichTable(title="Open Positions", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Position", width=28, no_wrap=True)
    tbl.add_column("DTE", justify="right", width=5)
    tbl.add_column("P&L", justify="right", width=8)
    tbl.add_column("Alert", width=10)

    for p in positions[:15]:
        gl_style = "green" if p['gl'] >= 0 else "red"
        urgency_style = {
            'CRITICAL': 'red bold', 'SOON': 'yellow bold',
            'WATCH': 'yellow', 'OK': 'dim',
        }.get(p['urgency'], 'dim')
        tbl.add_row(
            p['desc'][:28], str(p['dte']),
            Text(f"${p['gl']:+,.0f}", style=gl_style),
            Text(p['urgency'], style=urgency_style),
        )

    if not positions:
        tbl.add_row("(none)", "", "", "")

    return Panel(tbl, border_style="blue")


def build_alerts_panel(alerts, db):
    tbl = RichTable(title="Alerts", expand=True,
                    title_style="bold white", show_header=False)
    tbl.add_column("Sev", width=6)
    tbl.add_column("Message", width=45)

    if alerts:
        for a in alerts[:8]:
            sev = a['severity']
            sev_style = "red bold" if sev == 'HIGH' else "yellow"
            tbl.add_row(Text(f"! {sev}", style=sev_style), a['message'])
    else:
        tbl.add_row("", "[green]No active alerts[/green]")

    # Recent alert history
    recent_keys = db.keys(prefix="/Alerts/")
    if len(recent_keys) > len(alerts):
        tbl.add_row("", "")
        tbl.add_row(Text("History", style="dim"), Text(f"{len(recent_keys)} total alerts stored", style="dim"))

    return Panel(tbl, border_style="red" if alerts and any(
        a['severity'] == 'HIGH' for a in alerts) else "yellow" if alerts else "green")


def build_status_panel(runner, job_names, start_time, db):
    tbl = RichTable(title="Status", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Job", style="bold", width=14)
    tbl.add_column("Status", justify="center", width=14)

    status_styles = {
        JobStatus.SUCCEEDED: ("●", "green"),
        JobStatus.RUNNING: ("●", "yellow"),
        JobStatus.FAILED: ("●", "red"),
        JobStatus.PENDING: ("○", "dim"),
        JobStatus.STOPPED: ("●", "dim"),
    }

    for name in job_names:
        status = runner.get_status(name)
        if status is None:
            status = JobStatus.PENDING
        dot, style = status_styles.get(status, ("?", "dim"))
        tbl.add_row(name, Text(f"{dot} {status.value.upper()}", style=style))

    uptime = int(time.time() - start_time)
    tbl.add_row("", "")
    tbl.add_row(Text("Uptime", style="dim"), Text(f"{uptime}s", style="dim"))

    # DB stats
    n_keys = len(db.keys(prefix="/"))
    tbl.add_row(Text("DB keys", style="dim"), Text(str(n_keys), style="dim"))

    return Panel(tbl, border_style="blue")


def run_live_dashboard(db, config, console):
    """Run a live-updating 4-panel dashboard using Rich Live + Walpole."""
    db_lock = threading.Lock()
    metrics_state = {"current": compute_all_metrics(db, config)}
    alerts_state = {"current": []}
    snapshots = load_snapshot_history(db)

    def job_recompute():
        with db_lock:
            metrics_state["current"] = compute_all_metrics(db, config)
        return metrics_state["current"]

    def job_snapshot():
        with db_lock:
            snap = take_snapshot(db, config)
        snapshots.append(snap)
        if len(snapshots) > 30:
            snapshots.pop(0)
        return snap

    def job_rules():
        with db_lock:
            alerts_state["current"] = check_rules(db, metrics_state["current"], config)
        return alerts_state["current"]

    runner = JobRunner(barbara=db)
    runner.add_job(JobConfig(
        name="recompute", callable=job_recompute,
        mode=JobMode.PERIODIC, interval=45.0,
    ))
    runner.add_job(JobConfig(
        name="snapshot", callable=job_snapshot,
        mode=JobMode.PERIODIC, interval=300.0,
    ))
    runner.add_job(JobConfig(
        name="rules", callable=job_rules,
        mode=JobMode.PERIODIC, interval=300.0,
        depends_on=["recompute"],
    ))

    job_names = ["recompute", "snapshot", "rules"]

    # Run initial rules check before starting background jobs
    alerts_state["current"] = check_rules(db, metrics_state["current"], config)

    console.print("[bold cyan]Starting live dashboard (Ctrl+C to stop)...[/bold cyan]\n")
    runner.start()
    start_time = time.time()

    def build_dashboard():
        layout = Layout()
        layout.split_column(
            Layout(name="top", ratio=1),
            Layout(name="bottom", ratio=1),
        )
        layout["top"].split_row(
            Layout(build_metrics_panel(metrics_state["current"], snapshots, config), name="metrics"),
            Layout(build_positions_panel(metrics_state["current"]), name="positions"),
        )
        layout["bottom"].split_row(
            Layout(build_alerts_panel(alerts_state["current"], db), name="alerts"),
            Layout(build_status_panel(runner, job_names, start_time, db), name="status"),
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


# ── Config Display/Edit ──────────────────────────────────────────────────

def show_config(db, console):
    config = get_config(db)
    tbl = RichTable(title="Trade Tracker Configuration", expand=True,
                    title_style="bold white", show_header=True,
                    header_style="bold cyan")
    tbl.add_column("Key", style="bold", width=24)
    tbl.add_column("Value", justify="right", width=14)
    tbl.add_column("Default", justify="right", width=14, style="dim")

    for key, default in DEFAULT_CONFIG.items():
        val = config[key]
        style = "" if val == default else "yellow"
        tbl.add_row(key, Text(str(val), style=style), str(default))

    console.print(Panel(tbl, border_style="cyan"))
    console.print("[dim]Edit with: python trade_tracker.py config --set KEY VALUE[/dim]")


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Trade Tracker — Talebian options strategy tracker",
    )
    parser.add_argument('--live', action='store_true',
                        help='Run live dashboard instead of one-shot report')

    sub = parser.add_subparsers(dest='command')

    # import
    imp = sub.add_parser('import', help='Import Fidelity CSV exports')
    imp.add_argument('--dir', type=str, default=None,
                     help='Directory containing CSVs')

    # log
    log = sub.add_parser('log', help='Manually log a trade')
    log.add_argument('sym', help='Option symbol (e.g. -SPY260320P350)')
    log.add_argument('qty', type=int, help='Number of contracts')
    log.add_argument('price', type=float, help='Price per contract')
    log.add_argument('--action', default='BUY_OPEN',
                     choices=['BUY_OPEN', 'SELL_CLOSE', 'BUY_CLOSE', 'SELL_OPEN'],
                     help='Trade action')
    log.add_argument('--bucket', default=None,
                     choices=['insurance', 'conviction'],
                     help='Override auto-classification')
    log.add_argument('--note', default='', help='Trade note')

    # snapshot
    sub.add_parser('snapshot', help='Force a metric snapshot')

    # config
    cfg = sub.add_parser('config', help='Show/edit configuration')
    cfg.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                     help='Set a config value')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    console = Console(width=100)

    db = open_db()
    ensure_config(db)

    try:
        if args.command == 'import':
            console.print(Panel("[bold]Importing Fidelity CSV exports...[/bold]",
                                style="cyan", expand=False))
            stats = import_trades(db, data_dir=args.dir, console=console)
            console.print(f"\n[green]Import complete.[/green]")

        elif args.command == 'log':
            trade = log_trade(db, args.sym, args.qty, args.price,
                              action=args.action, bucket=args.bucket,
                              note=args.note)
            console.print(f"[green]Logged:[/green] {args.sym} x{args.qty} "
                          f"@ ${args.price:.2f} [{trade['bucket']}]")
            if args.note:
                console.print(f"  Note: {args.note}")

        elif args.command == 'snapshot':
            console.print("Taking snapshot...")
            metrics = take_snapshot(db, get_config(db))
            console.print(f"[green]Snapshot saved.[/green] "
                          f"({metrics['n_realized']} realized, "
                          f"{metrics['n_open']} open)")

        elif args.command == 'config':
            if args.set:
                key, value = args.set
                if key not in DEFAULT_CONFIG:
                    console.print(f"[red]Unknown key: {key}[/red]")
                    console.print(f"Valid keys: {', '.join(DEFAULT_CONFIG.keys())}")
                else:
                    # Coerce to same type as default
                    default = DEFAULT_CONFIG[key]
                    typed_val = type(default)(value)
                    db[f"/Config/{key}"] = typed_val
                    console.print(f"[green]Set {key} = {typed_val}[/green]")
            else:
                show_config(db, console)

        elif args.live:
            run_live_dashboard(db, get_config(db), console)

        else:
            # Default: one-shot report
            config = get_config(db)
            metrics = compute_all_metrics(db, config)
            alerts = check_rules(db, metrics, config)
            snapshots = load_snapshot_history(db)

            console.print(Panel(
                "[bold]TRADE TRACKER[/bold]\n[dim]Talebian Edition[/dim]",
                style="cyan", expand=False,
            ))
            console.print(f"  {metrics['n_realized']} realized | "
                          f"{metrics['n_txns']} transactions | "
                          f"{metrics['n_open']} open | "
                          f"{len(snapshots)} snapshots\n")

            if metrics['n_realized'] == 0 and metrics['n_open'] == 0:
                console.print("[yellow]No trade data found. "
                              "Run 'python trade_tracker.py import' first.[/yellow]")
            else:
                display_report(console, metrics, alerts, snapshots)
    finally:
        db.close()


if __name__ == "__main__":
    main()

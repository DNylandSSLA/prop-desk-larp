#!/usr/bin/env python3
"""
Trade Analysis Toolkit — Talebian Edition.

Reads Fidelity CSV exports and generates a trade analysis report focused on
convexity, optionality, and identifying where execution leaked value.

Sections:
1. Convexity Profile — payoff asymmetry, tail ratio, distribution shape
2. Bleed Analysis — monthly premium cost of carrying the options book
3. Concentration Report — over-trading detection, diminishing returns
4. Optimal Exit Analysis — where the underlying peaked during your hold
5. Open Position Monitor — current positions with expiry/spread warnings
6. Talebian Scorecard — overall assessment
"""

import csv
import io
import re
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import yfinance as yf

from rich.console import Console
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.text import Text

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path(__file__).parent / "personal_doNOTputongithub"
POSITIONS_CSV = DATA_DIR / "Portfolio_Positions_Feb-07-2026.csv"
REALIZED_CSV = DATA_DIR / "Portfolio_Positions_Feb-07-2026 (1).csv"
HISTORY_CSV = DATA_DIR / "Accounts_History.csv"


def parse_dollar(s):
    if not s:
        return None
    s = s.strip().replace('$', '').replace(',', '').replace('+', '')
    if not s or s == '--':
        return None
    return float(s)


def parse_date(s):
    """Parse MM/DD/YYYY to datetime."""
    try:
        return datetime.strptime(s.strip(), '%m/%d/%Y')
    except (ValueError, AttributeError):
        return None


def parse_option_symbol(sym):
    """Parse ' -UVXY250417C55' into components."""
    sym = sym.strip().lstrip(' -')
    m = re.match(r'^([A-Za-z]+\d?)(\d{6})([CP])(.+)$', sym)
    if not m:
        return None
    ticker, date_str, cp, strike_str = m.groups()
    try:
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        strike = float(strike_str)
    except ValueError:
        return None
    # Clean underlying: UVXY1 -> UVXY
    underlying = re.sub(r'\d+$', '', ticker)
    return {
        'underlying': underlying,
        'expiry': datetime(year, month, day),
        'type': 'call' if cp == 'C' else 'put',
        'strike': strike,
    }


# ── CSV Parsing ───────────────────────────────────────────────────────────

def parse_realized(csv_path=None):
    trades = []
    with open(csv_path or REALIZED_CSV, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sym = (row.get('Symbol') or '').strip()
            if not sym:
                continue
            gl = parse_dollar(row.get('Total Term Gain/Loss', ''))
            if gl is None:
                continue
            cost = parse_dollar(row.get('Cost Basis', '')) or 0
            proceeds = parse_dollar(row.get('Proceeds', '')) or 0
            desc = (row.get('Description') or '').strip()
            acct = (row.get('Account Name') or '').strip()
            opt = parse_option_symbol(sym)
            trades.append({
                'sym': sym, 'desc': desc, 'acct': acct,
                'cost': cost, 'proceeds': proceeds, 'gl': gl,
                'opt': opt,
                'is_option': opt is not None,
            })
    return trades


def parse_transactions(csv_path=None):
    """Parse Accounts_History.csv for trade dates."""
    txns = []
    with open(csv_path or HISTORY_CSV, newline='', encoding='utf-8-sig') as f:
        # Skip blank lines before header
        for line in f:
            if line.strip():
                # Found header — build reader from here
                rest = line + f.read()
                break
        else:
            return txns
        for row in csv.DictReader(io.StringIO(rest)):
            action = (row.get('Action') or '').strip()
            if not action:
                continue
            sym = (row.get('Symbol') or '').strip()
            if not sym:
                continue
            run_date = parse_date((row.get('Run Date') or '').strip())
            if not run_date:
                continue
            price = parse_dollar(row.get('Price', ''))
            qty = parse_dollar(row.get('Quantity', ''))
            amount = parse_dollar(row.get('Amount', ''))
            acct = (row.get('Account') or '').strip()

            is_open = 'OPENING' in action.upper() or (
                'YOU BOUGHT' in action.upper() and 'CLOSING' not in action.upper()
            )
            is_close = 'CLOSING' in action.upper() or (
                'YOU SOLD' in action.upper() and 'OPENING' not in action.upper()
            )

            txns.append({
                'date': run_date, 'sym': sym, 'acct': acct,
                'action': action, 'price': price or 0,
                'qty': qty or 0, 'amount': amount or 0,
                'is_open': is_open, 'is_close': is_close,
            })
    return txns


def parse_current_options(csv_path=None):
    """Parse current option positions from Portfolio CSV."""
    options = []
    with open(csv_path or POSITIONS_CSV, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sym = (row.get('Symbol') or '').strip()
            if not sym or not sym.startswith('-'):
                continue
            acct = (row.get('Account Name') or '').strip()
            if acct != 'ROTH IRA':
                continue
            try:
                qty = int(float(row['Quantity']))
                mv = parse_dollar(row.get('Current Value', ''))
                cost = parse_dollar(row.get('Cost Basis Total', '')) or 0
                gl = parse_dollar(row.get('Total Gain/Loss Dollar', '')) or 0
            except (ValueError, KeyError):
                continue
            if mv is None:
                continue
            opt = parse_option_symbol(sym)
            options.append({
                'sym': sym, 'desc': (row.get('Description') or '').strip(),
                'qty': qty, 'mv': mv, 'cost': cost, 'gl': gl, 'opt': opt,
            })
    return options


# ── Trade Date Matching ───────────────────────────────────────────────────

def get_trade_dates(txns, realized_trades):
    """For each realized trade, find the earliest open and latest close date."""
    # Group transactions by symbol
    by_sym = defaultdict(list)
    for t in txns:
        by_sym[t['sym']].append(t)

    results = {}
    for trade in realized_trades:
        sym = trade['sym']
        sym_txns = by_sym.get(sym, [])
        if not sym_txns:
            continue
        opens = [t for t in sym_txns if t['is_open']]
        closes = [t for t in sym_txns if t['is_close']]
        if opens and closes:
            first_open = min(t['date'] for t in opens)
            last_close = max(t['date'] for t in closes)
            results[sym] = {'open': first_open, 'close': last_close}
        elif opens:
            first_open = min(t['date'] for t in opens)
            results[sym] = {'open': first_open, 'close': None}
    return results


# ── Optimal Exit Analysis ─────────────────────────────────────────────────

def fetch_underlying_history(tickers_and_ranges):
    """Batch-fetch historical underlying prices.

    tickers_and_ranges: dict of underlying -> (start_date, end_date)
    Returns: dict of underlying -> list of (date, close_price)
    """
    results = {}
    for underlying, (start, end) in tickers_and_ranges.items():
        # Add buffer days
        fetch_start = (start - timedelta(days=5)).strftime('%Y-%m-%d')
        fetch_end = (end + timedelta(days=5)).strftime('%Y-%m-%d')
        try:
            data = yf.download(
                underlying, start=fetch_start, end=fetch_end,
                progress=False, threads=False,
            )
            if data.empty:
                continue
            prices = []
            for ts, row in data.iterrows():
                close = row['Close']
                if hasattr(close, 'iloc'):
                    close = close.iloc[0]
                prices.append((ts.to_pydatetime(), float(close)))
            results[underlying] = prices
        except Exception:
            continue
    return results


def compute_optimal_exits(realized, trade_dates):
    """For top trades, compute where the underlying peaked during the hold."""
    # Focus on Roth IRA option trades with known dates
    candidates = []
    for t in realized:
        if t['acct'] != 'ROTH IRA' or not t['is_option']:
            continue
        if t['sym'] not in trade_dates:
            continue
        dates = trade_dates[t['sym']]
        if not dates.get('close'):
            continue
        candidates.append({**t, 'dates': dates})

    # Top 25 by absolute P&L
    candidates.sort(key=lambda x: abs(x['gl']), reverse=True)
    top = candidates[:25]

    if not top:
        return []

    # Collect unique underlyings with date ranges
    tickers = {}
    for t in top:
        u = t['opt']['underlying']
        start = t['dates']['open']
        end = t['dates']['close']
        if u not in tickers:
            tickers[u] = (start, end)
        else:
            prev_start, prev_end = tickers[u]
            tickers[u] = (min(start, prev_start), max(end, prev_end))

    # Fetch prices
    prices = fetch_underlying_history(tickers)

    # Analyze each trade
    results = []
    for t in top:
        u = t['opt']['underlying']
        if u not in prices:
            continue
        opt = t['opt']
        open_date = t['dates']['open']
        close_date = t['dates']['close']

        # Filter prices to holding period
        holding_prices = [
            (d, p) for d, p in prices[u]
            if open_date <= d <= close_date + timedelta(days=1)
        ]
        if len(holding_prices) < 2:
            continue

        entry_price = holding_prices[0][1]
        exit_price = holding_prices[-1][1]

        if opt['type'] == 'call':
            # For calls: best when underlying is highest
            best_date, best_price = max(holding_prices, key=lambda x: x[1])
            mfe_pct = (best_price - entry_price) / entry_price * 100
            actual_pct = (exit_price - entry_price) / entry_price * 100
        else:
            # For puts: best when underlying is lowest
            best_date, best_price = min(holding_prices, key=lambda x: x[1])
            mfe_pct = (entry_price - best_price) / entry_price * 100
            actual_pct = (entry_price - exit_price) / entry_price * 100

        # Capture ratio: how much of the favorable move you captured
        if abs(mfe_pct) > 0.1:
            capture = (actual_pct / mfe_pct * 100) if mfe_pct > 0 else 0
        else:
            capture = 0

        hold_days = (close_date - open_date).days

        results.append({
            'desc': t['desc'][:35],
            'underlying': u,
            'type': opt['type'],
            'gl': t['gl'],
            'cost': t['cost'],
            'hold_days': hold_days,
            'entry_price': entry_price,
            'best_price': best_price,
            'best_date': best_date,
            'exit_price': exit_price,
            'mfe_pct': mfe_pct,
            'actual_pct': actual_pct,
            'capture': min(capture, 999),
        })

    return results


# ── Analytics ─────────────────────────────────────────────────────────────

def compute_convexity(trades):
    """Compute Talebian convexity metrics."""
    roth_options = [t for t in trades if t['acct'] == 'ROTH IRA' and t['is_option']]
    if not roth_options:
        return {}

    gains = [t['gl'] for t in roth_options if t['gl'] > 0]
    losses = [t['gl'] for t in roth_options if t['gl'] < 0]

    avg_win = sum(gains) / len(gains) if gains else 0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1

    # Tail ratio: avg of top 10% wins / avg of bottom 10% losses
    n_tail = max(1, len(roth_options) // 10)
    sorted_gl = sorted([t['gl'] for t in roth_options], reverse=True)
    top_tail = sum(sorted_gl[:n_tail]) / n_tail
    bottom_tail = abs(sum(sorted_gl[-n_tail:])) / n_tail

    # ROI distribution
    rois = []
    for t in roth_options:
        if t['cost'] > 0:
            rois.append(t['gl'] / t['cost'] * 100)

    return {
        'n_trades': len(roth_options),
        'n_wins': len(gains),
        'n_losses': len(losses),
        'win_rate': len(gains) / (len(gains) + len(losses)) * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'convexity_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
        'tail_ratio': top_tail / bottom_tail if bottom_tail > 0 else 0,
        'top_tail_avg': top_tail,
        'bottom_tail_avg': bottom_tail,
        'max_win': max(gains) if gains else 0,
        'max_loss': min(losses) if losses else 0,
        'expected_value': (len(gains) / len(roth_options) * avg_win -
                          len(losses) / len(roth_options) * avg_loss),
        'roi_median': sorted(rois)[len(rois) // 2] if rois else 0,
        'total_gl': sum(t['gl'] for t in roth_options),
    }


def compute_concentration(trades):
    """Detect over-trading on individual names."""
    by_name = defaultdict(lambda: {'n': 0, 'cost': 0, 'gl': 0, 'wins': 0, 'losses': 0})
    for t in trades:
        if t['acct'] != 'ROTH IRA':
            continue
        if t['opt']:
            name = t['opt']['underlying']
        else:
            # Stock — extract from symbol
            name = t['sym'].strip()
        d = by_name[name]
        d['n'] += 1
        d['cost'] += t['cost']
        d['gl'] += t['gl']
        if t['gl'] > 0:
            d['wins'] += 1
        elif t['gl'] < 0:
            d['losses'] += 1

    results = []
    for name, d in by_name.items():
        roi = (d['gl'] / d['cost'] * 100) if d['cost'] > 0 else 0
        marginal_flag = ''
        # Flag diminishing returns: lots of trades, low or negative ROI
        if d['n'] >= 5 and roi < 5:
            marginal_flag = 'DIMINISHING'
        if d['n'] >= 3 and roi < -20:
            marginal_flag = 'BLEEDING'
        results.append({
            'name': name, **d, 'roi': roi, 'flag': marginal_flag,
        })

    results.sort(key=lambda x: x['n'], reverse=True)
    return results


def compute_bleed(trades, txns):
    """Compute monthly premium bleed from option purchases."""
    # From transactions: sum of option opening purchases by month
    monthly = defaultdict(lambda: {'spent': 0, 'received': 0, 'n_open': 0, 'n_close': 0})

    for t in txns:
        if t['acct'] != 'ROTH IRA':
            continue
        opt = parse_option_symbol(t['sym'])
        if not opt:
            continue
        month_key = t['date'].strftime('%Y-%m')
        if t['is_open'] and t['amount'] < 0:
            monthly[month_key]['spent'] += abs(t['amount'])
            monthly[month_key]['n_open'] += abs(t['qty'])
        elif t['is_close'] and t['amount'] > 0:
            monthly[month_key]['received'] += t['amount']
            monthly[month_key]['n_close'] += abs(t['qty'])

    results = []
    for month, d in sorted(monthly.items()):
        net = d['received'] - d['spent']
        results.append({
            'month': month,
            'spent': d['spent'],
            'received': d['received'],
            'net': net,
            'n_open': d['n_open'],
            'n_close': d['n_close'],
        })
    return results


def compute_open_monitor(current_options):
    """Analyze open positions for expiry urgency and value decay."""
    results = []
    now = datetime.now()
    for o in current_options:
        opt = o['opt']
        if not opt:
            continue
        dte = (opt['expiry'] - now).days
        pnl_pct = (o['gl'] / o['cost'] * 100) if o['cost'] > 0 else 0
        # Urgency
        if dte <= 14:
            urgency = 'CRITICAL'
        elif dte <= 30:
            urgency = 'SOON'
        elif dte <= 90:
            urgency = 'WATCH'
        else:
            urgency = 'OK'
        results.append({
            'desc': o['desc'][:40],
            'qty': o['qty'],
            'mv': o['mv'],
            'cost': o['cost'],
            'gl': o['gl'],
            'pnl_pct': pnl_pct,
            'dte': dte,
            'urgency': urgency,
            'type': opt['type'],
            'underlying': opt['underlying'],
        })
    results.sort(key=lambda x: x['dte'])
    return results


# ── Display ───────────────────────────────────────────────────────────────

def display_convexity(console, cx):
    tbl = RichTable(title="1. CONVEXITY PROFILE", expand=True,
                    title_style="bold white", show_header=False)
    tbl.add_column("Metric", style="bold", width=24)
    tbl.add_column("Value", justify="right", width=14)
    tbl.add_column("Assessment", width=20)

    # Convexity ratio
    cr = cx['convexity_ratio']
    cr_grade = ("POOR" if cr < 1.5 else "OK" if cr < 3 else "GOOD")
    cr_style = "red" if cr < 1.5 else "yellow" if cr < 3 else "green"
    tbl.add_row("Convexity Ratio", f"{cr:.2f}x",
                Text(f"{cr_grade} (target >3x)", style=cr_style))

    # Tail ratio
    tr = cx['tail_ratio']
    tr_grade = "POOR" if tr < 2 else "OK" if tr < 4 else "GOOD"
    tr_style = "red" if tr < 2 else "yellow" if tr < 4 else "green"
    tbl.add_row("Tail Ratio (10%)", f"{tr:.2f}x",
                Text(f"{tr_grade} (target >4x)", style=tr_style))

    # Win rate (for Taleb: low is fine if convexity is high)
    wr = cx['win_rate']
    tbl.add_row("Win Rate", f"{wr:.1f}%",
                Text("(irrelevant if convex)", style="dim"))

    # Expected value per trade
    ev = cx['expected_value']
    ev_style = "green" if ev > 0 else "red"
    tbl.add_row("E[V] per trade", f"${ev:,.0f}",
                Text("POSITIVE" if ev > 0 else "NEGATIVE", style=ev_style))

    tbl.add_row("", "", "")
    tbl.add_row("Avg Win", f"+${cx['avg_win']:,.0f}", "")
    tbl.add_row("Avg Loss", f"-${cx['avg_loss']:,.0f}", "")
    tbl.add_row("Top Tail Avg", f"+${cx['top_tail_avg']:,.0f}",
                Text("(top 10% of trades)", style="dim"))
    tbl.add_row("Bottom Tail Avg", f"-${cx['bottom_tail_avg']:,.0f}",
                Text("(bottom 10% of trades)", style="dim"))
    tbl.add_row("Max Win", f"+${cx['max_win']:,.0f}", "")
    tbl.add_row("Max Loss", f"-${abs(cx['max_loss']):,.0f}", "")
    tbl.add_row("Median ROI", f"{cx['roi_median']:+.0f}%", "")
    tbl.add_row("", "", "")
    tbl.add_row(
        Text("Net Options P&L", style="bold"),
        Text(f"${cx['total_gl']:+,.0f}",
             style="green" if cx['total_gl'] >= 0 else "red"),
        Text(f"({cx['n_trades']} trades)", style="dim"),
    )

    console.print(Panel(tbl, border_style="cyan"))


def display_bleed(console, bleed):
    tbl = RichTable(title="2. BLEED ANALYSIS (Monthly Premium Flow)",
                    expand=True, title_style="bold white",
                    show_header=True, header_style="bold cyan")
    tbl.add_column("Month", width=8)
    tbl.add_column("Deployed", justify="right", width=10)
    tbl.add_column("Recovered", justify="right", width=10)
    tbl.add_column("Net", justify="right", width=10)
    tbl.add_column("Contracts", justify="right", width=10)

    total_spent = 0
    total_recv = 0
    for m in bleed:
        net_style = "green" if m['net'] >= 0 else "red"
        tbl.add_row(
            m['month'],
            f"${m['spent']:,.0f}",
            f"${m['received']:,.0f}",
            Text(f"${m['net']:+,.0f}", style=net_style),
            f"{int(m['n_open'])}/{int(m['n_close'])}",
        )
        total_spent += m['spent']
        total_recv += m['received']

    total_net = total_recv - total_spent
    n_months = len(bleed) or 1
    tbl.add_row("", "", "", "", "")
    tbl.add_row(
        Text("TOTAL", style="bold"),
        Text(f"${total_spent:,.0f}", style="bold"),
        Text(f"${total_recv:,.0f}", style="bold"),
        Text(f"${total_net:+,.0f}",
             style="bold green" if total_net >= 0 else "bold red"),
        "",
    )
    tbl.add_row(
        Text("Avg/month", style="dim"), "",  "",
        Text(f"${total_net/n_months:+,.0f}/mo", style="dim"), "",
    )

    console.print(Panel(tbl, border_style="cyan"))


def display_concentration(console, conc):
    tbl = RichTable(title="3. CONCENTRATION REPORT (Roth Options by Underlying)",
                    expand=True, title_style="bold white",
                    show_header=True, header_style="bold cyan")
    tbl.add_column("Underlying", style="bold", width=10)
    tbl.add_column("Trades", justify="right", width=6)
    tbl.add_column("Deployed", justify="right", width=10)
    tbl.add_column("Net P&L", justify="right", width=9)
    tbl.add_column("ROI", justify="right", width=7)
    tbl.add_column("W/L", justify="center", width=5)
    tbl.add_column("Flag", width=12)

    for c in conc[:20]:
        gl_style = "green" if c['gl'] >= 0 else "red"
        roi_style = "green" if c['roi'] >= 0 else "red"
        flag_style = "red bold" if c['flag'] == 'BLEEDING' else (
            "yellow" if c['flag'] == 'DIMINISHING' else "dim"
        )
        tbl.add_row(
            c['name'],
            str(c['n']),
            f"${c['cost']:,.0f}",
            Text(f"${c['gl']:+,.0f}", style=gl_style),
            Text(f"{c['roi']:+.0f}%", style=roi_style),
            f"{c['wins']}/{c['losses']}",
            Text(c['flag'] or '', style=flag_style),
        )

    console.print(Panel(tbl, border_style="cyan"))


def display_optimal_exits(console, exits):
    tbl = RichTable(
        title="4. OPTIMAL EXIT ANALYSIS (path-dependent, non-look-forwardable)",
        expand=True, title_style="bold white",
        show_header=True, header_style="bold cyan",
    )
    tbl.add_column("Trade", width=35, no_wrap=True)
    tbl.add_column("P&L", justify="right", width=9)
    tbl.add_column("Hold", justify="right", width=6)
    tbl.add_column("Best Move", justify="right", width=9)
    tbl.add_column("Your Move", justify="right", width=9)
    tbl.add_column("Captured", justify="right", width=8)
    tbl.add_column("Peak Date", width=10)

    for e in exits:
        gl_style = "green" if e['gl'] >= 0 else "red"

        # Capture ratio coloring
        cap = e['capture']
        if cap > 80:
            cap_style = "green"
        elif cap > 40:
            cap_style = "yellow"
        elif cap > 0:
            cap_style = "red"
        else:
            cap_style = "dim"

        # For losing trades where underlying never moved favorably
        if e['mfe_pct'] < 0.5:
            best_str = "none"
            cap_str = "n/a"
            cap_style = "dim"
        else:
            best_str = f"{e['mfe_pct']:+.1f}%"
            cap_str = f"{cap:.0f}%"

        tbl.add_row(
            e['desc'][:35],
            Text(f"${e['gl']:+,.0f}", style=gl_style),
            f"{e['hold_days']}d",
            best_str,
            f"{e['actual_pct']:+.1f}%",
            Text(cap_str, style=cap_style),
            e['best_date'].strftime('%m/%d'),
        )

    console.print(Panel(tbl, border_style="cyan"))
    console.print(
        "  [dim]Capture = how much of the underlying's best favorable move you realized.[/]\n"
        "  [dim]>80% = excellent timing. <40% = left significant value. "
        "Remember: you couldn't see the peak coming.[/]"
    )


def display_open_monitor(console, positions):
    tbl = RichTable(title="5. OPEN POSITION MONITOR",
                    expand=True, title_style="bold white",
                    show_header=True, header_style="bold cyan")
    tbl.add_column("Position", width=35, no_wrap=True)
    tbl.add_column("Qty", justify="right", width=4)
    tbl.add_column("DTE", justify="right", width=5)
    tbl.add_column("MV", justify="right", width=8)
    tbl.add_column("P&L", justify="right", width=9)
    tbl.add_column("P&L%", justify="right", width=7)
    tbl.add_column("Alert", width=10)

    for p in positions:
        gl_style = "green" if p['gl'] >= 0 else "red"
        urgency_style = {
            'CRITICAL': 'red bold',
            'SOON': 'yellow bold',
            'WATCH': 'yellow',
            'OK': 'dim',
        }.get(p['urgency'], 'dim')

        tbl.add_row(
            p['desc'][:32],
            str(p['qty']),
            str(p['dte']),
            f"${p['mv']:,.0f}",
            Text(f"${p['gl']:+,.0f}", style=gl_style),
            Text(f"{p['pnl_pct']:+.0f}%", style=gl_style),
            Text(p['urgency'], style=urgency_style),
        )

    # Summary
    total_mv = sum(p['mv'] for p in positions)
    total_cost = sum(p['cost'] for p in positions)
    total_gl = sum(p['gl'] for p in positions)
    critical = sum(1 for p in positions if p['urgency'] == 'CRITICAL')
    soon = sum(1 for p in positions if p['urgency'] == 'SOON')

    console.print(Panel(tbl, border_style="cyan"))
    alerts = []
    if critical:
        alerts.append(f"[red bold]{critical} positions expiring within 2 weeks[/]")
    if soon:
        alerts.append(f"[yellow]{soon} positions expiring within 30 days[/]")
    console.print(f"  MV: ${total_mv:,.0f} | Cost: ${total_cost:,.0f} | "
                  f"Unrealized: ${total_gl:+,.0f}")
    for a in alerts:
        console.print(f"  {a}")


def display_scorecard(console, cx, bleed, conc, exits):
    tbl = RichTable(title="6. TALEBIAN SCORECARD",
                    expand=True, title_style="bold white", show_header=False)
    tbl.add_column("", width=30)
    tbl.add_column("", width=10)
    tbl.add_column("", width=30)

    def grade(val, thresholds, labels=None):
        """Return (emoji-free grade, style) based on thresholds."""
        if labels is None:
            labels = ['F', 'D', 'C', 'B', 'A']
        styles = ['red bold', 'red', 'yellow', 'green', 'green bold']
        for i, thresh in enumerate(thresholds):
            if val < thresh:
                return labels[i], styles[i]
        return labels[-1], styles[-1]

    # 1. Convexity
    cr_grade, cr_style = grade(cx['convexity_ratio'], [1.0, 1.5, 2.5, 4.0])
    tbl.add_row("Convexity (avg win/loss)", Text(cr_grade, style=cr_style),
                f"ratio={cx['convexity_ratio']:.2f}x")

    # 2. Tail ratio
    tr_grade, tr_style = grade(cx['tail_ratio'], [1.5, 2.5, 4.0, 6.0])
    tbl.add_row("Tail Asymmetry", Text(tr_grade, style=tr_style),
                f"ratio={cx['tail_ratio']:.2f}x")

    # 3. Expected value
    ev = cx['expected_value']
    ev_grade, ev_style = grade(ev, [-50, 0, 20, 50])
    tbl.add_row("Expected Value/Trade", Text(ev_grade, style=ev_style),
                f"${ev:+,.0f}")

    # 4. Bleed sustainability
    total_bleed = sum(m['net'] for m in bleed)
    n_months = len(bleed) or 1
    monthly_bleed = total_bleed / n_months
    bl_grade, bl_style = grade(monthly_bleed, [-1000, -500, -200, 0])
    tbl.add_row("Bleed Sustainability", Text(bl_grade, style=bl_style),
                f"${monthly_bleed:+,.0f}/mo")

    # 5. Concentration discipline
    overtraded = sum(1 for c in conc if c['flag'] in ('BLEEDING', 'DIMINISHING'))
    cd_grade, cd_style = grade(-overtraded, [-6, -4, -2, -1],
                               ['F', 'D', 'C', 'B', 'A'])
    tbl.add_row("Concentration Discipline", Text(cd_grade, style=cd_style),
                f"{overtraded} names flagged")

    # 6. Exit capture (if we have data)
    if exits:
        avg_capture = sum(e['capture'] for e in exits if e['mfe_pct'] > 0.5) / max(
            1, sum(1 for e in exits if e['mfe_pct'] > 0.5))
        ec_grade, ec_style = grade(avg_capture, [20, 35, 50, 70])
        tbl.add_row("Exit Capture", Text(ec_grade, style=ec_style),
                    f"avg {avg_capture:.0f}% of MFE")
    else:
        tbl.add_row("Exit Capture", Text("N/A", style="dim"), "no data")

    console.print(Panel(tbl, border_style="cyan"))


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    console = Console(width=140)
    console.print(Panel(
        "[bold]TRADE ANALYSIS TOOLKIT[/bold]\n[dim]Talebian Edition[/dim]",
        style="cyan", expand=False,
    ))

    # Verify files
    for f in [POSITIONS_CSV, REALIZED_CSV, HISTORY_CSV]:
        if not f.exists():
            console.print(f"[red]Missing: {f}[/red]")
            return

    # 1. Parse
    console.print("Parsing Fidelity exports...")
    realized = parse_realized()
    txns = parse_transactions()
    current_opts = parse_current_options()
    console.print(f"  {len(realized)} closed trades, {len(txns)} transactions, "
                  f"{len(current_opts)} open options")

    # 2. Trade date matching
    console.print("Matching trade dates...")
    trade_dates = get_trade_dates(txns, realized)
    console.print(f"  Matched dates for {len(trade_dates)} positions")

    # 3. Compute analytics
    console.print("Computing analytics...")
    cx = compute_convexity(realized)
    conc = compute_concentration(realized)
    bleed = compute_bleed(realized, txns)

    # 4. Optimal exit analysis (fetches yfinance)
    console.print("Fetching historical prices for exit analysis...")
    exits = compute_optimal_exits(realized, trade_dates)
    console.print(f"  Analyzed {len(exits)} trades\n")

    # 5. Open position monitor
    open_mon = compute_open_monitor(current_opts)

    # 6. Display
    if cx:
        display_convexity(console, cx)
    display_bleed(console, bleed)
    display_concentration(console, conc)
    if exits:
        display_optimal_exits(console, exits)
    if open_mon:
        display_open_monitor(console, open_mon)
    if cx:
        display_scorecard(console, cx, bleed, conc, exits)

    console.print("\n[dim]All analysis is retrospective. Past performance, "
                  "optimal exits, and capture ratios are path-dependent — "
                  "they could not have been known at the time.[/dim]")


if __name__ == "__main__":
    main()

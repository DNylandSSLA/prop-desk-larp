"""
Serialize the full prop desk state into JSON for WebSocket broadcast.
"""

import time
from datetime import datetime

import numpy as np

from prop_desk import (
    EQUITY_TICKERS, FX_TICKERS,
    compute_trader_risk, aggregate_desk_risk, check_risk_limits, compute_greeks,
)
from bank_python import Bond, CreditDefaultSwap, Equity, FXRate, Option


def serialize_state(
    traders, state, config, db, tick, start_time,
    recent_orders, mc_results, stress_results, price_history,
    pnl_history=None, attribution=None, bump_ladder=None,
    correlation=None, trader_stats=None,
):
    """Build the full JSON snapshot sent to clients each tick."""
    now = datetime.now()

    return {
        "type": "tick",
        "tick": tick,
        "timestamp": now.isoformat(),
        "uptime": int(time.time() - start_time),
        "market_data": _serialize_market_data(state, price_history),
        "desk": _serialize_desk(traders),
        "traders": _serialize_traders(traders),
        "positions": _serialize_positions(traders),
        "alerts": _serialize_alerts(traders, config, db),
        "orders": recent_orders,
        "risk": _serialize_risk(mc_results),
        "stress": _serialize_stress(stress_results, traders),
        "pnl_series": pnl_history[-200:] if pnl_history else [],
        "attribution": attribution,
        "bump_ladder": bump_ladder,
        "correlation": correlation,
        "trader_stats": trader_stats,
    }


def _serialize_market_data(state, price_history):
    """Equities, FX, VIX, SOFR, spreads with sparkline data."""
    mgr = state["mgr"]
    equities = []

    # Top equities (first 30 for the ticker bar + table)
    for ticker in EQUITY_TICKERS[:30]:
        eq = state["equities"].get(ticker)
        if not eq or eq.value <= 0:
            continue
        hist = price_history.get(ticker, [])
        equities.append({
            "ticker": ticker,
            "price": round(eq.value, 2),
            "history": [round(p, 2) for p in hist[-30:]],
        })

    # FX
    fx_map = {
        "EURUSD": "eurusd_md", "GBPUSD": "gbpusd_md", "USDJPY": "usdjpy_md",
        "AUDUSD": "audusd_md", "USDCAD": "usdcad_md", "USDCHF": "usdchf_md",
    }
    fx = []
    for label, key in fx_map.items():
        md = state.get(key)
        if md:
            fx.append({"pair": label, "rate": round(md.value, 4)})

    return {
        "equities": equities,
        "fx": fx,
        "vix": round(state["vix_md"].value, 2),
        "sofr": round(state["sofr_md"].value, 4),
        "spread": round(state["spread_md"].value, 4),
    }


def _serialize_desk(traders):
    """Desk-level aggregates."""
    desk_risk = aggregate_desk_risk(traders)
    total_mv = sum(t.book.total_value for t in traders)
    total_pnl = sum(t.book.total_value - t.initial_value for t in traders)
    return {
        "total_mv": round(total_mv, 0),
        "total_pnl": round(total_pnl, 0),
        "delta": round(desk_risk["delta"], 0),
        "gamma": round(desk_risk["gamma"], 1),
        "vega": round(desk_risk["vega"], 0),
        "n_traders": len(traders),
        "n_positions": sum(len(t.book.positions) for t in traders),
    }


def _serialize_traders(traders):
    """Per-trader summary."""
    result = []
    for t in traders:
        risk = compute_trader_risk(t)
        pnl = t.book.total_value - t.initial_value
        result.append({
            "name": t.name,
            "strategy": t.strategy,
            "mv": round(t.book.total_value, 0),
            "pnl": round(pnl, 0),
            "delta": round(risk["delta"], 0),
            "gamma": round(risk["gamma"], 1),
            "vega": round(risk["vega"], 0),
            "n_positions": len(t.book.positions),
        })
    return result


def _serialize_positions(traders):
    """Flat list of all positions across traders."""
    positions = []
    for t in traders:
        for pos in t.book.positions:
            inst = pos.instrument
            inst_type = _instrument_type(inst)
            positions.append({
                "trader": t.name,
                "instrument": inst.name,
                "type": inst_type,
                "quantity": pos.quantity,
                "price": round(inst.value, 4) if inst.value else 0,
                "mv": round(pos.market_value, 0),
            })
    return positions


def _instrument_type(inst):
    if isinstance(inst, Option):
        return "option"
    elif isinstance(inst, Bond):
        return "bond"
    elif isinstance(inst, CreditDefaultSwap):
        return "cds"
    elif isinstance(inst, FXRate):
        return "fx"
    elif isinstance(inst, Equity):
        return "equity"
    return "other"


def _serialize_alerts(traders, config, db):
    """Current risk alerts."""
    alerts = check_risk_limits(traders, config, db)
    return [
        {
            "severity": a["severity"],
            "rule": a["rule"],
            "message": a["message"],
            "timestamp": a["timestamp"],
        }
        for a in alerts[:15]  # cap at 15 most recent
    ]


def _serialize_risk(mc_results):
    """VaR/CVaR from most recent MC run."""
    if not mc_results:
        return None

    desk_risk = mc_results.get("desk_risk", {})
    pnl = mc_results.get("pnl_desk")

    risk_data = {"var": {}, "contributions": {}}

    for conf, (var, cvar) in desk_risk.items():
        key = f"{int(conf * 100)}"
        risk_data["var"][key] = {
            "var": round(float(var), 0),
            "cvar": round(float(cvar), 0),
        }

    if pnl is not None:
        risk_data["prob_loss"] = round(float(np.mean(pnl < 0) * 100), 1)
        risk_data["mean_pnl"] = round(float(np.mean(pnl)), 0)
        risk_data["worst"] = round(float(np.min(pnl)), 0)
        risk_data["best"] = round(float(np.max(pnl)), 0)

    contributions = mc_results.get("risk_contributions", {})
    for name, pct in contributions.items():
        risk_data["contributions"][name] = round(float(pct), 1)

    return risk_data


def _serialize_stress(stress_results, traders):
    """Stress test scenario results."""
    if not stress_results:
        return None

    scenarios = {}
    for scenario_name, result in stress_results.items():
        per_trader = {}
        for t in traders:
            per_trader[t.name] = round(
                result.get("traders", {}).get(t.name, {}).get("pnl", 0), 0
            )
        scenarios[scenario_name] = {
            "traders": per_trader,
            "desk_pnl": round(result.get("desk_pnl", 0), 0),
        }
    return scenarios

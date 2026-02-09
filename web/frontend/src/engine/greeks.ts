/**
 * Greeks computation: delta, gamma, vega per position.
 * Exact port of prop_desk.compute_greeks / compute_trader_risk / aggregate_desk_risk.
 */

import { normCdf, normPdf } from "./math";
import type { MarketState } from "./market";
import type { Position, Trader } from "./traders";

export interface Greeks {
  delta: number;
  gamma: number;
  vega: number;
}

/** Compute Greeks for a single position. */
export function computeGreeks(pos: Position, market: MarketState): Greeks {
  const key = pos.instrumentKey;
  const qty = pos.quantity;

  if (pos.instrumentType === "option") {
    const opt = market.options.get(key);
    if (!opt) return { delta: 0, gamma: 0, vega: 0 };
    const eq = market.equities.get(opt.spotTicker);
    const S = eq ? eq.spotPrice : 0;
    const K = opt.strike;
    const sigma = opt.volatility;
    const T = opt.timeToExpiry;

    if (T <= 0 || sigma <= 0 || S <= 0) {
      return { delta: 0, gamma: 0, vega: 0 };
    }

    const sqrtT = Math.sqrt(T);
    const d1 = (Math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT);
    const nd1 = normCdf(d1);
    const npd1 = normPdf(d1);

    const deltaPer = opt.isCall ? nd1 : nd1 - 1.0;
    const gammaPer = npd1 / (S * sigma * sqrtT);
    const vegaPer = S * npd1 * sqrtT / 100.0; // per 1% vol move

    const mult = 100.0;
    return {
      delta: deltaPer * qty * mult,
      gamma: gammaPer * qty * mult,
      vega: vegaPer * qty * mult,
    };
  }

  if (pos.instrumentType === "bond") {
    const bond = market.bonds.get(key);
    if (!bond) return { delta: 0, gamma: 0, vega: 0 };
    const rate = bond.rate;
    let dur = 0;
    for (let t = 1; t <= bond.maturity; t++) {
      dur += t / Math.pow(1 + rate, t);
    }
    dur /= Math.max(bond.value, 0.01);
    return {
      delta: -dur * bond.value * qty,
      gamma: 0,
      vega: 0,
    };
  }

  // Equity, FX, CDS: delta = notional value
  const inst = market.getInstrument(key);
  if (!inst) return { delta: 0, gamma: 0, vega: 0 };
  return {
    delta: inst.value * qty,
    gamma: 0,
    vega: 0,
  };
}

/** Aggregate Greeks across all positions for a trader. */
export function computeTraderRisk(trader: Trader, market: MarketState): Greeks {
  const totals: Greeks = { delta: 0, gamma: 0, vega: 0 };
  for (const pos of trader.positions) {
    const g = computeGreeks(pos, market);
    totals.delta += g.delta;
    totals.gamma += g.gamma;
    totals.vega += g.vega;
  }
  return totals;
}

/** Sum Greeks across all traders. */
export function aggregateDeskRisk(traders: Trader[], market: MarketState): Greeks {
  const desk: Greeks = { delta: 0, gamma: 0, vega: 0 };
  for (const t of traders) {
    const risk = computeTraderRisk(t, market);
    desk.delta += risk.delta;
    desk.gamma += risk.gamma;
    desk.vega += risk.vega;
  }
  return desk;
}

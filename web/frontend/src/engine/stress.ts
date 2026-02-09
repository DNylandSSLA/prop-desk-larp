/**
 * 8 stress scenarios with Greek-based + BS repricing.
 * Exact port of mc_engine.StressEngine._apply_scenario().
 */

import { bsPrice, bondPv, cdsValue } from "./instruments";
import type { MarketState } from "./market";
import type { Trader } from "./traders";
import type { StressScenario } from "../types/simulation";

interface ScenarioDef {
  equityShock: number;
  rateShockBp: number;
  fxShock: number;
  vixLevel: number | null;
}

const STRESS_SCENARIOS: Record<string, ScenarioDef> = {
  "2008 GFC":      { equityShock: -0.40, rateShockBp: 200,  fxShock: 0.0,   vixLevel: 80.0 },
  "COVID":         { equityShock: -0.35, rateShockBp: -100, fxShock: 0.0,   vixLevel: 65.0 },
  "Rate Shock":    { equityShock: 0.0,   rateShockBp: 200,  fxShock: 0.0,   vixLevel: 35.0 },
  "Vol Spike":     { equityShock: -0.15, rateShockBp: 0,    fxShock: 0.0,   vixLevel: 50.0 },
  "USD Rally":     { equityShock: 0.0,   rateShockBp: 0,    fxShock: -0.10, vixLevel: null },
  "Tech Crash":    { equityShock: -0.25, rateShockBp: -50,  fxShock: 0.0,   vixLevel: 45.0 },
  "Stagflation":   { equityShock: -0.20, rateShockBp: 300,  fxShock: -0.05, vixLevel: 40.0 },
  "Credit Crunch": { equityShock: -0.15, rateShockBp: 150,  fxShock: 0.0,   vixLevel: 35.0 },
};

function applyScenario(
  traders: Trader[],
  market: MarketState,
  scenario: ScenarioDef,
): StressScenario {
  const { equityShock, rateShockBp, fxShock, vixLevel } = scenario;
  const stressedRate = market.sofr + rateShockBp / 10000;

  const perTrader: Record<string, number> = {};
  let deskPnl = 0;

  for (const trader of traders) {
    let traderPnl = 0;

    for (const pos of trader.positions) {
      const key = pos.instrumentKey;
      const qty = pos.quantity;

      if (pos.instrumentType === "option") {
        const opt = market.options.get(key);
        if (!opt) continue;
        const eq = market.equities.get(opt.spotTicker);
        const sCurrent = eq ? eq.spotPrice : 0;
        const sStressed = sCurrent * (1 + equityShock);

        let stressedVol = opt.volatility;
        if (vixLevel !== null && market.vix > 0) {
          stressedVol = opt.volatility * (vixLevel / market.vix);
        }

        const stressedPrice = bsPrice(
          sStressed, opt.strike, stressedVol, opt.timeToExpiry, opt.isCall,
        );
        const stressedMv = stressedPrice * qty * 100;
        const currentMv = opt.value * qty * 100;
        traderPnl += stressedMv - currentMv;

      } else if (pos.instrumentType === "bond") {
        const bond = market.bonds.get(key);
        if (!bond) continue;
        const currentMv = bond.value * qty;
        const stressedPv = bondPv(
          stressedRate, bond.face, bond.couponRate, bond.maturity,
        );
        traderPnl += stressedPv * qty - currentMv;

      } else if (pos.instrumentType === "fx") {
        const fx = market.fx.get(key);
        if (!fx) continue;
        const currentMv = fx.value * qty;
        const stressedMv = fx.value * (1 + fxShock) * qty;
        traderPnl += stressedMv - currentMv;

      } else if (pos.instrumentType === "equity") {
        const eq = market.equities.get(key);
        if (!eq) continue;
        const currentMv = eq.value * qty;
        const stressedMv = eq.value * (1 + equityShock) * qty;
        traderPnl += stressedMv - currentMv;

      } else if (pos.instrumentType === "cds") {
        // CDS: stressed spread = current spread + some shock
        // Approximate: equity shock correlated with spread widening
        const cds = market.cds.get(key);
        if (!cds) continue;
        const spreadMult = 1 + Math.abs(equityShock) * 3;
        const stressedSpread = market.spread * spreadMult;
        const stressedVal = cdsValue(cds.notional, cds.maturity, stressedSpread, stressedRate);
        traderPnl += (stressedVal - cds.value) * qty;
      }
    }

    perTrader[trader.name] = Math.round(traderPnl);
    deskPnl += traderPnl;
  }

  return {
    traders: perTrader,
    desk_pnl: Math.round(deskPnl),
  };
}

/** Run all 8 stress scenarios. */
export function runAllStressScenarios(
  traders: Trader[], market: MarketState,
): Record<string, StressScenario> {
  const results: Record<string, StressScenario> = {};
  for (const [name, scenario] of Object.entries(STRESS_SCENARIOS)) {
    results[name] = applyScenario(traders, market, scenario);
  }
  return results;
}

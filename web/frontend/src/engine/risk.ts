/**
 * Parametric delta-normal VaR/CVaR (replaces 50K-path Monte Carlo).
 * Risk limit checking → alert generation.
 */

import { mean, std } from "./math";
import type { MarketState } from "./market";
import type { Trader } from "./traders";
import { bookValue } from "./traders";
import { computeTraderRisk } from "./greeks";
import type { RiskData, AlertData } from "../types/simulation";
import { DEFAULT_CONFIG } from "./seed-data";

/** Z-scores for confidence levels. */
const Z_SCORES: Record<number, number> = {
  95: 1.6449,
  99: 2.3263,
};

/**
 * Parametric VaR: VaR = z * portfolio_sigma
 * Uses recent P&L changes to estimate portfolio sigma.
 */
export function computeParametricRisk(
  traders: Trader[],
  market: MarketState,
  traderPnlSeries: Map<string, number[]>,
): RiskData {
  // Desk-level P&L series
  const deskSeries: number[] = [];
  const nTicks = Math.min(
    ...Array.from(traderPnlSeries.values()).map((s) => s.length),
    300,
  );

  if (nTicks < 5) {
    return {
      var: {
        "95": { var: 0, cvar: 0 },
        "99": { var: 0, cvar: 0 },
      },
      contributions: {},
    };
  }

  // Build desk-level series from trader series
  for (let i = 0; i < nTicks; i++) {
    let deskPnl = 0;
    for (const [, series] of traderPnlSeries) {
      if (i < series.length) deskPnl += series[i];
    }
    deskSeries.push(deskPnl);
  }

  // Changes (tick-over-tick)
  const deskChanges: number[] = [];
  for (let i = 1; i < deskSeries.length; i++) {
    deskChanges.push(deskSeries[i] - deskSeries[i - 1]);
  }

  const deskStd = std(deskChanges, 1);
  const deskMean = mean(deskChanges);

  const varData: Record<string, { var: number; cvar: number }> = {};

  for (const conf of [95, 99]) {
    const z = Z_SCORES[conf];
    const varVal = z * deskStd;
    // CVaR for normal distribution: sigma * phi(z) / (1-alpha)
    const alpha = conf / 100;
    const phiZ = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
    const cvar = deskStd * phiZ / (1 - alpha) - deskMean;
    varData[String(conf)] = {
      var: Math.round(Math.max(0, varVal)),
      cvar: Math.round(Math.max(0, cvar)),
    };
  }

  // Risk contributions: proportion of covariance with desk
  const contributions: Record<string, number> = {};
  let totalAbsContrib = 0;

  for (const [name, series] of traderPnlSeries) {
    if (series.length < 5) continue;
    const changes: number[] = [];
    for (let i = 1; i < Math.min(series.length, deskSeries.length); i++) {
      changes.push(series[i] - series[i - 1]);
    }
    // Covariance with desk changes
    const n = Math.min(changes.length, deskChanges.length);
    if (n < 3) continue;
    const mA = mean(changes.slice(0, n));
    const mB = mean(deskChanges.slice(0, n));
    let covSum = 0;
    for (let i = 0; i < n; i++) {
      covSum += (changes[i] - mA) * (deskChanges[i] - mB);
    }
    const covVal = covSum / (n - 1);
    contributions[name] = covVal;
    totalAbsContrib += Math.abs(covVal);
  }

  // Normalize to percentages
  if (totalAbsContrib > 0) {
    for (const name in contributions) {
      contributions[name] = Math.round((contributions[name] / totalAbsContrib) * 1000) / 10;
    }
  }

  // Distribution stats
  const sortedChanges = deskChanges.slice().sort((a, b) => a - b);
  const probLoss = deskChanges.filter((c) => c < 0).length / deskChanges.length * 100;

  return {
    var: varData,
    contributions,
    prob_loss: Math.round(probLoss * 10) / 10,
    mean_pnl: Math.round(deskMean),
    worst: Math.round(sortedChanges[0] ?? 0),
    best: Math.round(sortedChanges[sortedChanges.length - 1] ?? 0),
  };
}

/** Check risk limits, generate alerts. */
export function checkRiskLimits(
  traders: Trader[], market: MarketState,
): AlertData[] {
  const alerts: AlertData[] = [];
  const now = new Date().toISOString();
  const cfg = DEFAULT_CONFIG;

  let deskPnl = 0;
  for (const t of traders) {
    deskPnl += bookValue(t, market) - t.initialValue;
  }

  if (deskPnl < cfg.deskLossLimit) {
    alerts.push({
      severity: "CRITICAL",
      rule: "DESK_LOSS",
      message: `Desk P&L $${deskPnl.toLocaleString("en-US", { maximumFractionDigits: 0, signDisplay: "always" })} < $${cfg.deskLossLimit.toLocaleString()}`,
      timestamp: now,
    });
  }

  for (const t of traders) {
    const pnl = bookValue(t, market) - t.initialValue;
    const risk = computeTraderRisk(t, market);

    if (pnl < cfg.traderLossLimit) {
      alerts.push({
        severity: "CRITICAL",
        rule: "DAILY_LOSS",
        message: `${t.name} P&L $${Math.round(pnl).toLocaleString("en-US", { signDisplay: "always" })} < $${cfg.traderLossLimit.toLocaleString()}`,
        timestamp: now,
      });
    }

    if (Math.abs(risk.delta) > cfg.maxDelta) {
      alerts.push({
        severity: "HIGH",
        rule: "DELTA_LIMIT",
        message: `${t.name} |delta| ${Math.abs(risk.delta).toLocaleString("en-US", { maximumFractionDigits: 0 })} > ${cfg.maxDelta.toLocaleString()}`,
        timestamp: now,
      });
    }

    if (Math.abs(risk.vega) > cfg.maxVega) {
      alerts.push({
        severity: "HIGH",
        rule: "VEGA_LIMIT",
        message: `${t.name} |vega| ${Math.abs(risk.vega).toLocaleString("en-US", { maximumFractionDigits: 0 })} > ${cfg.maxVega.toLocaleString()}`,
        timestamp: now,
      });
    }

    const totalBook = Math.abs(bookValue(t, market)) || 1;
    for (const pos of t.positions) {
      const inst = market.getInstrument(pos.instrumentKey);
      if (!inst) continue;
      const mv = Math.abs(
        pos.instrumentType === "option"
          ? inst.value * pos.quantity * 100
          : inst.value * pos.quantity,
      );
      const pct = (mv / totalBook) * 100;

      if (pct > cfg.maxConcentrationPct) {
        alerts.push({
          severity: "MEDIUM",
          rule: "CONCENTRATION",
          message: `${t.name} ${pos.instrumentName} ${pct.toFixed(0)}% > ${cfg.maxConcentrationPct}%`,
          timestamp: now,
        });
      }
      if (mv > cfg.maxPositionNotional) {
        alerts.push({
          severity: "MEDIUM",
          rule: "POSITION_SIZE",
          message: `${t.name} ${pos.instrumentName} $${mv.toLocaleString("en-US", { maximumFractionDigits: 0 })} > $${cfg.maxPositionNotional.toLocaleString()}`,
          timestamp: now,
        });
      }
    }
  }

  return alerts.slice(0, 15);
}

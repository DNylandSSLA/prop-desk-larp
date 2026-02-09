/**
 * P&L attribution, bump ladder, correlation matrix, trader stats.
 * Ports of SimulationEngine._compute_attribution, _compute_bump_ladder,
 * _compute_correlation, _compute_trader_stats.
 */

import { mean, std, cumMax, corrMatrix } from "./math";
import type { MarketState } from "./market";
import type { Trader } from "./traders";
import { bookValue } from "./traders";
import { computeTraderRisk, aggregateDeskRisk } from "./greeks";
import type { Greeks } from "./greeks";
import type {
  Attribution, BumpResult, CorrelationData, TraderStats, PnLPoint,
} from "../types/simulation";

export interface PrevState {
  greeks: Map<string, Greeks>;
  prices: Map<string, number>;
  vix: number;
}

/** Snapshot current Greeks and prices for next tick's attribution. */
export function snapshotPrevState(
  traders: Trader[], market: MarketState,
): PrevState {
  const greeks = new Map<string, Greeks>();
  for (const t of traders) {
    greeks.set(t.name, computeTraderRisk(t, market));
  }
  const prices = new Map<string, number>();
  for (const [ticker, eq] of market.equities) {
    if (eq.value > 0) prices.set(ticker, eq.value);
  }
  return { greeks, prices, vix: market.vix };
}

/** Record P&L for time series chart. */
export function recordPnl(
  tick: number,
  traders: Trader[],
  market: MarketState,
  pnlHistory: PnLPoint[],
  traderPnlSeries: Map<string, number[]>,
): void {
  let deskPnl = 0;
  const traderPnls: Record<string, number> = {};

  for (const t of traders) {
    const pnl = bookValue(t, market) - t.initialValue;
    deskPnl += pnl;
    traderPnls[t.name] = Math.round(pnl);

    let series = traderPnlSeries.get(t.name);
    if (!series) {
      series = [];
      traderPnlSeries.set(t.name, series);
    }
    series.push(pnl);
  }

  pnlHistory.push({
    tick,
    ts: new Date().toLocaleTimeString("en-US", { hour12: false }),
    desk: Math.round(deskPnl),
    traders: traderPnls,
  });

  // Keep last 300 ticks
  if (pnlHistory.length > 300) pnlHistory.shift();
}

/** P&L attribution using Greeks and price changes. */
export function computeAttribution(
  traders: Trader[],
  market: MarketState,
  prev: PrevState,
  pnlHistory: PnLPoint[],
): Attribution {
  let totalDeltaPnl = 0;
  let totalGammaPnl = 0;
  let totalVegaPnl = 0;

  // Gather spot returns from top 20 equities
  const tickers = Array.from(prev.prices.keys()).slice(0, 20);
  const spotReturns: number[] = [];
  const currentPrices: number[] = [];

  for (const ticker of tickers) {
    const eq = market.equities.get(ticker);
    const prevPx = prev.prices.get(ticker);
    if (eq && prevPx && prevPx > 0 && eq.value > 0) {
      spotReturns.push((eq.value - prevPx) / prevPx);
      currentPrices.push(eq.value);
    }
  }

  if (spotReturns.length > 0) {
    const avgReturn = mean(spotReturns);
    const avgPrice = mean(currentPrices);
    const _ds = avgReturn * avgPrice; // dollar move per share (unused directly)

    for (const t of traders) {
      const prevG = prev.greeks.get(t.name);
      if (!prevG) continue;

      totalDeltaPnl += prevG.delta * avgReturn;
      totalGammaPnl += 0.5 * prevG.gamma * avgReturn * avgReturn * avgPrice;
      totalVegaPnl +=
        prevG.vega * ((market.vix - prev.vix) / 100);
    }
  }

  // Current desk P&L
  let deskPnl = 0;
  for (const t of traders) {
    deskPnl += bookValue(t, market) - t.initialValue;
  }
  const prevDeskPnl = pnlHistory.length >= 2 ? pnlHistory[pnlHistory.length - 2].desk : 0;
  const tickPnl = deskPnl - prevDeskPnl;

  // Theta: approximate as time decay (~-0.1% of vega per tick)
  let totalTheta = 0;
  for (const t of traders) {
    const prevG = prev.greeks.get(t.name);
    if (prevG) {
      totalTheta += -Math.abs(prevG.vega) * 0.001;
    }
  }

  const explained = totalDeltaPnl + totalGammaPnl + totalVegaPnl + totalTheta;
  const unexplained = tickPnl - explained;

  return {
    delta_pnl: Math.round(totalDeltaPnl),
    gamma_pnl: Math.round(totalGammaPnl),
    vega_pnl: Math.round(totalVegaPnl),
    theta_pnl: Math.round(totalTheta),
    unexplained: Math.round(unexplained),
    total: Math.round(tickPnl),
  };
}

/** Compute P&L impact for parallel spot bumps using Greeks. */
export function computeBumpLadder(
  traders: Trader[], market: MarketState,
): BumpResult[] {
  const bumps = [-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0];
  const deskRisk = aggregateDeskRisk(traders, market);

  const ladder: BumpResult[] = [];
  for (const bumpPct of bumps) {
    const b = bumpPct / 100;
    const deskImpact = deskRisk.delta * b + 0.5 * deskRisk.gamma * b * b;

    const perTrader: Record<string, number> = {};
    for (const t of traders) {
      const risk = computeTraderRisk(t, market);
      perTrader[t.name] = Math.round(risk.delta * b + 0.5 * risk.gamma * b * b);
    }

    ladder.push({
      bump: bumpPct,
      desk: Math.round(deskImpact),
      traders: perTrader,
    });
  }
  return ladder;
}

/** Compute return correlation matrix from price history. */
export function computeCorrelation(
  market: MarketState,
): CorrelationData | null {
  const tickers: string[] = [];
  const returnRows: number[][] = [];

  for (const [ticker, hist] of market.priceHistory) {
    if (hist.length < 5) continue;
    // Compute returns
    const rets: number[] = [];
    for (let i = 1; i < hist.length; i++) {
      if (hist[i - 1] > 0) {
        rets.push((hist[i] - hist[i - 1]) / hist[i - 1]);
      }
    }
    if (rets.length >= 4) {
      tickers.push(ticker);
      returnRows.push(rets.slice(-20)); // last 20 returns
    }
    if (tickers.length >= 12) break; // cap at 12 for display
  }

  if (tickers.length < 3) return null;

  // Align to same length
  const minLen = Math.min(...returnRows.map((r) => r.length));
  const aligned = returnRows.map((r) => r.slice(r.length - minLen));

  const matrix = corrMatrix(aligned);
  const n = Math.min(12, tickers.length);

  return {
    tickers: tickers.slice(0, n),
    matrix: matrix.slice(0, n).map((row) => row.slice(0, n)),
  };
}

/** Compute rolling stats per trader: Sharpe, max drawdown, win rate. */
export function computeTraderStats(
  traders: Trader[],
  traderPnlSeries: Map<string, number[]>,
): TraderStats[] {
  const stats: TraderStats[] = [];

  for (const t of traders) {
    const series = traderPnlSeries.get(t.name) ?? [];
    if (series.length < 3) {
      stats.push({
        name: t.name, sharpe: 0, max_dd: 0, win_rate: 0,
        total_pnl: 0, n_ticks: series.length,
      });
      continue;
    }

    // Tick-over-tick changes
    const changes: number[] = [];
    for (let i = 1; i < series.length; i++) {
      changes.push(series[i] - series[i - 1]);
    }

    // Sharpe (annualized-ish: ~450 ticks/15min)
    const meanRet = mean(changes);
    const stdRet = std(changes, 1);
    const sharpe = stdRet > 0 ? (meanRet / stdRet) * Math.sqrt(450) : 0;

    // Max drawdown
    const peaks = cumMax(series);
    let maxDd = 0;
    for (let i = 0; i < series.length; i++) {
      const dd = series[i] - peaks[i];
      if (dd < maxDd) maxDd = dd;
    }

    // Win rate
    let wins = 0;
    for (const c of changes) {
      if (c > 0) wins++;
    }
    const winRate = changes.length > 0 ? (wins / changes.length) * 100 : 0;

    stats.push({
      name: t.name,
      sharpe: Math.round(sharpe * 100) / 100,
      max_dd: Math.round(maxDd),
      win_rate: Math.round(winRate * 10) / 10,
      total_pnl: Math.round(series[series.length - 1]),
      n_ticks: series.length,
    });
  }

  return stats;
}

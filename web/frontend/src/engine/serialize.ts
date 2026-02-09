/**
 * Assemble SimulationMessage matching types/simulation.ts exactly.
 */

import type { MarketState } from "./market";
import type { Trader } from "./traders";
import { bookValue, positionMv } from "./traders";
import { computeTraderRisk, aggregateDeskRisk } from "./greeks";
import { checkRiskLimits } from "./risk";
import { EQUITY_TICKERS } from "./seed-data";
import type {
  SimulationMessage, MarketData, DeskData, TraderData,
  PositionData, AlertData, OrderData, RiskData,
  StressScenario, PnLPoint, Attribution, BumpResult,
  CorrelationData, TraderStats,
} from "../types/simulation";

function serializeMarketData(market: MarketState): MarketData {
  const equities = [];
  // Top 30 equities for the ticker bar
  for (const ticker of EQUITY_TICKERS.slice(0, 30)) {
    const eq = market.equities.get(ticker);
    if (!eq || eq.value <= 0) continue;
    const hist = market.priceHistory.get(ticker) ?? [];
    equities.push({
      ticker,
      price: Math.round(eq.value * 100) / 100,
      history: hist.slice(-30).map((p) => Math.round(p * 100) / 100),
    });
  }

  const fxPairs = [
    { pair: "EURUSD", key: "eurusd" },
    { pair: "GBPUSD", key: "gbpusd" },
    { pair: "USDJPY", key: "usdjpy" },
    { pair: "AUDUSD", key: "audusd" },
    { pair: "USDCAD", key: "usdcad" },
    { pair: "USDCHF", key: "usdchf" },
  ];
  const fx = fxPairs
    .map(({ pair, key }) => {
      const f = market.fx.get(key);
      return f ? { pair, rate: Math.round(f.rate * 10000) / 10000 } : null;
    })
    .filter((x): x is { pair: string; rate: number } => x !== null);

  return {
    equities,
    fx,
    vix: Math.round(market.vix * 100) / 100,
    sofr: Math.round(market.sofr * 10000) / 10000,
    spread: Math.round(market.spread * 10000) / 10000,
  };
}

function serializeDesk(traders: Trader[], market: MarketState): DeskData {
  const deskRisk = aggregateDeskRisk(traders, market);
  let totalMv = 0;
  let totalPnl = 0;
  let nPositions = 0;
  for (const t of traders) {
    const mv = bookValue(t, market);
    totalMv += mv;
    totalPnl += mv - t.initialValue;
    nPositions += t.positions.length;
  }
  return {
    total_mv: Math.round(totalMv),
    total_pnl: Math.round(totalPnl),
    delta: Math.round(deskRisk.delta),
    gamma: Math.round(deskRisk.gamma * 10) / 10,
    vega: Math.round(deskRisk.vega),
    n_traders: traders.length,
    n_positions: nPositions,
  };
}

function serializeTraders(traders: Trader[], market: MarketState): TraderData[] {
  return traders.map((t) => {
    const risk = computeTraderRisk(t, market);
    const pnl = bookValue(t, market) - t.initialValue;
    return {
      name: t.name,
      strategy: t.strategy,
      mv: Math.round(bookValue(t, market)),
      pnl: Math.round(pnl),
      delta: Math.round(risk.delta),
      gamma: Math.round(risk.gamma * 10) / 10,
      vega: Math.round(risk.vega),
      n_positions: t.positions.length,
    };
  });
}

function serializePositions(traders: Trader[], market: MarketState): PositionData[] {
  const positions: PositionData[] = [];
  for (const t of traders) {
    for (const pos of t.positions) {
      const inst = market.getInstrument(pos.instrumentKey);
      positions.push({
        trader: t.name,
        instrument: pos.instrumentName,
        type: pos.instrumentType,
        quantity: pos.quantity,
        price: inst ? Math.round(inst.value * 10000) / 10000 : 0,
        mv: Math.round(positionMv(pos, market)),
      });
    }
  }
  return positions;
}

export function buildMessage(
  tick: number,
  startTime: number,
  traders: Trader[],
  market: MarketState,
  recentOrders: OrderData[],
  riskData: RiskData | null,
  stressData: Record<string, StressScenario> | null,
  pnlHistory: PnLPoint[],
  attribution: Attribution | null,
  bumpLadder: BumpResult[] | null,
  correlation: CorrelationData | null,
  traderStats: TraderStats[] | null,
): SimulationMessage {
  return {
    type: "tick",
    tick,
    timestamp: new Date().toISOString(),
    uptime: Math.floor((Date.now() - startTime) / 1000),
    market_data: serializeMarketData(market),
    desk: serializeDesk(traders, market),
    traders: serializeTraders(traders, market),
    positions: serializePositions(traders, market),
    alerts: checkRiskLimits(traders, market),
    orders: recentOrders,
    risk: riskData,
    stress: stressData,
    pnl_series: pnlHistory.slice(-200),
    attribution,
    bump_ladder: bumpLadder,
    correlation,
    trader_stats: traderStats,
  };
}

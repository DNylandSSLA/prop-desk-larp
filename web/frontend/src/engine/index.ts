/**
 * SimulationEngine — the main tick loop orchestrator.
 * Runs entirely in the browser, no server needed.
 */

import { PRNG } from "./prng";
import { MarketState } from "./market";
import { createTraders } from "./traders";
import type { Trader } from "./traders";
import type { PriceOverrides } from "./fetch-prices";
import {
  snapshotPrevState, recordPnl,
  computeAttribution, computeBumpLadder,
  computeCorrelation, computeTraderStats,
} from "./analytics";
import type { PrevState } from "./analytics";
import { computeParametricRisk } from "./risk";
import { runAllStressScenarios } from "./stress";
import { generateOrders } from "./orders";
import { buildMessage } from "./serialize";
import type { SimulationMessage, PnLPoint, Attribution, BumpResult,
  CorrelationData, TraderStats, RiskData, StressScenario, OrderData,
} from "../types/simulation";

export class SimulationEngine {
  private rng: PRNG;
  private market: MarketState;
  private traders: Trader[];
  private tickCount = 0;
  private startTime: number;

  // Analytics state
  private pnlHistory: PnLPoint[] = [];
  private traderPnlSeries = new Map<string, number[]>();
  private prevState: PrevState | null = null;
  private attribution: Attribution | null = null;
  private bumpLadder: BumpResult[] | null = null;
  private correlation: CorrelationData | null = null;
  private traderStats: TraderStats[] | null = null;
  private riskData: RiskData | null = null;
  private stressData: Record<string, StressScenario> | null = null;
  private recentOrders: OrderData[] = [];

  constructor(seed?: number, overrides?: PriceOverrides) {
    this.rng = new PRNG(seed ?? Date.now());
    this.market = new MarketState(overrides);
    this.traders = createTraders(this.market);
    this.startTime = Date.now();

    // Initial Greeks snapshot
    this.prevState = snapshotPrevState(this.traders, this.market);
  }

  /** Execute one tick. Returns the message to feed into the reducer. */
  tick(): SimulationMessage {
    this.tickCount++;

    // Jiggle market prices
    let intensity = 0.8 + 0.2 * Math.min(this.tickCount / 100, 1.0);
    if (this.rng.random() < 0.10) {
      intensity *= 2.0;
    }
    this.market.jiggle(this.rng, intensity);
    this.market.recordHistory();

    // Record P&L
    recordPnl(this.tickCount, this.traders, this.market, this.pnlHistory, this.traderPnlSeries);

    // P&L attribution (every tick)
    if (this.prevState) {
      this.attribution = computeAttribution(
        this.traders, this.market, this.prevState, this.pnlHistory,
      );
    }

    // Bump ladder (every 5 ticks)
    if (this.tickCount % 5 === 0) {
      this.bumpLadder = computeBumpLadder(this.traders, this.market);
    }

    // Correlation matrix (every 10 ticks)
    if (this.tickCount % 10 === 0) {
      this.correlation = computeCorrelation(this.market);
    }

    // Snapshot Greeks for next tick
    this.prevState = snapshotPrevState(this.traders, this.market);

    // Order generation (every 10 ticks)
    if (this.tickCount % 10 === 0) {
      const newOrders = generateOrders(this.traders, this.market, this.rng);
      this.recentOrders = [...newOrders, ...this.recentOrders].slice(0, 25);
      this.traderStats = computeTraderStats(this.traders, this.traderPnlSeries);
    }

    // Heavy analytics: parametric VaR + stress (every 30 ticks)
    if (this.tickCount % 30 === 0) {
      this.riskData = computeParametricRisk(
        this.traders, this.market, this.traderPnlSeries,
      );
      this.stressData = runAllStressScenarios(this.traders, this.market);
    }

    return buildMessage(
      this.tickCount,
      this.startTime,
      this.traders,
      this.market,
      this.recentOrders,
      this.riskData,
      this.stressData,
      this.pnlHistory,
      this.attribution,
      this.bumpLadder,
      this.correlation,
      this.traderStats,
    );
  }
}

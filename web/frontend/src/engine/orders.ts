/**
 * Personality-driven order generation.
 * Port of SimulationEngine._run_signals_and_orders().
 */

import type { PRNG } from "./prng";
import type { MarketState } from "./market";
import type { Trader } from "./traders";
import { bookValue } from "./traders";
import { TRADER_PROFILES } from "./seed-data";
import type { OrderData } from "../types/simulation";

const CYCLICALS = new Set(["TSLA", "COIN", "RIVN", "NVDA", "AMD"]);

export function generateOrders(
  traders: Trader[],
  market: MarketState,
  rng: PRNG,
): OrderData[] {
  const newOrders: OrderData[] = [];

  // Pick 4-7 traders to be active this round
  const nActive = rng.randInt(4, 8);
  const activeTraders = rng.sample(traders, Math.min(nActive, traders.length));

  for (const trader of activeTraders) {
    const profile = TRADER_PROFILES[trader.name];
    if (!profile) continue;

    // Pick 1-2 tickers from this trader's preferred list
    const nPicks = rng.randInt(1, 3);
    const picks = rng.sample(profile.tickers, Math.min(nPicks, profile.tickers.length));

    for (const sym of picks) {
      const eq = market.equities.get(sym);
      if (!eq || eq.value <= 0) continue;

      // Determine side from bias
      let side: "BUY" | "SELL";
      if (profile.bias === null) {
        side = rng.random() < 0.5 ? "BUY" : "SELL";
      } else if (rng.random() < 0.75) {
        side = profile.bias;
      } else {
        side = profile.bias === "BUY" ? "SELL" : "BUY";
      }

      // Claudia special: sells cyclicals, buys defensives
      if (trader.name === "Claudia") {
        side = CYCLICALS.has(sym) ? "SELL" : "BUY";
      }

      const bookVal = Math.max(Math.abs(bookValue(trader, market)), 50_000);
      const maxNotional = bookVal * 0.20;
      const maxQty = Math.max(1, Math.floor(maxNotional / eq.value));
      const qty = rng.randInt(1, Math.max(2, maxQty));

      const isLimit = rng.random() < profile.limitPct;
      const orderType = isLimit ? "LIMIT" : "MARKET";
      const limitPrice = isLimit
        ? Math.round(eq.value * (1 + rng.uniform(-0.02, 0.02)) * 100) / 100
        : null;

      // Simulate fill: MARKET always fills, LIMIT fills 60% of the time
      const fills = orderType === "MARKET" || rng.random() < 0.6;
      const fillPrice = fills
        ? Math.round(eq.value * (1 + rng.normal(0, 0.001)) * 100) / 100
        : null;

      newOrders.push({
        trader: trader.name,
        instrument: sym,
        side,
        quantity: qty,
        type: orderType,
        status: fills ? "FILLED" : "PENDING",
        fill_price: fillPrice,
        timestamp: new Date().toISOString(),
      });

      // If filled, update positions
      if (fills && fillPrice) {
        const signedQty = side === "BUY" ? qty : -qty;
        const existing = trader.positions.find(
          (p) => p.instrumentKey === sym,
        );
        if (existing) {
          existing.quantity += signedQty;
        } else {
          trader.positions.push({
            instrumentKey: sym,
            instrumentName: sym,
            instrumentType: "equity",
            quantity: signedQty,
          });
        }
      }
    }
  }

  return newOrders;
}

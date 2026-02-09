/**
 * Trader, Book, Position types and initialization from seed data.
 */

import type { MarketState } from "./market";
import { TRADER_DEFS } from "./seed-data";
import type { InstrumentType } from "./instruments";

export interface Position {
  instrumentKey: string;  // equity ticker or option/bond/cds/fx key
  instrumentName: string; // display name
  instrumentType: InstrumentType;
  quantity: number;
}

export interface Trader {
  name: string;
  strategy: string;
  positions: Position[];
  initialValue: number;
}

/** Resolve an instrument key to its type and display name. */
function resolveInstrument(
  key: string,
  market: MarketState,
): { name: string; type: InstrumentType } | null {
  const eq = market.equities.get(key);
  if (eq) return { name: eq.name, type: "equity" };
  const opt = market.options.get(key);
  if (opt) return { name: opt.name, type: "option" };
  const bond = market.bonds.get(key);
  if (bond) return { name: bond.name, type: "bond" };
  const cds = market.cds.get(key);
  if (cds) return { name: cds.name, type: "cds" };
  const fx = market.fx.get(key);
  if (fx) return { name: fx.name, type: "fx" };
  return null;
}

/** Compute market value of a position. */
export function positionMv(pos: Position, market: MarketState): number {
  const inst = market.getInstrument(pos.instrumentKey);
  if (!inst) return 0;
  // Options have 100x multiplier
  if (pos.instrumentType === "option") {
    return inst.value * pos.quantity * 100;
  }
  return inst.value * pos.quantity;
}

/** Total book market value for a trader. */
export function bookValue(trader: Trader, market: MarketState): number {
  let total = 0;
  for (const pos of trader.positions) {
    total += positionMv(pos, market);
  }
  return total;
}

/** Create all 14 traders from seed data. */
export function createTraders(market: MarketState): Trader[] {
  const traders: Trader[] = [];

  for (const def of TRADER_DEFS) {
    const positions: Position[] = [];
    for (const [key, qty] of def.positions) {
      const inst = resolveInstrument(key, market);
      if (!inst) continue;
      positions.push({
        instrumentKey: key,
        instrumentName: inst.name,
        instrumentType: inst.type,
        quantity: qty,
      });
    }

    const trader: Trader = {
      name: def.name,
      strategy: def.strategy,
      positions,
      initialValue: 0,
    };
    // Compute initial value
    trader.initialValue = bookValue(trader, market);
    traders.push(trader);
  }

  return traders;
}

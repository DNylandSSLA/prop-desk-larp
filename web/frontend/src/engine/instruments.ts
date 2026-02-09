/**
 * Instrument value models: Equity, Option (Black-Scholes), Bond (PV),
 * CDS (risky annuity), FX.
 */

import { normCdf } from "./math";

export type InstrumentType = "equity" | "option" | "bond" | "cds" | "fx";

export interface Instrument {
  name: string;
  type: InstrumentType;
  value: number;
}

export interface EquityInst extends Instrument {
  type: "equity";
  spotPrice: number;
}

export interface OptionInst extends Instrument {
  type: "option";
  spotTicker: string;
  strike: number;
  volatility: number;
  timeToExpiry: number;
  isCall: boolean;
}

export interface BondInst extends Instrument {
  type: "bond";
  face: number;
  couponRate: number;
  maturity: number;
  rate: number; // current discount rate (SOFR)
}

export interface CdsInst extends Instrument {
  type: "cds";
  notional: number;
  maturity: number;
  spread: number; // current credit spread
  rate: number;   // discount rate
}

export interface FxInst extends Instrument {
  type: "fx";
  pair: string;
  rate: number;
}

/** Black-Scholes price (no dividends, no discounting — matches Python). */
export function bsPrice(
  S: number, K: number, sigma: number, T: number, isCall: boolean,
): number {
  if (T <= 0 || sigma <= 0 || S <= 0) {
    return isCall ? Math.max(0, S - K) : Math.max(0, K - S);
  }
  const sqrtT = Math.sqrt(T);
  const d1 = (Math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT);
  const d2 = d1 - sigma * sqrtT;
  const nd1 = normCdf(d1);
  const nd2 = normCdf(d2);
  if (isCall) {
    return S * nd1 - K * nd2;
  } else {
    return K * (1 - nd2) - S * (1 - nd1);
  }
}

/** Bond present value. */
export function bondPv(rate: number, face: number, couponRate: number, maturity: number): number {
  const coupon = face * couponRate;
  let pv = 0;
  for (let t = 1; t <= maturity; t++) {
    pv += coupon / Math.pow(1 + rate, t);
  }
  pv += face / Math.pow(1 + rate, maturity);
  return pv;
}

/** CDS mark-to-market: risky annuity * (market spread - contract spread).
 *  Simplified: value = notional * (spread / (spread + rate)) * (1 - 1/(1+spread+rate)^maturity) */
export function cdsValue(
  notional: number, maturity: number, spread: number, rate: number,
): number {
  const hazard = spread;
  const discount = rate + hazard;
  if (discount <= 0) return 0;
  const riskyAnnuity =
    (1 - Math.pow(1 / (1 + discount), maturity)) / discount;
  return notional * spread * riskyAnnuity;
}

/** Revalue an option instrument given current spot. */
export function revalueOption(opt: OptionInst, spotPrice: number): number {
  return bsPrice(spotPrice, opt.strike, opt.volatility, opt.timeToExpiry, opt.isCall);
}

/** Revalue a bond given current rate. */
export function revalueBond(bond: BondInst): number {
  return bondPv(bond.rate, bond.face, bond.couponRate, bond.maturity);
}

/** Revalue a CDS given current spread and rate. */
export function revalueCds(cds: CdsInst): number {
  return cdsValue(cds.notional, cds.maturity, cds.spread, cds.rate);
}

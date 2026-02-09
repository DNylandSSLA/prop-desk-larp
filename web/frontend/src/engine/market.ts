/**
 * MarketState: holds all live prices and instruments.
 * jiggle() is a line-for-line port of Python _jiggle_market().
 */

import type { PRNG } from "./prng";
import {
  EQUITY_TICKERS, SEED_PRICES, FX_SEEDS,
  SEED_VIX, SEED_SOFR, SEED_SPREAD,
  OPTION_DEFS, BOND_DEFS, CDS_DEFS, OPTION_BASE_VOLS,
} from "./seed-data";
import type { EquityInst, OptionInst, BondInst, CdsInst, FxInst } from "./instruments";
import { bsPrice, bondPv, cdsValue } from "./instruments";
import type { PriceOverrides } from "./fetch-prices";

export class MarketState {
  equities: Map<string, EquityInst> = new Map();
  options: Map<string, OptionInst> = new Map();
  bonds: Map<string, BondInst> = new Map();
  cds: Map<string, CdsInst> = new Map();
  fx: Map<string, FxInst> = new Map();

  vix: number;
  sofr: number;
  spread: number;

  /** Ticker -> last 30 prices for sparklines */
  priceHistory: Map<string, number[]> = new Map();

  constructor(overrides?: PriceOverrides) {
    // Equities — use live prices when available, fall back to seeds
    for (const ticker of EQUITY_TICKERS) {
      const price = overrides?.prices?.[ticker] ?? SEED_PRICES[ticker] ?? 100;
      this.equities.set(ticker, {
        name: ticker, type: "equity", spotPrice: price, value: price,
      });
      this.priceHistory.set(ticker, [price]);
    }

    // VIX, SOFR, Spread
    this.vix = overrides?.vix ?? SEED_VIX;
    this.sofr = overrides?.sofr ?? SEED_SOFR;
    this.spread = SEED_SPREAD;

    // FX — use live rates when available
    for (const def of FX_SEEDS) {
      const rate = overrides?.fx?.[def.key] ?? def.rate;
      this.fx.set(def.key, {
        name: def.pair, type: "fx", pair: def.pair, rate, value: rate,
      });
    }

    // Options — compute initial BS price
    for (const def of OPTION_DEFS) {
      const spot = overrides?.prices?.[def.spotTicker] ?? SEED_PRICES[def.spotTicker] ?? 100;
      const val = bsPrice(spot, def.strike, def.volatility, def.timeToExpiry, def.isCall);
      this.options.set(def.key, {
        name: def.name,
        type: "option",
        spotTicker: def.spotTicker,
        strike: def.strike,
        volatility: def.volatility,
        timeToExpiry: def.timeToExpiry,
        isCall: def.isCall,
        value: val,
      });
    }

    // Bonds
    for (const def of BOND_DEFS) {
      const val = bondPv(this.sofr, def.face, def.couponRate, def.maturity);
      this.bonds.set(def.key, {
        name: def.name, type: "bond",
        face: def.face, couponRate: def.couponRate, maturity: def.maturity,
        rate: this.sofr, value: val,
      });
    }

    // CDS
    for (const def of CDS_DEFS) {
      const val = cdsValue(def.notional, def.maturity, this.spread, this.sofr);
      this.cds.set(def.key, {
        name: def.name, type: "cds",
        notional: def.notional, maturity: def.maturity,
        spread: this.spread, rate: this.sofr, value: val,
      });
    }
  }

  /** Get any instrument by key (equity ticker or named key). */
  getInstrument(key: string): { value: number; type: string } | undefined {
    return (
      this.equities.get(key) ??
      this.options.get(key) ??
      this.bonds.get(key) ??
      this.cds.get(key) ??
      this.fx.get(key)
    );
  }

  /** Port of Python _jiggle_market(). */
  jiggle(rng: PRNG, intensity: number): void {
    // Equity + ETF spot moves — correlated via a common market factor
    const marketFactor = rng.normal(0, 0.005 * intensity);
    for (const [, eq] of this.equities) {
      if (eq.spotPrice <= 0) continue;
      const idio = rng.normal(0, 0.008 * intensity);
      const move = marketFactor + idio;
      eq.spotPrice *= 1 + move;
      eq.value = eq.spotPrice;
    }

    // FX moves — independent random walks
    for (const [, fxInst] of this.fx) {
      const move = rng.normal(0, 0.003 * intensity);
      fxInst.rate *= 1 + move;
      fxInst.value = fxInst.rate;
    }

    // VIX — mean-reverting (Ornstein-Uhlenbeck toward 20)
    const vixMean = 20.0;
    const vixKappa = 0.15 * intensity;
    const vixVol = 2.0 * intensity;
    let newVix = this.vix + vixKappa * (vixMean - this.vix) + rng.normal(0, vixVol);
    newVix = Math.max(10.0, Math.min(80.0, newVix));
    this.vix = newVix;

    // Rescale option implied vols based on VIX
    const vixRatio = newVix / 20.0;
    for (const [key, opt] of this.options) {
      const baseVol = OPTION_BASE_VOLS[key];
      if (baseVol !== undefined) {
        opt.volatility = baseVol * (0.5 + 0.5 * vixRatio);
      }
    }

    // SOFR — small drift ±2bp
    const sofrShock = rng.normal(0, 0.0002 * intensity);
    this.sofr = Math.max(0.01, this.sofr + sofrShock);

    // Credit spread — mean-reverting around 2% with jumps
    const spreadMean = 0.02;
    let spreadShock =
      0.1 * (spreadMean - this.spread) + rng.normal(0, 0.002 * intensity);
    if (rng.random() < 0.05) {
      spreadShock += (rng.random() < 0.5 ? -1 : 1) * rng.uniform(0.005, 0.015);
    }
    this.spread = Math.max(0.003, this.spread + spreadShock);

    // Revalue all derived instruments
    this._revalue();
  }

  /** Record current prices into history (call after jiggle). */
  recordHistory(): void {
    for (const [ticker, eq] of this.equities) {
      if (eq.value <= 0) continue;
      let hist = this.priceHistory.get(ticker);
      if (!hist) {
        hist = [];
        this.priceHistory.set(ticker, hist);
      }
      hist.push(eq.value);
      if (hist.length > 30) hist.shift();
    }
  }

  private _revalue(): void {
    // Options
    for (const [, opt] of this.options) {
      const eq = this.equities.get(opt.spotTicker);
      const spot = eq ? eq.spotPrice : 0;
      opt.value = bsPrice(spot, opt.strike, opt.volatility, opt.timeToExpiry, opt.isCall);
    }

    // Bonds
    for (const [, bond] of this.bonds) {
      bond.rate = this.sofr;
      bond.value = bondPv(this.sofr, bond.face, bond.couponRate, bond.maturity);
    }

    // CDS
    for (const [, c] of this.cds) {
      c.spread = this.spread;
      c.rate = this.sofr;
      c.value = cdsValue(c.notional, c.maturity, this.spread, this.sofr);
    }
  }
}

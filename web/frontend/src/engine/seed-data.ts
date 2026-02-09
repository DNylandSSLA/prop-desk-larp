/**
 * Static seed data for the client-side simulation.
 * Approximate prices as of mid-2025; the jiggle function will drift them.
 */

export const EQUITY_TICKERS = [
  "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "SPY",
  "AVGO", "CRM", "ORCL", "ADBE", "AMD", "INTC", "NFLX", "CSCO", "TXN",
  "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "PANW",
  "NOW", "SHOP", "SNOW", "DDOG", "NET", "CRWD", "ZS", "TEAM", "WDAY",
  "DASH", "RBLX", "U", "TTD", "ROKU", "PINS", "SNAP", "SPOT",
  "JPM", "GS", "BAC", "MS", "V", "MA", "BRK-B", "C", "WFC", "AXP",
  "SCHW", "BLK", "ICE", "CME", "SPGI", "MCO", "COF", "USB",
  "JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK", "TMO", "ABT", "AMGN",
  "BMY", "GILD", "ISRG", "MDT", "SYK", "REGN", "VRTX", "MRNA", "BIIB",
  "WMT", "COST", "KO", "PEP", "MCD", "NKE", "DIS", "SBUX", "TGT",
  "LOW", "HD", "TJX", "ROST", "YUM", "CMG", "ABNB", "BKNG",
  "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
  "CAT", "BA", "GE", "HON", "LMT", "RTX", "DE", "UNP", "UPS", "FDX",
  "WM", "EMR", "ITW", "GD", "NOC",
  "LIN", "APD", "FCX", "NEM", "NUE", "STLD",
  "T", "VZ", "TMUS", "CMCSA", "CHTR", "WBD",
  "PLD", "AMT", "CCI", "EQIX", "SPG", "O",
  "PYPL", "COIN", "PLTR", "UBER", "LYFT", "RIVN", "LCID", "SOFI",
  "HOOD", "AFRM", "MELI", "SE", "NU",
  "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY",
  "XLB", "XLU", "XLRE", "GLD", "SLV", "TLT", "HYG", "LQD", "EEM", "EFA",
  "VWO", "ARKK", "SOXX", "SMH", "KWEB", "FXI", "VNQ", "IBIT",
] as const;

/** Approximate mid-2025 prices. Good enough — they'll jiggle immediately. */
export const SEED_PRICES: Record<string, number> = {
  AAPL: 230, MSFT: 450, TSLA: 340, NVDA: 135, GOOGL: 180, AMZN: 200, META: 620, SPY: 580,
  AVGO: 185, CRM: 310, ORCL: 170, ADBE: 480, AMD: 165, INTC: 30, NFLX: 780, CSCO: 60, TXN: 195,
  QCOM: 185, MU: 115, AMAT: 205, LRCX: 105, KLAC: 780, MRVL: 110, SNPS: 570, CDNS: 310, PANW: 380,
  NOW: 925, SHOP: 105, SNOW: 170, DDOG: 130, NET: 115, CRWD: 370, ZS: 230, TEAM: 260, WDAY: 270,
  DASH: 185, RBLX: 60, U: 25, TTD: 110, ROKU: 80, PINS: 38, SNAP: 13, SPOT: 560,
  JPM: 235, GS: 540, BAC: 43, MS: 115, V: 305, MA: 510, "BRK-B": 460, C: 70, WFC: 70, AXP: 270,
  SCHW: 80, BLK: 960, ICE: 160, CME: 230, SPGI: 500, MCO: 480, COF: 175, USB: 48,
  JNJ: 155, UNH: 580, PFE: 27, LLY: 880, ABBV: 190, MRK: 125, TMO: 570, ABT: 115, AMGN: 310,
  BMY: 50, GILD: 95, ISRG: 510, MDT: 88, SYK: 380, REGN: 1100, VRTX: 470, MRNA: 120, BIIB: 180,
  WMT: 85, COST: 900, KO: 70, PEP: 170, MCD: 300, NKE: 77, DIS: 110, SBUX: 100, TGT: 135,
  LOW: 270, HD: 380, TJX: 115, ROST: 155, YUM: 140, CMG: 62, ABNB: 155, BKNG: 4700,
  XOM: 115, CVX: 160, COP: 110, SLB: 48, EOG: 130, MPC: 175, PSX: 140, VLO: 145, OXY: 55,
  CAT: 370, BA: 195, GE: 195, HON: 215, LMT: 480, RTX: 130, DE: 400, UNP: 240, UPS: 135, FDX: 275,
  WM: 215, EMR: 120, ITW: 260, GD: 300, NOC: 500,
  LIN: 470, APD: 310, FCX: 48, NEM: 50, NUE: 155, STLD: 160,
  T: 22, VZ: 43, TMUS: 210, CMCSA: 38, CHTR: 370, WBD: 10,
  PLD: 120, AMT: 210, CCI: 100, EQIX: 850, SPG: 165, O: 58,
  PYPL: 75, COIN: 260, PLTR: 80, UBER: 78, LYFT: 15, RIVN: 14, LCID: 3, SOFI: 12,
  HOOD: 30, AFRM: 55, MELI: 1800, SE: 110, NU: 14,
  QQQ: 510, IWM: 230, DIA: 430, XLF: 44, XLE: 90, XLK: 225, XLV: 150, XLI: 130, XLP: 80, XLY: 195,
  XLB: 90, XLU: 75, XLRE: 42, GLD: 230, SLV: 27, TLT: 90, HYG: 78, LQD: 108, EEM: 44, EFA: 82,
  VWO: 44, ARKK: 55, SOXX: 250, SMH: 260, KWEB: 30, FXI: 30, VNQ: 90, IBIT: 55,
};

export interface FxSeed {
  pair: string;
  key: string;
  rate: number;
  base: string;
  quote: string;
}

export const FX_SEEDS: FxSeed[] = [
  { pair: "EURUSD", key: "eurusd", rate: 1.08, base: "EUR", quote: "USD" },
  { pair: "GBPUSD", key: "gbpusd", rate: 1.27, base: "GBP", quote: "USD" },
  { pair: "USDJPY", key: "usdjpy", rate: 150.0, base: "USD", quote: "JPY" },
  { pair: "AUDUSD", key: "audusd", rate: 0.65, base: "AUD", quote: "USD" },
  { pair: "USDCAD", key: "usdcad", rate: 1.35, base: "USD", quote: "CAD" },
  { pair: "USDCHF", key: "usdchf", rate: 0.88, base: "USD", quote: "CHF" },
];

export const SEED_VIX = 18.0;
export const SEED_SOFR = 0.053;
export const SEED_SPREAD = 0.02;

export interface OptionDef {
  name: string;
  key: string;
  spotTicker: string;
  strike: number;
  volatility: number;
  timeToExpiry: number;
  isCall: boolean;
}

export const OPTION_DEFS: OptionDef[] = [
  { name: "AAPL_C230", key: "aapl_call", spotTicker: "AAPL", strike: 230, volatility: 0.25, timeToExpiry: 0.5, isCall: true },
  { name: "MSFT_P400", key: "msft_put", spotTicker: "MSFT", strike: 400, volatility: 0.22, timeToExpiry: 0.5, isCall: false },
  { name: "NVDA_C200", key: "nvda_call", spotTicker: "NVDA", strike: 200, volatility: 0.35, timeToExpiry: 0.5, isCall: true },
  { name: "NVDA_P170", key: "nvda_put", spotTicker: "NVDA", strike: 170, volatility: 0.35, timeToExpiry: 0.5, isCall: false },
  { name: "TSLA_C450", key: "tsla_call", spotTicker: "TSLA", strike: 450, volatility: 0.40, timeToExpiry: 0.5, isCall: true },
  { name: "META_C700", key: "meta_call", spotTicker: "META", strike: 700, volatility: 0.28, timeToExpiry: 0.5, isCall: true },
];

export interface BondDef {
  name: string;
  key: string;
  face: number;
  couponRate: number;
  maturity: number;
}

export const BOND_DEFS: BondDef[] = [
  { name: "BOND_5Y", key: "bond_5y", face: 100, couponRate: 0.06, maturity: 5 },
  { name: "BOND_10Y", key: "bond_10y", face: 100, couponRate: 0.055, maturity: 10 },
  { name: "BOND_2Y", key: "bond_2y", face: 100, couponRate: 0.05, maturity: 2 },
];

export interface CdsDef {
  name: string;
  key: string;
  notional: number;
  maturity: number;
}

export const CDS_DEFS: CdsDef[] = [
  { name: "CDS_IG", key: "cds_ig", notional: 5_000_000, maturity: 5 },
  { name: "CDS_HY", key: "cds_hy", notional: 5_000_000, maturity: 3 },
];

/** Position definition: [instrumentKey, quantity] where instrumentKey is
 *  either an equity ticker or a key from OPTION_DEFS/BOND_DEFS/CDS_DEFS/FX_SEEDS */
export type PosDef = [string, number];

export interface TraderDef {
  name: string;
  strategy: string;
  positions: PosDef[];
}

export const TRADER_DEFS: TraderDef[] = [
  {
    name: "Joe", strategy: "macro rates + duration",
    positions: [
      ["bond_10y", 400], ["bond_5y", 300], ["TLT", 200], ["GLD", 250],
      ["eurusd", 40_000], ["usdjpy", -35_000], ["XOM", 80], ["CAT", 60], ["SPY", -40],
    ],
  },
  {
    name: "Tracy", strategy: "credit + plumbing",
    positions: [
      ["cds_ig", 2], ["cds_hy", -1], ["bond_10y", 200], ["LQD", 180], ["HYG", -80],
      ["JPM", 120], ["BAC", 150], ["GS", 80], ["C", 100], ["msft_put", -8],
    ],
  },
  {
    name: "Matt", strategy: "relative value",
    positions: [
      ["GOOGL", 90], ["META", -70], ["XOM", 100], ["CVX", -80],
      ["V", 55], ["MA", -50], ["HD", 45], ["LOW", -40],
      ["SPY", -120], ["AAPL", 60], ["MSFT", 50], ["NVDA", 30],
    ],
  },
  {
    name: "Katie", strategy: "equities flow",
    positions: [
      ["AAPL", 180], ["MSFT", 120], ["NVDA", 80], ["LLY", 60], ["UNH", 40],
      ["JNJ", 100], ["COST", 50], ["AMZN", 70], ["QQQ", 60],
    ],
  },
  {
    name: "Nero", strategy: "credit arb",
    positions: [
      ["cds_ig", 1], ["cds_hy", -2], ["bond_10y", 150], ["HYG", -60], ["LQD", 100],
      ["nvda_put", 20], ["BAC", 120], ["GLD", 80],
    ],
  },
  {
    name: "Tony", strategy: "tech conviction",
    positions: [
      ["NVDA", 150], ["AVGO", 80], ["AMD", 100], ["MSFT", 60], ["META", 90],
      ["CRM", 50], ["NOW", 35], ["PANW", 45], ["TSLA", 40], ["NFLX", 30], ["SPY", -50],
    ],
  },
  {
    name: "Adam", strategy: "geopolitical macro",
    positions: [
      ["EEM", 300], ["EFA", 200], ["eurusd", 35_000], ["gbpusd", -20_000],
      ["bond_5y", 250], ["GLD", 150], ["XOM", 60], ["LMT", 40], ["SPY", 80],
    ],
  },
  {
    name: "Charlie", strategy: "quant vol",
    positions: [
      ["nvda_call", 20], ["nvda_put", 18], ["NVDA", -110], ["meta_call", 12],
      ["META", -45], ["aapl_call", -8], ["tsla_call", 10], ["TSLA", -30],
    ],
  },
  {
    name: "Izzy", strategy: "commodities + inflation",
    positions: [
      ["XOM", 120], ["CVX", 90], ["COP", 80], ["SLB", 100], ["FCX", 150],
      ["NEM", 100], ["GLD", 120], ["audusd", 30_000], ["usdcad", -25_000], ["bond_10y", -100],
    ],
  },
  {
    name: "Claudia", strategy: "defensive macro",
    positions: [
      ["JNJ", 150], ["PFE", 120], ["KO", 100], ["PEP", 80], ["WMT", 90],
      ["COST", 60], ["bond_10y", 300], ["TSLA", -30], ["COIN", -40], ["RIVN", -50],
    ],
  },
  {
    name: "Noah", strategy: "growth momentum",
    positions: [
      ["COIN", 250], ["PLTR", 400], ["SOFI", 500], ["HOOD", 300], ["AFRM", 200],
      ["SHOP", 60], ["DDOG", 80], ["NET", 100], ["SPY", -60],
    ],
  },
  {
    name: "Skanda", strategy: "rates + curve",
    positions: [
      ["bond_2y", 500], ["bond_10y", -200], ["bond_5y", 150], ["TLT", -80],
      ["JPM", 70], ["SCHW", 60], ["BLK", 40], ["msft_put", 10],
    ],
  },
  {
    name: "Torsten", strategy: "cross-asset tactical",
    positions: [
      ["AAPL", 70], ["GOOGL", 50], ["JPM", 60], ["eurusd", 25_000],
      ["usdjpy", -20_000], ["bond_5y", 100], ["cds_ig", 1], ["GLD", 50],
      ["EEM", 80], ["QQQ", 40], ["nvda_call", 5],
    ],
  },
  {
    name: "Robin", strategy: "FX + EM",
    positions: [
      ["eurusd", 50_000], ["gbpusd", 35_000], ["audusd", 30_000],
      ["usdjpy", -40_000], ["usdcad", -20_000], ["EEM", 250], ["EFA", 180],
      ["MELI", 40], ["NU", 200], ["SE", 80],
    ],
  },
];

export interface TraderProfile {
  tickers: string[];
  bias: "BUY" | "SELL" | null;
  limitPct: number;
}

export const TRADER_PROFILES: Record<string, TraderProfile> = {
  Joe:     { tickers: ["TLT", "GLD", "XOM", "CAT", "SPY", "GE", "DE"], bias: "BUY", limitPct: 0.4 },
  Tracy:   { tickers: ["JPM", "BAC", "GS", "C", "WFC", "MS", "LQD", "HYG"], bias: "BUY", limitPct: 0.3 },
  Matt:    { tickers: ["GOOGL", "META", "XOM", "CVX", "V", "MA", "HD", "LOW", "AAPL", "MSFT", "SPY"], bias: null, limitPct: 0.5 },
  Katie:   { tickers: ["AAPL", "MSFT", "NVDA", "LLY", "UNH", "AMZN", "COST", "JNJ", "QQQ", "GOOGL"], bias: "BUY", limitPct: 0.2 },
  Nero:    { tickers: ["BAC", "GLD", "LQD", "HYG", "SPY"], bias: null, limitPct: 0.6 },
  Tony:    { tickers: ["NVDA", "AVGO", "AMD", "MSFT", "META", "CRM", "NOW", "PANW", "TSLA", "NFLX", "DDOG", "NET"], bias: "BUY", limitPct: 0.15 },
  Adam:    { tickers: ["EEM", "EFA", "GLD", "XOM", "LMT", "SPY", "RTX"], bias: "BUY", limitPct: 0.35 },
  Charlie: { tickers: ["NVDA", "META", "TSLA", "AAPL", "MSFT"], bias: null, limitPct: 0.4 },
  Izzy:    { tickers: ["XOM", "CVX", "COP", "SLB", "FCX", "NEM", "GLD", "EOG", "MPC", "OXY"], bias: "BUY", limitPct: 0.3 },
  Claudia: { tickers: ["JNJ", "PFE", "KO", "PEP", "WMT", "COST", "TLT", "TSLA", "COIN", "RIVN"], bias: null, limitPct: 0.3 },
  Noah:    { tickers: ["COIN", "PLTR", "SOFI", "HOOD", "AFRM", "SHOP", "DDOG", "NET", "RBLX", "U"], bias: "BUY", limitPct: 0.1 },
  Skanda:  { tickers: ["TLT", "JPM", "SCHW", "BLK", "SPY"], bias: null, limitPct: 0.5 },
  Torsten: { tickers: ["AAPL", "GOOGL", "JPM", "EEM", "QQQ", "GLD", "META", "AMZN"], bias: "BUY", limitPct: 0.25 },
  Robin:   { tickers: ["EEM", "EFA", "MELI", "NU", "SE", "SPY"], bias: "BUY", limitPct: 0.35 },
};

export const DEFAULT_CONFIG = {
  deskLossLimit: -200_000,
  traderLossLimit: -50_000,
  maxDelta: 100_000,
  maxVega: 50_000,
  maxConcentrationPct: 40,
  maxPositionNotional: 500_000,
};

/** Option keys whose vol should scale with VIX */
export const OPTION_BASE_VOLS: Record<string, number> = {
  aapl_call: 0.25,
  msft_put: 0.22,
  nvda_call: 0.35,
  nvda_put: 0.35,
  tsla_call: 0.40,
  meta_call: 0.28,
};

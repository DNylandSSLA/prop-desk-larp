export interface EquityData {
  ticker: string;
  price: number;
  history: number[];
}

export interface FxData {
  pair: string;
  rate: number;
}

export interface MarketData {
  equities: EquityData[];
  fx: FxData[];
  vix: number;
  sofr: number;
  spread: number;
}

export interface DeskData {
  total_mv: number;
  total_pnl: number;
  delta: number;
  gamma: number;
  vega: number;
  n_traders: number;
  n_positions: number;
}

export interface TraderData {
  name: string;
  strategy: string;
  mv: number;
  pnl: number;
  delta: number;
  gamma: number;
  vega: number;
  n_positions: number;
}

export interface PositionData {
  trader: string;
  instrument: string;
  type: string;
  quantity: number;
  price: number;
  mv: number;
}

export interface AlertData {
  severity: "CRITICAL" | "HIGH" | "MEDIUM";
  rule: string;
  message: string;
  timestamp: string;
}

export interface OrderData {
  trader: string;
  instrument: string;
  side: "BUY" | "SELL";
  quantity: number;
  type: string;
  status: string;
  fill_price: number | null;
  timestamp: string;
}

export interface VarLevel {
  var: number;
  cvar: number;
}

export interface RiskData {
  var: Record<string, VarLevel>;
  contributions: Record<string, number>;
  prob_loss?: number;
  mean_pnl?: number;
  worst?: number;
  best?: number;
}

export interface StressScenario {
  traders: Record<string, number>;
  desk_pnl: number;
}

// New analytics types

export interface PnLPoint {
  tick: number;
  ts: string;
  desk: number;
  traders: Record<string, number>;
}

export interface Attribution {
  delta_pnl: number;
  gamma_pnl: number;
  vega_pnl: number;
  theta_pnl: number;
  unexplained: number;
  total: number;
}

export interface BumpResult {
  bump: number;
  desk: number;
  traders: Record<string, number>;
}

export interface CorrelationData {
  tickers: string[];
  matrix: number[][];
}

export interface TraderStats {
  name: string;
  sharpe: number;
  max_dd: number;
  win_rate: number;
  total_pnl: number;
  n_ticks: number;
}

export interface SimulationState {
  connected: boolean;
  tick: number;
  timestamp: string;
  uptime: number;
  market_data: MarketData | null;
  desk: DeskData | null;
  traders: TraderData[];
  positions: PositionData[];
  alerts: AlertData[];
  orders: OrderData[];
  risk: RiskData | null;
  stress: Record<string, StressScenario> | null;
  pnl_series: PnLPoint[];
  attribution: Attribution | null;
  bump_ladder: BumpResult[] | null;
  correlation: CorrelationData | null;
  trader_stats: TraderStats[] | null;
  view: number;
}

export type SimulationMessage = {
  type: "tick";
  tick: number;
  timestamp: string;
  uptime: number;
  market_data: MarketData;
  desk: DeskData;
  traders: TraderData[];
  positions: PositionData[];
  alerts: AlertData[];
  orders: OrderData[];
  risk: RiskData | null;
  stress: Record<string, StressScenario> | null;
  pnl_series: PnLPoint[];
  attribution: Attribution | null;
  bump_ladder: BumpResult[] | null;
  correlation: CorrelationData | null;
  trader_stats: TraderStats[] | null;
};

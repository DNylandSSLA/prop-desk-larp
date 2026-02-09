import { useReducer, useCallback } from "react";
import type { SimulationState, SimulationMessage } from "../types/simulation";

const initialState: SimulationState = {
  connected: false,
  tick: 0,
  timestamp: "",
  uptime: 0,
  market_data: null,
  desk: null,
  traders: [],
  positions: [],
  alerts: [],
  orders: [],
  risk: null,
  stress: null,
  pnl_series: [],
  attribution: null,
  bump_ladder: null,
  correlation: null,
  trader_stats: null,
  view: 1,
};

type Action =
  | { type: "TICK"; payload: SimulationMessage }
  | { type: "SET_CONNECTED"; payload: boolean }
  | { type: "SET_VIEW"; payload: number };

function reducer(state: SimulationState, action: Action): SimulationState {
  switch (action.type) {
    case "TICK":
      return {
        ...state,
        tick: action.payload.tick,
        timestamp: action.payload.timestamp,
        uptime: action.payload.uptime,
        market_data: action.payload.market_data,
        desk: action.payload.desk,
        traders: action.payload.traders,
        positions: action.payload.positions,
        alerts: action.payload.alerts,
        orders: action.payload.orders,
        risk: action.payload.risk ?? state.risk,
        stress: action.payload.stress ?? state.stress,
        pnl_series: action.payload.pnl_series ?? state.pnl_series,
        attribution: action.payload.attribution ?? state.attribution,
        bump_ladder: action.payload.bump_ladder ?? state.bump_ladder,
        correlation: action.payload.correlation ?? state.correlation,
        trader_stats: action.payload.trader_stats ?? state.trader_stats,
      };
    case "SET_CONNECTED":
      return { ...state, connected: action.payload };
    case "SET_VIEW":
      return { ...state, view: action.payload };
    default:
      return state;
  }
}

export function useSimulationState() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const handleMessage = useCallback((data: unknown) => {
    const msg = data as SimulationMessage;
    if (msg.type === "tick") {
      dispatch({ type: "TICK", payload: msg });
    }
  }, []);

  const setConnected = useCallback((connected: boolean) => {
    dispatch({ type: "SET_CONNECTED", payload: connected });
  }, []);

  const setView = useCallback((view: number) => {
    dispatch({ type: "SET_VIEW", payload: view });
  }, []);

  return { state, handleMessage, setConnected, setView };
}

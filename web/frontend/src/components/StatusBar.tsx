import type { SimulationState } from "../types/simulation";

interface StatusBarProps {
  state: SimulationState;
  viewName?: string;
}

export function StatusBar({ state, viewName }: StatusBarProps) {
  const upMinutes = Math.floor(state.uptime / 60);
  const upSeconds = state.uptime % 60;

  return (
    <div className="status-bar">
      <span className="status-item">
        <span
          className={`status-dot ${state.connected ? "connected" : "disconnected"}`}
        />
        {state.connected ? "CONNECTED" : "DISCONNECTED"}
      </span>
      {viewName && (
        <span className="status-item view-indicator">
          {viewName}
        </span>
      )}
      <span className="status-item">
        T+{state.tick}
      </span>
      <span className="status-item">
        UP {upMinutes}:{upSeconds.toString().padStart(2, "0")}
      </span>
      {state.desk && (
        <>
          <span className="status-item">
            {state.desk.n_traders} BOOKS
          </span>
          <span className="status-item">
            {state.desk.n_positions} POSITIONS
          </span>
        </>
      )}
      <span className="status-item" style={{ marginLeft: "auto" }}>
        {state.timestamp
          ? new Date(state.timestamp).toLocaleTimeString("en-US", { hour12: false })
          : "--:--:--"}
      </span>
      <span className="status-item dimmed">
        all persons fictitious
      </span>
      <span className="status-item dimmed">
        not real money
      </span>
      <span className="status-item brand">BANK PYTHON</span>
    </div>
  );
}

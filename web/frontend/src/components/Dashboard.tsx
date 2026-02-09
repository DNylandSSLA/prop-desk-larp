import { useEffect, useRef } from "react";
import type { SimulationState } from "../types/simulation";
import { TopBar } from "./TopBar";
import { MarketTicker } from "./MarketTicker";
import { MarketDataPanel } from "./MarketDataPanel";
import { PnLPanel } from "./PnLPanel";
import { RiskPanel } from "./RiskPanel";
import { PositionsPanel } from "./PositionsPanel";
import { AlertsFeed } from "./AlertsFeed";
import { OrderFeed } from "./OrderFeed";
import { StressPanel } from "./StressPanel";
import { StatusBar } from "./StatusBar";
import { PnLChart } from "./PnLChart";
import { AttributionPanel } from "./AttributionPanel";
import { BumpLadder } from "./BumpLadder";
import { CorrelationMatrix } from "./CorrelationMatrix";
import { TraderStatsPanel } from "./TraderStatsPanel";

const VIEW_NAMES = ["", "TRADING", "RISK ANALYTICS", "PERFORMANCE"];

interface DashboardProps {
  state: SimulationState;
  setView: (view: number) => void;
}

// Simple beep using Web Audio API
let audioCtx: AudioContext | null = null;
function playAlertBeep() {
  try {
    if (!audioCtx) audioCtx = new AudioContext();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.frequency.value = 880;
    osc.type = "square";
    gain.gain.value = 0.05;
    osc.start();
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.15);
    osc.stop(audioCtx.currentTime + 0.15);
  } catch {
    // Audio not available
  }
}

export function Dashboard({ state, setView }: DashboardProps) {
  const prevAlertCount = useRef(0);

  // Hotkey listener for view switching
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      // Don't intercept if typing in an input
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === "1") setView(1);
      else if (e.key === "2") setView(2);
      else if (e.key === "3") setView(3);
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [setView]);

  // Audio alert on new critical alerts
  useEffect(() => {
    const criticalCount = state.alerts.filter(
      (a) => a.severity === "CRITICAL"
    ).length;
    if (criticalCount > prevAlertCount.current && prevAlertCount.current >= 0) {
      playAlertBeep();
    }
    prevAlertCount.current = criticalCount;
  }, [state.alerts]);

  if (!state.connected && state.tick === 0) {
    return (
      <div className="loading-screen">
        <div className="loading-logo">BANK PYTHON</div>
        <div className="loading-text">
          Initializing prop desk simulation...
        </div>
        <div className="loading-sub">
          14 traders, ~180 equities, live yfinance data, zero actual money
        </div>
        <div className="loading-disclaimer">
          Any resemblance to actual persons, living or dead, or actual
          trading desks, solvent or otherwise, is purely coincidental.
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <TopBar state={state} />
      <MarketTicker data={state.market_data} />

      <div className="view-tabs">
        {[1, 2, 3].map((v) => (
          <button
            key={v}
            className={`view-tab ${state.view === v ? "active" : ""}`}
            onClick={() => setView(v)}
          >
            <span className="view-tab-key">{v}</span> {VIEW_NAMES[v]}
          </button>
        ))}
      </div>

      {state.view === 1 && <TradingView state={state} />}
      {state.view === 2 && <RiskAnalyticsView state={state} />}
      {state.view === 3 && <PerformanceView state={state} />}

      <StatusBar state={state} viewName={VIEW_NAMES[state.view]} />
    </div>
  );
}

function TradingView({ state }: { state: SimulationState }) {
  return (
    <>
      <div className="dashboard-grid">
        <div className="grid-col-1">
          <MarketDataPanel data={state.market_data} />
        </div>
        <div className="grid-col-2">
          <PnLPanel desk={state.desk} traders={state.traders} />
        </div>
        <div className="grid-col-3">
          <RiskPanel
            desk={state.desk}
            risk={state.risk}
            traders={state.traders}
          />
        </div>
      </div>

      <div className="dashboard-grid bottom-grid">
        <div className="grid-col-bottom-1">
          <PositionsPanel positions={state.positions} />
        </div>
        <div className="grid-col-bottom-2">
          <AlertsFeed alerts={state.alerts} />
          <OrderFeed orders={state.orders} />
          <StressPanel stress={state.stress} traders={state.traders} />
        </div>
      </div>
    </>
  );
}

function RiskAnalyticsView({ state }: { state: SimulationState }) {
  return (
    <>
      <div className="dashboard-grid">
        <div className="grid-col-1">
          <PnLChart series={state.pnl_series} />
          <AttributionPanel attribution={state.attribution} />
        </div>
        <div className="grid-col-2">
          <BumpLadder ladder={state.bump_ladder} traders={state.traders} />
          <StressPanel stress={state.stress} traders={state.traders} />
        </div>
        <div className="grid-col-3">
          <RiskPanel
            desk={state.desk}
            risk={state.risk}
            traders={state.traders}
          />
        </div>
      </div>

      <div className="dashboard-grid bottom-grid">
        <div className="grid-col-bottom-1">
          <CorrelationMatrix data={state.correlation} />
        </div>
        <div className="grid-col-bottom-2">
          <AlertsFeed alerts={state.alerts} />
          <OrderFeed orders={state.orders} />
        </div>
      </div>
    </>
  );
}

function PerformanceView({ state }: { state: SimulationState }) {
  return (
    <>
      <div className="dashboard-grid" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="grid-col-1">
          <PnLChart series={state.pnl_series} />
          <TraderStatsPanel stats={state.trader_stats} />
        </div>
        <div className="grid-col-2">
          <PnLPanel desk={state.desk} traders={state.traders} />
          <CorrelationMatrix data={state.correlation} />
        </div>
      </div>

      <div className="dashboard-grid bottom-grid">
        <div className="grid-col-bottom-1">
          <AttributionPanel attribution={state.attribution} />
          <BumpLadder ladder={state.bump_ladder} traders={state.traders} />
        </div>
        <div className="grid-col-bottom-2">
          <AlertsFeed alerts={state.alerts} />
          <StressPanel stress={state.stress} traders={state.traders} />
        </div>
      </div>
    </>
  );
}

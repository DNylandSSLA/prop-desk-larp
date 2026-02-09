import { useState, useEffect } from "react";
import type { SimulationState } from "../types/simulation";

interface TopBarProps {
  state: SimulationState;
}

export function TopBar({ state }: TopBarProps) {
  const [clock, setClock] = useState("");

  useEffect(() => {
    const update = () => {
      const now = new Date();
      setClock(
        now.toLocaleTimeString("en-US", { hour12: false }) +
          "." +
          String(now.getMilliseconds()).padStart(3, "0").slice(0, 2)
      );
    };
    update();
    const id = setInterval(update, 100);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="top-bar">
      <span className="top-bar-logo">BANK PYTHON</span>
      <span className="top-bar-subtitle">PROP DESK LARP</span>
      <input
        className="top-bar-cmd"
        placeholder="this doesn't do anything lol"
        readOnly
      />
      <div className="top-bar-info">
        <span>{state.desk?.n_traders ?? 0} DESKS</span>
        <span>{state.desk?.n_positions ?? 0} POS</span>
        {state.tick > 0 && <span>T{state.tick}</span>}
      </div>
      <div className="top-bar-clock">{clock}</div>
    </div>
  );
}

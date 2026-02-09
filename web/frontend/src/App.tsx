import { useEffect } from "react";
import { useSimulation } from "./hooks/useSimulation";
import { useSimulationState } from "./hooks/useSimulationState";
import { Dashboard } from "./components/Dashboard";

export function App() {
  const { state, handleMessage, setConnected, setView } = useSimulationState();
  const { connected } = useSimulation({ onMessage: handleMessage });

  useEffect(() => {
    setConnected(connected);
  }, [connected, setConnected]);

  return <Dashboard state={state} setView={setView} />;
}

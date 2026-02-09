import { useEffect } from "react";
import { useReplay } from "./hooks/useReplay";
import { useSimulationState } from "./hooks/useSimulationState";
import { Dashboard } from "./components/Dashboard";

export function App() {
  const { state, handleMessage, setConnected, setView } = useSimulationState();
  const { connected } = useReplay({ onMessage: handleMessage });

  useEffect(() => {
    setConnected(connected);
  }, [connected, setConnected]);

  return <Dashboard state={state} setView={setView} />;
}

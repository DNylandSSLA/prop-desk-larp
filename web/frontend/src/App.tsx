import { useEffect } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import { useSimulationState } from "./hooks/useSimulationState";
import { Dashboard } from "./components/Dashboard";

const WS_URL =
  import.meta.env.DEV
    ? `ws://${window.location.hostname}:5173/ws/simulation/`
    : `ws://${window.location.host}/ws/simulation/`;

export function App() {
  const { state, handleMessage, setConnected, setView } = useSimulationState();
  const { connected } = useWebSocket({ url: WS_URL, onMessage: handleMessage });

  useEffect(() => {
    setConnected(connected);
  }, [connected, setConnected]);

  return <Dashboard state={state} setView={setView} />;
}

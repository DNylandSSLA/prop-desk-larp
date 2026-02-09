import { useEffect, useRef, useState, useCallback } from "react";
import { SimulationEngine } from "../engine";

interface UseSimulationOptions {
  onMessage: (data: unknown) => void;
}

export function useSimulation({ onMessage }: UseSimulationOptions) {
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    const engine = new SimulationEngine();
    setConnected(true);

    // Send first tick immediately
    const msg = engine.tick();
    onMessageRef.current(msg);

    // Then tick every 2 seconds
    const id = setInterval(() => {
      const msg = engine.tick();
      onMessageRef.current(msg);
    }, 2000);

    return () => {
      clearInterval(id);
    };
  }, []);

  return { connected };
}

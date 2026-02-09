import { useEffect, useRef, useState, useCallback } from "react";
import { SimulationEngine } from "../engine";
import { fetchLivePrices } from "../engine/fetch-prices";

interface UseSimulationOptions {
  onMessage: (data: unknown) => void;
}

export function useSimulation({ onMessage }: UseSimulationOptions) {
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    (async () => {
      // Fetch live prices — never throws, returns partial/empty on failure
      const overrides = await fetchLivePrices();
      if (cancelled) return;

      const engine = new SimulationEngine(undefined, overrides);
      setConnected(true);

      // Send first tick immediately
      const msg = engine.tick();
      onMessageRef.current(msg);

      // Then tick every 2 seconds
      intervalId = setInterval(() => {
        const msg = engine.tick();
        onMessageRef.current(msg);
      }, 2000);
    })();

    return () => {
      cancelled = true;
      if (intervalId !== undefined) clearInterval(intervalId);
    };
  }, []);

  return { connected };
}

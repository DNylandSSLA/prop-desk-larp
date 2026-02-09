import { useEffect, useRef, useState, useCallback } from "react";

interface UseReplayOptions {
  onMessage: (data: unknown) => void;
}

export function useReplay({ onMessage }: UseReplayOptions) {
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const startPlayback = useCallback((ticks: unknown[]) => {
    let index = 0;
    setConnected(true);

    // Send first tick immediately
    const sendTick = () => {
      const tick = JSON.parse(JSON.stringify(ticks[index])) as Record<string, unknown>;
      // Patch timestamp to current wall-clock time so the clock looks live
      tick.timestamp = new Date().toISOString();
      onMessageRef.current(tick);
      index = (index + 1) % ticks.length;
    };

    sendTick();
    const id = setInterval(sendTick, 2000);
    return id;
  }, []);

  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval> | null = null;

    fetch(`${import.meta.env.BASE_URL}replay-data.json`)
      .then((r) => r.json())
      .then((ticks: unknown[]) => {
        intervalId = startPlayback(ticks);
      })
      .catch((err) => {
        console.error("Failed to load replay data:", err);
      });

    return () => {
      if (intervalId !== null) clearInterval(intervalId);
    };
  }, [startPlayback]);

  return { connected };
}

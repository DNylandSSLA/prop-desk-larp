import { useRef, useEffect } from "react";
import type { MarketData } from "../types/simulation";
import { formatPrice } from "../utils/format";

interface MarketTickerProps {
  data: MarketData | null;
}

export function MarketTicker({ data }: MarketTickerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    let animId: number;
    let pos = 0;
    const speed = 0.5;

    function animate() {
      pos += speed;
      if (pos >= el!.scrollWidth / 2) pos = 0;
      el!.scrollLeft = pos;
      animId = requestAnimationFrame(animate);
    }
    animId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animId);
  }, []);

  if (!data) {
    return <div className="market-ticker">Loading market data...</div>;
  }

  const items = [
    ...data.equities.slice(0, 25).map((e) => {
      const prev = e.history.length >= 2
        ? e.history[e.history.length - 2]
        : e.price;
      const chg = ((e.price - prev) / prev) * 100;
      return {
        label: e.ticker,
        value: formatPrice(e.price),
        change: chg,
      };
    }),
    ...data.fx.map((f) => ({
      label: f.pair,
      value: f.rate.toFixed(4),
      change: 0,
    })),
    { label: "VIX", value: data.vix.toFixed(2), change: 0 },
    { label: "SOFR", value: (data.sofr * 100).toFixed(2) + "%", change: 0 },
  ];

  // Duplicate for seamless scroll
  const allItems = [...items, ...items];

  return (
    <div className="market-ticker" ref={scrollRef}>
      <div className="ticker-track">
        {allItems.map((item, i) => (
          <span key={i} className="ticker-item">
            <span className="ticker-label">{item.label}</span>
            <span className="ticker-value">{item.value}</span>
            {item.change !== 0 && (
              <span
                className={`ticker-change ${item.change > 0 ? "up" : "down"}`}
              >
                {item.change > 0 ? "\u25B2" : "\u25BC"}{Math.abs(item.change).toFixed(2)}%
              </span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

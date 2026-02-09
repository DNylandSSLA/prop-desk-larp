import type { MarketData } from "../types/simulation";
import { formatPrice } from "../utils/format";
import { Sparkline } from "./Sparkline";

interface MarketDataPanelProps {
  data: MarketData | null;
}

export function MarketDataPanel({ data }: MarketDataPanelProps) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-header">Market Data</div>
        <div className="panel-body dimmed">Waiting for data...</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">Market Data</div>
      <div className="panel-body">
        <table className="data-table">
          <thead>
            <tr>
              <th>Ticker</th>
              <th className="right">Last</th>
              <th className="right">Chg</th>
              <th className="right">Chg%</th>
              <th>Trend</th>
            </tr>
          </thead>
          <tbody>
            {data.equities.slice(0, 20).map((e) => {
              const prev =
                e.history.length >= 2
                  ? e.history[e.history.length - 2]
                  : e.price;
              const chgAbs = e.price - prev;
              const chgPct = ((e.price - prev) / prev) * 100;
              const color = chgAbs >= 0 ? "var(--green)" : "var(--red)";
              const sign = chgAbs >= 0 ? "+" : "";
              return (
                <tr key={e.ticker} className="flash-row">
                  <td style={{ color: "var(--text-bright)" }}>{e.ticker}</td>
                  <td className="right mono">{formatPrice(e.price)}</td>
                  <td className="right mono" style={{ color }}>
                    {sign}{chgAbs.toFixed(2)}
                  </td>
                  <td className="right mono" style={{ color }}>
                    {sign}{chgPct.toFixed(2)}%
                  </td>
                  <td>
                    <Sparkline
                      data={e.history}
                      width={50}
                      height={12}
                      color={color}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="subsection">
          <div className="subsection-header">FX Spot</div>
          <table className="data-table compact">
            <thead>
              <tr>
                <th>Pair</th>
                <th className="right">Mid</th>
              </tr>
            </thead>
            <tbody>
              {data.fx.map((f) => (
                <tr key={f.pair}>
                  <td style={{ color: "var(--text-bright)" }}>{f.pair}</td>
                  <td className="right mono">{f.rate.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="subsection">
          <div className="subsection-header">Indices / Rates</div>
          <table className="data-table compact">
            <tbody>
              <tr>
                <td style={{ color: "var(--text-bright)" }}>CBOE VIX</td>
                <td className="right mono">{data.vix.toFixed(2)}</td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-bright)" }}>SOFR</td>
                <td className="right mono">{(data.sofr * 100).toFixed(3)}%</td>
              </tr>
              <tr>
                <td style={{ color: "var(--text-bright)" }}>IG Spread</td>
                <td className="right mono">{(data.spread * 10000).toFixed(0)} bps</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

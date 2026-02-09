import type { PositionData } from "../types/simulation";
import { formatDollar, formatPrice, pnlColor } from "../utils/format";

interface PositionsPanelProps {
  positions: PositionData[];
}

export function PositionsPanel({ positions }: PositionsPanelProps) {
  const sorted = [...positions]
    .sort((a, b) => Math.abs(b.mv) - Math.abs(a.mv))
    .slice(0, 40);

  return (
    <div className="panel">
      <div className="panel-header">
        Position Book
        <span className="panel-badge">{positions.length}</span>
      </div>
      <div className="panel-body scrollable">
        <table className="data-table compact">
          <thead>
            <tr>
              <th>Book</th>
              <th>Instrument</th>
              <th>Type</th>
              <th className="right">Qty</th>
              <th className="right">Avg Px</th>
              <th className="right">Notional</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((p, i) => (
              <tr key={i}>
                <td style={{ color: "var(--text-bright)" }}>{p.trader}</td>
                <td className="mono">{p.instrument}</td>
                <td className="dimmed">{p.type}</td>
                <td className="right mono">{p.quantity.toLocaleString()}</td>
                <td className="right mono">{formatPrice(p.price)}</td>
                <td
                  className="right mono"
                  style={{ color: pnlColor(p.mv) }}
                >
                  {formatDollar(p.mv)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

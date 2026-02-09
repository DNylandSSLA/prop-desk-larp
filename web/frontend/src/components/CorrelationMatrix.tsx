import type { CorrelationData } from "../types/simulation";

interface CorrelationMatrixProps {
  data: CorrelationData | null;
}

function corrColor(v: number): string {
  if (v >= 0.8) return "rgba(0,200,83,0.6)";
  if (v >= 0.5) return "rgba(0,200,83,0.3)";
  if (v >= 0.2) return "rgba(0,200,83,0.1)";
  if (v >= -0.2) return "transparent";
  if (v >= -0.5) return "rgba(255,61,61,0.15)";
  if (v >= -0.8) return "rgba(255,61,61,0.3)";
  return "rgba(255,61,61,0.5)";
}

function corrText(v: number): string {
  if (v >= 0.5) return "var(--green)";
  if (v <= -0.5) return "var(--red)";
  return "var(--text-dim)";
}

export function CorrelationMatrix({ data }: CorrelationMatrixProps) {
  if (!data) {
    return (
      <div className="panel">
        <div className="panel-header">Return Correlation</div>
        <div className="panel-body dimmed">Accumulating returns...</div>
      </div>
    );
  }

  const { tickers, matrix } = data;

  return (
    <div className="panel">
      <div className="panel-header">Return Correlation (Rolling)</div>
      <div className="panel-body scrollable">
        <table className="data-table compact corr-table">
          <thead>
            <tr>
              <th></th>
              {tickers.map((t) => (
                <th key={t} className="center corr-th">{t}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tickers.map((rowTicker, i) => (
              <tr key={rowTicker}>
                <td className="mono" style={{ color: "var(--text-bright)" }}>{rowTicker}</td>
                {matrix[i].map((val, j) => (
                  <td
                    key={j}
                    className="center mono corr-cell"
                    style={{
                      background: i === j ? "var(--border)" : corrColor(val),
                      color: i === j ? "var(--text-dim)" : corrText(val),
                    }}
                  >
                    {i === j ? "1.00" : val.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

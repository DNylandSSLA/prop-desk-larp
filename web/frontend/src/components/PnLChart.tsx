import type { PnLPoint } from "../types/simulation";
import { formatDollar, pnlColor } from "../utils/format";

interface PnLChartProps {
  series: PnLPoint[];
}

export function PnLChart({ series }: PnLChartProps) {
  if (series.length < 2) {
    return (
      <div className="panel">
        <div className="panel-header">Intraday P&L</div>
        <div className="panel-body dimmed">Accumulating data...</div>
      </div>
    );
  }

  const W = 600;
  const H = 160;
  const PAD_L = 55;
  const PAD_R = 10;
  const PAD_T = 10;
  const PAD_B = 20;
  const chartW = W - PAD_L - PAD_R;
  const chartH = H - PAD_T - PAD_B;

  const values = series.map((p) => p.desk);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 0);
  const range = max - min || 1;

  const toX = (i: number) => PAD_L + (i / (values.length - 1)) * chartW;
  const toY = (v: number) => PAD_T + (1 - (v - min) / range) * chartH;

  // Build SVG path
  const points = values.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`);
  const linePath = `M${points.join("L")}`;

  // Fill area to zero line
  const zeroY = toY(0);
  const fillPath = `${linePath}L${toX(values.length - 1).toFixed(1)},${zeroY.toFixed(1)}L${toX(0).toFixed(1)},${zeroY.toFixed(1)}Z`;

  const latest = values[values.length - 1];
  const color = latest >= 0 ? "var(--green)" : "var(--red)";
  const fillColor = latest >= 0 ? "rgba(0,200,83,0.12)" : "rgba(255,61,61,0.12)";

  // Y-axis labels
  const yLabels = [max, max * 0.5, 0, min * 0.5, min].filter(
    (v, i, a) => a.indexOf(v) === i
  );

  // X-axis time labels (every ~20%)
  const xStep = Math.max(1, Math.floor(series.length / 5));
  const xLabels = series.filter((_, i) => i % xStep === 0 || i === series.length - 1);

  return (
    <div className="panel">
      <div className="panel-header">
        Intraday P&L
        <span className="panel-badge" style={{ color, background: "transparent", border: "none" }}>
          {formatDollar(latest)}
        </span>
      </div>
      <div className="panel-body" style={{ padding: "4px" }}>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} className="pnl-chart">
          {/* Grid lines */}
          {yLabels.map((v) => (
            <line
              key={v}
              x1={PAD_L}
              y1={toY(v)}
              x2={W - PAD_R}
              y2={toY(v)}
              stroke="var(--border)"
              strokeWidth="0.5"
            />
          ))}

          {/* Zero line */}
          <line
            x1={PAD_L}
            y1={zeroY}
            x2={W - PAD_R}
            y2={zeroY}
            stroke="var(--text-dim)"
            strokeWidth="0.5"
            strokeDasharray="3,3"
          />

          {/* Fill area */}
          <path d={fillPath} fill={fillColor} />

          {/* P&L line */}
          <path d={linePath} fill="none" stroke={color} strokeWidth="1.5" />

          {/* Latest value dot */}
          <circle
            cx={toX(values.length - 1)}
            cy={toY(latest)}
            r="3"
            fill={color}
          />

          {/* Y-axis labels */}
          {yLabels.map((v) => (
            <text
              key={v}
              x={PAD_L - 4}
              y={toY(v) + 3}
              textAnchor="end"
              fill="var(--text-dim)"
              fontSize="8"
              fontFamily="var(--font-mono)"
            >
              {formatDollar(v)}
            </text>
          ))}

          {/* X-axis labels */}
          {xLabels.map((p) => {
            const i = series.indexOf(p);
            return (
              <text
                key={p.tick}
                x={toX(i)}
                y={H - 2}
                textAnchor="middle"
                fill="var(--text-dim)"
                fontSize="7"
                fontFamily="var(--font-mono)"
              >
                {p.ts}
              </text>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

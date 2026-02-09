import type { TraderStats } from "../types/simulation";
import { formatDollar, pnlColor } from "../utils/format";

interface TraderStatsPanelProps {
  stats: TraderStats[] | null;
}

function sharpeColor(s: number): string {
  if (s >= 2) return "var(--green)";
  if (s >= 1) return "var(--green-dim)";
  if (s >= 0) return "var(--amber)";
  if (s >= -1) return "var(--red-dim)";
  return "var(--red)";
}

export function TraderStatsPanel({ stats }: TraderStatsPanelProps) {
  if (!stats || stats.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">Trader Performance</div>
        <div className="panel-body dimmed">Accumulating data...</div>
      </div>
    );
  }

  const sorted = [...stats].sort((a, b) => b.total_pnl - a.total_pnl);

  return (
    <div className="panel">
      <div className="panel-header">Trader Performance (Rolling)</div>
      <div className="panel-body scrollable">
        <table className="data-table compact">
          <thead>
            <tr>
              <th>#</th>
              <th>Trader</th>
              <th className="right">Total P&L</th>
              <th className="right">Sharpe</th>
              <th className="right">Max DD</th>
              <th className="right">Win %</th>
              <th className="right">Ticks</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s, i) => (
              <tr key={s.name}>
                <td className="dimmed">{i + 1}</td>
                <td style={{ color: "var(--text-bright)" }}>{s.name}</td>
                <td className="right mono" style={{ color: pnlColor(s.total_pnl) }}>
                  {formatDollar(s.total_pnl)}
                </td>
                <td className="right mono" style={{ color: sharpeColor(s.sharpe) }}>
                  {s.sharpe.toFixed(2)}
                </td>
                <td className="right mono" style={{ color: s.max_dd < 0 ? "var(--red)" : "var(--text-dim)" }}>
                  {formatDollar(s.max_dd)}
                </td>
                <td className="right mono">
                  {s.win_rate.toFixed(1)}%
                </td>
                <td className="right dimmed">{s.n_ticks}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

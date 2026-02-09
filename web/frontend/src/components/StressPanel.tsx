import type { StressScenario, TraderData } from "../types/simulation";
import { formatDollar, pnlColor } from "../utils/format";

interface StressPanelProps {
  stress: Record<string, StressScenario> | null;
  traders: TraderData[];
}

export function StressPanel({ stress, traders }: StressPanelProps) {
  if (!stress) {
    return (
      <div className="panel">
        <div className="panel-header">Stress / Scenario Analysis</div>
        <div className="panel-body dimmed">Computing scenarios...</div>
      </div>
    );
  }

  const scenarios = Object.entries(stress);
  const allValues = scenarios.flatMap(([, s]) =>
    Object.values(s.traders).concat(s.desk_pnl)
  );
  const maxAbs = Math.max(...allValues.map(Math.abs), 1);

  return (
    <div className="panel">
      <div className="panel-header">Stress / Scenario Analysis</div>
      <div className="panel-body scrollable">
        <table className="data-table compact stress-table">
          <thead>
            <tr>
              <th>Scenario</th>
              {traders.slice(0, 7).map((t) => (
                <th key={t.name} className="right">
                  {t.name}
                </th>
              ))}
              <th className="right bold">DESK</th>
            </tr>
          </thead>
          <tbody>
            {scenarios.map(([name, data]) => (
              <tr key={name}>
                <td className="mono scenario-name">{name}</td>
                {traders.slice(0, 7).map((t) => {
                  const val = data.traders[t.name] ?? 0;
                  const intensity = Math.abs(val) / maxAbs;
                  const bg =
                    val >= 0
                      ? `rgba(0,200,100,${intensity * 0.4})`
                      : `rgba(200,50,50,${intensity * 0.4})`;
                  return (
                    <td
                      key={t.name}
                      className="right mono"
                      style={{ background: bg, color: pnlColor(val) }}
                    >
                      {formatDollar(val)}
                    </td>
                  );
                })}
                <td
                  className="right mono bold"
                  style={{ color: pnlColor(data.desk_pnl) }}
                >
                  {formatDollar(data.desk_pnl)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

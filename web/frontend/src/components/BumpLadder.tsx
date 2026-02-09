import type { BumpResult, TraderData } from "../types/simulation";
import { formatDollar, pnlColor } from "../utils/format";

interface BumpLadderProps {
  ladder: BumpResult[] | null;
  traders: TraderData[];
}

export function BumpLadder({ ladder, traders }: BumpLadderProps) {
  if (!ladder) {
    return (
      <div className="panel">
        <div className="panel-header">Greeks Sensitivity Ladder</div>
        <div className="panel-body dimmed">Computing...</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">Greeks Sensitivity Ladder (Parallel Spot Bump)</div>
      <div className="panel-body scrollable">
        <table className="data-table compact stress-table">
          <thead>
            <tr>
              <th>Bump</th>
              {traders.slice(0, 7).map((t) => (
                <th key={t.name} className="right">{t.name}</th>
              ))}
              <th className="right bold">DESK</th>
            </tr>
          </thead>
          <tbody>
            {ladder.map((row) => (
              <tr key={row.bump}>
                <td
                  className="mono"
                  style={{ color: row.bump >= 0 ? "var(--green)" : "var(--red)" }}
                >
                  {row.bump >= 0 ? "+" : ""}{row.bump}%
                </td>
                {traders.slice(0, 7).map((t) => {
                  const val = row.traders[t.name] ?? 0;
                  return (
                    <td key={t.name} className="right mono" style={{ color: pnlColor(val) }}>
                      {formatDollar(val)}
                    </td>
                  );
                })}
                <td className="right mono bold" style={{ color: pnlColor(row.desk) }}>
                  {formatDollar(row.desk)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

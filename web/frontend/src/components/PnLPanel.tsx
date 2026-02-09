import type { DeskData, TraderData } from "../types/simulation";
import { formatPnlRaw, pnlColor, formatDollar, formatGreek } from "../utils/format";

interface PnLPanelProps {
  desk: DeskData | null;
  traders: TraderData[];
}

export function PnLPanel({ desk, traders }: PnLPanelProps) {
  if (!desk) {
    return (
      <div className="panel">
        <div className="panel-header">Day PnL</div>
        <div className="panel-body dimmed">Waiting for data...</div>
      </div>
    );
  }

  const maxPnl = Math.max(...traders.map((t) => Math.abs(t.pnl)), 1);

  return (
    <div className="panel">
      <div className="panel-header">Day PnL</div>
      <div className="panel-body">
        <div className="desk-summary">
          <div>
            <span className="label">Notional</span>
            <span className="value">{formatDollar(desk.total_mv)}</span>
          </div>
          <div>
            <span className="label">Day PnL</span>
            <span className="value big" style={{ color: pnlColor(desk.total_pnl) }}>
              {formatPnlRaw(desk.total_pnl)}
            </span>
          </div>
          <div>
            <span className="label">Desk Delta</span>
            <span className="value">{formatGreek(desk.delta)}</span>
          </div>
        </div>

        <table className="data-table">
          <thead>
            <tr>
              <th>Trader</th>
              <th>Book</th>
              <th className="right">Notional</th>
              <th className="right">Day PnL</th>
              <th className="right">Delta</th>
              <th className="right">Vega</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {traders.map((t) => {
              const barWidth = (Math.abs(t.pnl) / maxPnl) * 100;
              const isPositive = t.pnl >= 0;
              return (
                <tr key={t.name}>
                  <td style={{ color: "var(--text-bright)" }}>{t.name}</td>
                  <td style={{ color: "var(--text-dim)" }}>{t.strategy}</td>
                  <td className="right">{formatDollar(t.mv)}</td>
                  <td className="right" style={{ color: pnlColor(t.pnl) }}>
                    {formatPnlRaw(t.pnl)}
                  </td>
                  <td className="right">{formatGreek(t.delta)}</td>
                  <td className="right">{formatGreek(t.vega)}</td>
                  <td className="bar-cell">
                    <div
                      className={`pnl-bar ${isPositive ? "positive" : "negative"}`}
                      style={{ width: `${barWidth}%` }}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

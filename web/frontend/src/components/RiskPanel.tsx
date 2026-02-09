import type { DeskData, RiskData, TraderData } from "../types/simulation";
import { formatDollar, formatGreek, formatPercent } from "../utils/format";

interface RiskPanelProps {
  desk: DeskData | null;
  risk: RiskData | null;
  traders: TraderData[];
}

function LimitBar({ label, pct }: { label: string; pct: number }) {
  const cls = pct > 90 ? "breach" : pct > 70 ? "warn" : "ok";
  return (
    <div className="limit-row">
      <span className="limit-label">{label}</span>
      <div className="limit-bar-bg">
        <div
          className={`limit-bar-fill ${cls}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <span className="limit-pct">{pct.toFixed(0)}%</span>
    </div>
  );
}

export function RiskPanel({ desk, risk, traders }: RiskPanelProps) {
  if (!desk) {
    return (
      <div className="panel">
        <div className="panel-header">Risk Monitor</div>
        <div className="panel-body dimmed">Waiting for data...</div>
      </div>
    );
  }

  // Simulate limit utilization from desk greeks
  const deltaUtil = Math.min(Math.abs(desk.delta) / 500_000 * 100, 100);
  const vegaUtil = Math.min(Math.abs(desk.vega) / 200_000 * 100, 100);
  const gammaUtil = Math.min(Math.abs(desk.gamma) / 50_000 * 100, 100);
  const notionalUtil = Math.min(Math.abs(desk.total_mv) / 50_000_000 * 100, 100);

  return (
    <div className="panel">
      <div className="panel-header">Risk Monitor</div>
      <div className="panel-body">
        <div className="subsection">
          <div className="subsection-header">Desk Greeks</div>
          <table className="data-table compact">
            <tbody>
              <tr>
                <td>Delta ($)</td>
                <td className="right mono">{formatGreek(desk.delta)}</td>
              </tr>
              <tr>
                <td>Gamma ($)</td>
                <td className="right mono">{desk.gamma.toFixed(1)}</td>
              </tr>
              <tr>
                <td>Vega ($)</td>
                <td className="right mono">{formatGreek(desk.vega)}</td>
              </tr>
              <tr>
                <td>DV01</td>
                <td className="right mono">{formatDollar(desk.delta * 0.01)}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="subsection">
          <div className="subsection-header">Limit Utilization</div>
          <LimitBar label="Delta" pct={deltaUtil} />
          <LimitBar label="Vega" pct={vegaUtil} />
          <LimitBar label="Gamma" pct={gammaUtil} />
          <LimitBar label="Notional" pct={notionalUtil} />
        </div>

        {risk && (
          <div className="subsection">
            <div className="subsection-header">VaR / CVaR (95%/99%)</div>
            <table className="data-table compact">
              <tbody>
                {Object.entries(risk.var).map(([conf, levels]) => (
                  <tr key={`var-${conf}`}>
                    <td>VaR {conf}%</td>
                    <td className="right mono var-value">
                      {formatDollar(levels.var)}
                    </td>
                    <td>CVaR</td>
                    <td className="right mono">{formatDollar(levels.cvar)}</td>
                  </tr>
                ))}
                {risk.prob_loss != null && (
                  <tr>
                    <td>P(Loss)</td>
                    <td className="right mono" colSpan={3}>
                      {formatPercent(risk.prob_loss)}
                    </td>
                  </tr>
                )}
                {risk.worst != null && (
                  <tr>
                    <td>Worst</td>
                    <td className="right mono" style={{ color: "var(--red)" }}>
                      {formatDollar(risk.worst)}
                    </td>
                    <td>Best</td>
                    <td className="right mono" style={{ color: "var(--green)" }}>
                      {formatDollar(risk.best ?? 0)}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        <div className="subsection">
          <div className="subsection-header">Delta Exposure by Book</div>
          <div className="delta-heatmap">
            {traders.map((t) => {
              const abs = Math.abs(t.delta);
              const intensity = Math.min(abs / 100_000, 1);
              const bg =
                t.delta >= 0
                  ? `rgba(0,200,100,${intensity * 0.5})`
                  : `rgba(200,50,50,${intensity * 0.5})`;
              return (
                <div key={t.name} className="heatmap-cell" style={{ background: bg }}>
                  <span className="heatmap-label">{t.name}</span>
                  <span className="heatmap-value mono">{formatGreek(t.delta)}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

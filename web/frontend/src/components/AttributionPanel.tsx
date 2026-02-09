import type { Attribution } from "../types/simulation";
import { formatDollar, pnlColor } from "../utils/format";

interface AttributionPanelProps {
  attribution: Attribution | null;
}

const COMPONENTS: { key: keyof Attribution; label: string }[] = [
  { key: "delta_pnl", label: "Delta" },
  { key: "gamma_pnl", label: "Gamma" },
  { key: "vega_pnl", label: "Vega" },
  { key: "theta_pnl", label: "Theta" },
  { key: "unexplained", label: "Unexplained" },
];

export function AttributionPanel({ attribution }: AttributionPanelProps) {
  if (!attribution) {
    return (
      <div className="panel">
        <div className="panel-header">P&L Attribution</div>
        <div className="panel-body dimmed">Waiting for data...</div>
      </div>
    );
  }

  const maxAbs = Math.max(
    ...COMPONENTS.map((c) => Math.abs(attribution[c.key] as number)),
    1
  );

  return (
    <div className="panel">
      <div className="panel-header">
        P&L Attribution
        <span
          className="panel-badge"
          style={{ color: pnlColor(attribution.total), background: "transparent", border: "none" }}
        >
          Tick: {formatDollar(attribution.total)}
        </span>
      </div>
      <div className="panel-body">
        <div className="attribution-grid">
          {COMPONENTS.map(({ key, label }) => {
            const val = attribution[key] as number;
            const pct = (Math.abs(val) / maxAbs) * 100;
            const isPos = val >= 0;
            return (
              <div key={key} className="attribution-row">
                <span className="attribution-label">{label}</span>
                <div className="attribution-bar-container">
                  {isPos ? (
                    <>
                      <div className="attribution-bar-space" />
                      <div
                        className="attribution-bar positive"
                        style={{ width: `${pct / 2}%` }}
                      />
                    </>
                  ) : (
                    <>
                      <div
                        className="attribution-bar negative"
                        style={{ width: `${pct / 2}%`, marginLeft: "auto" }}
                      />
                      <div className="attribution-bar-space" />
                    </>
                  )}
                </div>
                <span className="attribution-value" style={{ color: pnlColor(val) }}>
                  {val >= 0 ? "+" : ""}{formatDollar(val)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

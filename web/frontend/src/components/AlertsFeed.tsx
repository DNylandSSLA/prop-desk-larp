import type { AlertData } from "../types/simulation";
import { severityColor } from "../utils/format";

interface AlertsFeedProps {
  alerts: AlertData[];
}

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false });
  } catch {
    return "--:--:--";
  }
}

export function AlertsFeed({ alerts }: AlertsFeedProps) {
  return (
    <div className="panel alerts-panel">
      <div className="panel-header">
        Risk Alerts
        {alerts.length > 0 && (
          <span className="panel-badge alert-badge">{alerts.length}</span>
        )}
      </div>
      <div className="panel-body scrollable">
        {alerts.length === 0 ? (
          <div className="no-alerts">ALL CLEAR</div>
        ) : (
          <div className="alert-list">
            {alerts.map((a, i) => (
              <div
                key={i}
                className={`alert-item severity-${a.severity.toLowerCase()}`}
                style={{ borderLeftColor: severityColor(a.severity) }}
              >
                <span className="alert-time">{formatTime(a.timestamp)}</span>
                <span
                  className="alert-severity"
                  style={{ color: severityColor(a.severity) }}
                >
                  {a.severity}
                </span>
                <span className="alert-message">{a.rule}: {a.message}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

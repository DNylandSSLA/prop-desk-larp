import type { OrderData } from "../types/simulation";

interface OrderFeedProps {
  orders: OrderData[];
}

function formatTime(ts: string): string {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString("en-US", { hour12: false });
  } catch {
    return "--:--:--";
  }
}

export function OrderFeed({ orders }: OrderFeedProps) {
  return (
    <div className="panel">
      <div className="panel-header">
        Order Blotter
        {orders.length > 0 && (
          <span className="panel-badge">{orders.length}</span>
        )}
      </div>
      <div className="panel-body scrollable">
        {orders.length === 0 ? (
          <div className="dimmed">No recent orders</div>
        ) : (
          <table className="data-table compact">
            <thead>
              <tr>
                <th>Time</th>
                <th>Trader</th>
                <th>Instrument</th>
                <th>Side</th>
                <th>Type</th>
                <th className="right">Qty</th>
                <th>Status</th>
                <th className="right">Fill Px</th>
              </tr>
            </thead>
            <tbody>
              {orders.map((o, i) => {
                const sideColor =
                  o.side === "BUY" ? "var(--green)" : "var(--red)";
                const statusColor =
                  o.status === "FILLED"
                    ? "var(--green)"
                    : o.status === "REJECTED"
                      ? "var(--red)"
                      : "var(--yellow)";
                return (
                  <tr key={i} className={i === 0 ? "new-row" : ""}>
                    <td className="dimmed">{formatTime(o.timestamp)}</td>
                    <td style={{ color: "var(--text-bright)" }}>{o.trader}</td>
                    <td className="mono">{o.instrument}</td>
                    <td className="mono" style={{ color: sideColor }}>
                      {o.side}
                    </td>
                    <td className="dimmed">{o.type}</td>
                    <td className="right mono">{o.quantity.toLocaleString()}</td>
                    <td className="mono" style={{ color: statusColor }}>
                      {o.status}
                    </td>
                    <td className="right mono">
                      {o.fill_price ? `$${o.fill_price.toFixed(2)}` : "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

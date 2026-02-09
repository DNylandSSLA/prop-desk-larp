export function formatDollar(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) {
    return `${value < 0 ? "-" : ""}$${(abs / 1_000_000).toFixed(1)}M`;
  }
  if (abs >= 1_000) {
    return `${value < 0 ? "-" : ""}$${(abs / 1_000).toFixed(0)}K`;
  }
  return `$${value.toFixed(0)}`;
}

export function formatPnl(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${formatDollar(value)}`;
}

export function formatPnlRaw(value: number): string {
  const sign = value >= 0 ? "+" : "";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) {
    return `${sign}$${(value / 1_000_000).toFixed(1)}M`;
  }
  return `${sign}$${value.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function formatPrice(value: number): string {
  if (value >= 1) {
    return `$${value.toFixed(2)}`;
  }
  return value.toFixed(4);
}

export function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

export function formatNumber(value: number): string {
  return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

export function formatGreek(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${formatNumber(value)}`;
}

export function pnlColor(value: number): string {
  if (value > 0) return "var(--green)";
  if (value < 0) return "var(--red)";
  return "var(--text-dim)";
}

export function severityColor(severity: string): string {
  switch (severity) {
    case "CRITICAL":
      return "var(--red)";
    case "HIGH":
      return "var(--amber)";
    case "MEDIUM":
      return "var(--yellow)";
    default:
      return "var(--text-dim)";
  }
}

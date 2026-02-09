/**
 * Fetch live market prices at startup.
 *
 * - Equities + VIX + SOFR come from latest-prices.json (written at build time)
 * - FX rates come from Frankfurter API (CORS-friendly, fetched at runtime)
 * - Everything is optional — returns partial data on failure
 */

export interface PriceOverrides {
  /** Equity ticker → latest price */
  prices?: Record<string, number>;
  /** VIX level */
  vix?: number;
  /** SOFR rate (decimal, e.g. 0.053) */
  sofr?: number;
  /** FX pair key → rate  (e.g. "eurusd" → 1.08) */
  fx?: Record<string, number>;
  /** ISO timestamp of when prices were fetched */
  fetchedAt?: string;
}

/** Map from Frankfurter currency codes to our FX pair keys + conversion. */
const FX_MAP: { code: string; key: string; invert: boolean }[] = [
  { code: "EUR", key: "eurusd", invert: true },   // API gives USD/EUR, we want EUR/USD
  { code: "GBP", key: "gbpusd", invert: true },
  { code: "JPY", key: "usdjpy", invert: false },   // API gives USD/JPY directly
  { code: "AUD", key: "audusd", invert: true },
  { code: "CAD", key: "usdcad", invert: false },
  { code: "CHF", key: "usdchf", invert: false },
];

async function fetchWithTimeout(url: string, timeoutMs: number): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

async function fetchBuildTimePrices(): Promise<Partial<PriceOverrides>> {
  try {
    const resp = await fetchWithTimeout(
      `${import.meta.env.BASE_URL}latest-prices.json`,
      5000,
    );
    if (!resp.ok) return {};
    const data = await resp.json();
    return {
      prices: data.prices ?? undefined,
      vix: data.vix ?? undefined,
      sofr: data.sofr ?? undefined,
      fetchedAt: data.fetchedAt ?? undefined,
    };
  } catch {
    return {};
  }
}

async function fetchFxRates(): Promise<Record<string, number>> {
  try {
    const codes = FX_MAP.map((f) => f.code).join(",");
    const resp = await fetchWithTimeout(
      `https://api.frankfurter.dev/v1/latest?base=USD&symbols=${codes}`,
      5000,
    );
    if (!resp.ok) return {};
    const data = await resp.json();
    const rates: Record<string, number> = {};
    for (const { code, key, invert } of FX_MAP) {
      const raw = data.rates?.[code];
      if (raw != null) {
        rates[key] = invert ? +(1 / raw).toFixed(6) : +raw.toFixed(6);
      }
    }
    return rates;
  } catch {
    return {};
  }
}

/**
 * Fetch live prices from all available sources.
 * Never throws — returns whatever data it can get.
 */
export async function fetchLivePrices(): Promise<PriceOverrides> {
  const [buildTime, fx] = await Promise.all([
    fetchBuildTimePrices(),
    fetchFxRates(),
  ]);

  return {
    ...buildTime,
    fx: Object.keys(fx).length > 0 ? fx : undefined,
  };
}

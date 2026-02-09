#!/usr/bin/env python3
"""Fetch latest equity prices, VIX, and SOFR for the web dashboard.

Writes web/frontend/public/latest-prices.json with real market data.
Designed to run in GitHub Actions before `npm run build`, but works locally too.
"""

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Ticker list — must match EQUITY_TICKERS in seed-data.ts
# ---------------------------------------------------------------------------
EQUITY_TICKERS = [
    "AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN", "META", "SPY",
    "AVGO", "CRM", "ORCL", "ADBE", "AMD", "INTC", "NFLX", "CSCO", "TXN",
    "QCOM", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SNPS", "CDNS", "PANW",
    "NOW", "SHOP", "SNOW", "DDOG", "NET", "CRWD", "ZS", "TEAM", "WDAY",
    "DASH", "RBLX", "U", "TTD", "ROKU", "PINS", "SNAP", "SPOT",
    "JPM", "GS", "BAC", "MS", "V", "MA", "BRK-B", "C", "WFC", "AXP",
    "SCHW", "BLK", "ICE", "CME", "SPGI", "MCO", "COF", "USB",
    "JNJ", "UNH", "PFE", "LLY", "ABBV", "MRK", "TMO", "ABT", "AMGN",
    "BMY", "GILD", "ISRG", "MDT", "SYK", "REGN", "VRTX", "MRNA", "BIIB",
    "WMT", "COST", "KO", "PEP", "MCD", "NKE", "DIS", "SBUX", "TGT",
    "LOW", "HD", "TJX", "ROST", "YUM", "CMG", "ABNB", "BKNG",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY",
    "CAT", "BA", "GE", "HON", "LMT", "RTX", "DE", "UNP", "UPS", "FDX",
    "WM", "EMR", "ITW", "GD", "NOC",
    "LIN", "APD", "FCX", "NEM", "NUE", "STLD",
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "WBD",
    "PLD", "AMT", "CCI", "EQIX", "SPG", "O",
    "PYPL", "COIN", "PLTR", "UBER", "LYFT", "RIVN", "LCID", "SOFI",
    "HOOD", "AFRM", "MELI", "SE", "NU",
    "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY",
    "XLB", "XLU", "XLRE", "GLD", "SLV", "TLT", "HYG", "LQD", "EEM", "EFA",
    "VWO", "ARKK", "SOXX", "SMH", "KWEB", "FXI", "VNQ", "IBIT",
]

OUT_PATH = Path(__file__).resolve().parent.parent / "web" / "frontend" / "public" / "latest-prices.json"


def fetch_equities_and_vix() -> tuple[dict[str, float], float | None]:
    """Fetch latest closing prices for all equities + ^VIX via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("WARNING: yfinance not installed — skipping equity fetch", file=sys.stderr)
        return {}, None

    all_tickers = EQUITY_TICKERS + ["^VIX"]
    print(f"Fetching {len(all_tickers)} tickers via yfinance...")

    df = yf.download(all_tickers, period="5d", auto_adjust=True, progress=False)
    if df.empty:
        print("WARNING: yfinance returned empty dataframe", file=sys.stderr)
        return {}, None

    prices: dict[str, float] = {}
    vix: float | None = None

    close = df["Close"] if "Close" in df.columns else df
    for ticker in all_tickers:
        try:
            col = close[ticker].dropna()
            if col.empty:
                continue
            price = round(float(col.iloc[-1]), 2)
            if ticker == "^VIX":
                vix = price
            else:
                prices[ticker] = price
        except (KeyError, IndexError):
            print(f"  skip {ticker}: no data", file=sys.stderr)

    print(f"  Got {len(prices)} equity prices, VIX={'%.2f' % vix if vix else 'N/A'}")
    return prices, vix


def fetch_sofr() -> float | None:
    """Fetch latest SOFR rate from NY Fed API."""
    url = "https://markets.newyorkfed.org/api/rates/secured/sofr/last/1.json"
    print("Fetching SOFR from NY Fed...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bank-python-dashboard"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        rate_pct = float(data["refRates"][0]["percentRate"])
        sofr = round(rate_pct / 100.0, 6)  # convert percent to decimal
        print(f"  SOFR = {rate_pct}% ({sofr})")
        return sofr
    except Exception as e:
        print(f"WARNING: SOFR fetch failed: {e}", file=sys.stderr)
        return None


def main() -> None:
    prices, vix = fetch_equities_and_vix()
    sofr = fetch_sofr()

    result: dict = {"prices": prices, "fetchedAt": datetime.now(timezone.utc).isoformat()}
    if vix is not None:
        result["vix"] = vix
    if sofr is not None:
        result["sofr"] = sofr

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {OUT_PATH} ({len(prices)} prices)")


if __name__ == "__main__":
    main()

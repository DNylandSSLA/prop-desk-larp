"""
Risk Models — Stochastic volatility, jump diffusion, vol surface, P&L attribution.

Phase 1 of the prop desk platform build. These models feed into the Monte Carlo
engine (Heston/Merton as path generators) and provide risk analytics (vol surface,
P&L attribution) for the trading desk.

Classes:
    HestonProcess      — Stochastic volatility model (Euler-Maruyama, full truncation)
    MertonJumpDiffusion — GBM + compound Poisson jumps
    VolSurface          — Real implied vol surface from yfinance options chains
    PnLAttribution      — Taylor-expansion P&L decomposition
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from mc_engine_rs import simulate_heston, simulate_merton
    _HAS_MC_RS = True
except ImportError:
    _HAS_MC_RS = False

logger = logging.getLogger(__name__)


# ── Normal CDF / PDF helpers (no scipy) ────────────────────────────────

def _norm_cdf(x):
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ── HestonProcess ───────────────────────────────────────────────────────

class HestonProcess:
    """
    Heston stochastic volatility model.

    dS = mu*S*dt + sqrt(V)*S*dW_S
    dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
    corr(dW_S, dW_V) = rho

    Uses Euler-Maruyama discretization with full-truncation scheme
    (V is floored at 0 in diffusion terms). 50 time steps per day,
    fully vectorized over paths.
    """

    def __init__(self, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7, v0=0.04):
        """
        Parameters
        ----------
        kappa : float — mean-reversion speed of variance
        theta : float — long-run variance level
        xi    : float — vol-of-vol
        rho   : float — correlation between spot and variance Brownians
        v0    : float — initial variance level
        """
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0

    def simulate(self, S0, mu, n_paths, n_steps, dt, rng=None):
        """
        Simulate terminal asset prices under the Heston model.

        Parameters
        ----------
        S0      : float — initial spot price
        mu      : float — drift (annualized)
        n_paths : int — number of Monte Carlo paths
        n_steps : int — number of time steps (50 per day recommended)
        dt      : float — time step size (in years, e.g. 1/252/50)
        rng     : numpy Generator — random number generator

        Returns
        -------
        (S_terminal, V_terminal) : tuple of np.ndarray [n_paths]
        """
        if _HAS_MC_RS:
            seed = 42 if rng is None else int(rng.integers(0, 2**63))
            return simulate_heston(
                float(S0), float(mu), int(n_paths), int(n_steps), float(dt),
                self.kappa, self.theta, self.xi, self.rho, self.v0, seed,
            )

        if rng is None:
            rng = np.random.default_rng(42)

        sqrt_dt = np.sqrt(dt)

        # Initialize arrays
        S = np.full(n_paths, S0, dtype=np.float64)
        V = np.full(n_paths, self.v0, dtype=np.float64)

        # Precompute correlation structure
        # W_S = Z1
        # W_V = rho * Z1 + sqrt(1 - rho^2) * Z2
        rho_comp = np.sqrt(1.0 - self.rho ** 2)

        for _ in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            dW_S = Z1 * sqrt_dt
            dW_V = (self.rho * Z1 + rho_comp * Z2) * sqrt_dt

            # Full truncation: use max(V, 0) in diffusion terms
            V_pos = np.maximum(V, 0.0)
            sqrt_V = np.sqrt(V_pos)

            # Euler-Maruyama step for S
            S = S * np.exp(
                (mu - 0.5 * V_pos) * dt + sqrt_V * dW_S
            )

            # Euler-Maruyama step for V
            V = V + self.kappa * (self.theta - V) * dt + self.xi * sqrt_V * dW_V

        return S, np.maximum(V, 0.0)


# ── MertonJumpDiffusion ─────────────────────────────────────────────────

class MertonJumpDiffusion:
    """
    Merton jump-diffusion model: GBM + compound Poisson jumps.

    dS/S = (mu - lambda*m)*dt + sigma*dW + J*dN
    N ~ Poisson(lambda*dt), J ~ LogNormal(mu_j, sigma_j)
    m = E[e^J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1

    Vectorized: draw Poisson counts per path, then compound log-normal jumps.
    """

    def __init__(self, jump_intensity=1.0, jump_mean=-0.05, jump_vol=0.10):
        """
        Parameters
        ----------
        jump_intensity : float — expected number of jumps per year (lambda)
        jump_mean      : float — mean of log-jump size (mu_j)
        jump_vol       : float — volatility of log-jump size (sigma_j)
        """
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_vol = jump_vol

    def simulate(self, S0, mu, sigma, n_paths, dt, rng=None):
        """
        Simulate terminal asset prices under Merton jump-diffusion.

        Parameters
        ----------
        S0      : float — initial spot price
        mu      : float — annualized drift
        sigma   : float — annualized diffusion volatility
        n_paths : int — number of paths
        dt      : float — total time horizon (in years)
        rng     : numpy Generator

        Returns
        -------
        S_terminal : np.ndarray [n_paths]
        """
        if _HAS_MC_RS:
            seed = 42 if rng is None else int(rng.integers(0, 2**63))
            return simulate_merton(
                float(S0), float(mu), float(sigma), int(n_paths), float(dt),
                self.jump_intensity, self.jump_mean, self.jump_vol, seed,
            )

        if rng is None:
            rng = np.random.default_rng(42)

        # Compensator: m = E[e^J - 1]
        m = np.exp(self.jump_mean + 0.5 * self.jump_vol ** 2) - 1.0

        # GBM component
        Z = rng.standard_normal(n_paths)
        drift = (mu - 0.5 * sigma ** 2 - self.jump_intensity * m) * dt
        diffusion = sigma * np.sqrt(dt) * Z

        # Jump component: N ~ Poisson(lambda * dt)
        N = rng.poisson(self.jump_intensity * dt, n_paths)

        # Compound jumps: sum of N_i log-normal jumps per path
        # log(J_total) = sum_{k=1}^{N_i} (mu_j + sigma_j * Z_k)
        # For vectorized computation:
        max_jumps = int(N.max()) if N.max() > 0 else 0
        jump_log_sum = np.zeros(n_paths)

        if max_jumps > 0:
            # Draw all jump sizes at once
            all_jump_normals = rng.standard_normal((n_paths, max_jumps))
            all_jump_logs = self.jump_mean + self.jump_vol * all_jump_normals

            # Mask: only count up to N[i] jumps for path i
            mask = np.arange(max_jumps)[np.newaxis, :] < N[:, np.newaxis]
            jump_log_sum = np.sum(all_jump_logs * mask, axis=1)

        S_terminal = S0 * np.exp(drift + diffusion + jump_log_sum)

        return S_terminal


# ── VolSurface ──────────────────────────────────────────────────────────

class VolSurface:
    """
    Implied volatility surface built from yfinance options chains.

    Fetches all expirations, filters by volume, inverts Black-Scholes via
    bisection to get implied vol. Stores in MnTable, persists via Barbara.
    Provides bilinear interpolation for get_vol(moneyness, time_to_expiry).
    """

    def __init__(self, ticker, barbara=None, graph=None):
        self.ticker = ticker
        self.barbara = barbara
        self.graph = graph
        self._surface_data = []  # list of dicts
        self._moneyness_arr = None  # numpy array for vectorized lookup
        self._expiry_arr = None
        self._vol_arr = None
        self._table = None
        self._built = False

    def build(self):
        """
        Fetch options chains from yfinance and build the vol surface.

        Filters options by volume > 10, computes mid price from bid/ask,
        inverts BS via bisection to get implied vol.
        """
        import yfinance as yf
        from bank_python.mntable import Table

        yticker = yf.Ticker(self.ticker)

        try:
            expirations = yticker.options
        except Exception as e:
            logger.warning(f"Failed to fetch options for {self.ticker}: {e}")
            return

        if not expirations:
            logger.warning(f"No options expirations for {self.ticker}")
            return

        # Get current spot price
        info = yticker.info
        spot = info.get("regularMarketPrice") or info.get("previousClose", 100.0)

        now = datetime.now()
        surface_points = []

        for exp_str in expirations:
            try:
                chain = yticker.option_chain(exp_str)
            except Exception as e:
                logger.debug(f"Skipping expiration {exp_str}: {e}")
                continue

            # Parse expiry date
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            T = max((exp_date - now).days / 365.0, 1.0 / 365.0)
            expiry_days = max((exp_date - now).days, 1)

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    vol_val = row.get("volume", 0)
                    if vol_val is None or (isinstance(vol_val, (int, float)) and vol_val <= 10):
                        continue

                    bid = row.get("bid", 0.0) or 0.0
                    ask = row.get("ask", 0.0) or 0.0
                    if bid <= 0 or ask <= 0:
                        continue

                    mid = (bid + ask) / 2.0
                    strike = row["strike"]
                    moneyness = strike / spot

                    is_call = opt_type == "call"
                    iv = self._bs_implied_vol(mid, spot, strike, T, is_call)
                    if iv is None or iv <= 0.01:
                        continue

                    surface_points.append({
                        "strike": float(strike),
                        "expiry_days": int(expiry_days),
                        "moneyness": float(moneyness),
                        "implied_vol": float(iv),
                        "option_type": opt_type,
                        "ticker": self.ticker,
                    })

        self._surface_data = surface_points

        # Pre-extract numpy arrays for vectorized get_vol (100-1000x faster)
        if surface_points:
            self._moneyness_arr = np.array([p["moneyness"] for p in surface_points])
            self._expiry_arr = np.array([p["expiry_days"] for p in surface_points], dtype=np.float64)
            self._vol_arr = np.array([p["implied_vol"] for p in surface_points])

        # Build MnTable
        self._table = Table([
            ("strike", float),
            ("expiry_days", int),
            ("moneyness", float),
            ("implied_vol", float),
            ("option_type", str),
            ("ticker", str),
        ], name=f"vol_surface_{self.ticker}")

        if surface_points:
            self._table.extend(surface_points)
            self._table.create_index("moneyness")
            self._table.create_index("expiry_days")

        self._built = True

        # Persist to Barbara
        if self.barbara:
            date_key = now.strftime("%Y-%m-%d")
            bk = f"/Risk/vol_surface/{self.ticker}/{date_key}"
            self.barbara[bk] = {
                "ticker": self.ticker,
                "n_points": len(surface_points),
                "built_at": now.isoformat(),
                "surface_data": surface_points,
            }

        logger.info(f"VolSurface {self.ticker}: {len(surface_points)} points")

    def get_vol(self, moneyness, time_to_expiry):
        """
        Get implied vol via inverse-distance weighted interpolation.

        Uses vectorized numpy with argpartition for O(n) nearest-neighbor
        lookup instead of O(n log n) sort. Returns float or None.
        """
        if not self._surface_data:
            return None

        # Lazily build numpy arrays if _surface_data was set directly
        if self._vol_arr is None:
            self._moneyness_arr = np.array([p["moneyness"] for p in self._surface_data])
            self._expiry_arr = np.array([p["expiry_days"] for p in self._surface_data], dtype=np.float64)
            self._vol_arr = np.array([p["implied_vol"] for p in self._surface_data])

        expiry_days = time_to_expiry * 365.0

        # Vectorized distance computation (no Python loop)
        dm = np.abs(self._moneyness_arr - moneyness)
        dt = np.abs(self._expiry_arr - expiry_days)
        dist = (dm / max(moneyness, 0.01)) ** 2 + (dt / max(expiry_days, 1.0)) ** 2

        # O(n) argpartition instead of O(n log n) sort
        n_neighbors = min(4, len(dist))
        if n_neighbors >= len(dist):
            top_k_idx = np.arange(len(dist))
        else:
            top_k_idx = np.argpartition(dist, n_neighbors)[:n_neighbors]
        top_k_dist = dist[top_k_idx]
        top_k_vol = self._vol_arr[top_k_idx]

        weights = 1.0 / (top_k_dist + 1e-10)
        return float(np.sum(weights * top_k_vol) / np.sum(weights))

    @property
    def table(self):
        return self._table

    @property
    def built(self):
        return self._built

    @staticmethod
    def _bs_implied_vol(market_price, S, K, T, is_call):
        """
        Invert Black-Scholes via bisection to find implied vol.

        Parameters
        ----------
        market_price : float — observed option mid price
        S            : float — spot price
        K            : float — strike
        T            : float — time to expiry (years)
        is_call      : bool

        Returns
        -------
        float — implied vol, or None if bisection fails
        """
        lo, hi = 0.01, 5.0
        tol = 1e-6
        max_iter = 100

        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            price = VolSurface._bs_price(S, K, mid, T, is_call)

            if abs(price - market_price) < tol:
                return mid

            if price < market_price:
                lo = mid
            else:
                hi = mid

            if hi - lo < tol:
                return mid

        return (lo + hi) / 2.0

    @staticmethod
    def _bs_price(S, K, sigma, T, is_call):
        """Scalar Black-Scholes European option price (no rate for simplicity)."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(0, S - K) if is_call else max(0, K - S)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)

        if is_call:
            return S * nd1 - K * nd2
        else:
            return K * (1.0 - nd2) - S * (1.0 - nd1)


# ── PnLAttribution ─────────────────────────────────────────────────────

@dataclass
class PnLSnapshot:
    """Captured state of Greeks and prices at a point in time."""
    timestamp: str
    positions: list = field(default_factory=list)
    # Each position entry: {trader, instrument, price, delta, gamma, vega, theta, quantity}


class PnLAttribution:
    """
    Taylor-expansion P&L decomposition.

    P&L ~ delta*dS + 0.5*gamma*dS^2 + vega*d_sigma + theta*dt + residual

    Takes snapshots of Greeks/prices and attributes P&L between snapshots.
    """

    def __init__(self, barbara=None):
        self.barbara = barbara
        self._prev_snapshot = None

    def snapshot(self, traders, state, compute_greeks_fn):
        """
        Capture current Greeks + prices for all trader positions.

        Parameters
        ----------
        traders          : list of Trader objects
        state            : dict — desk state
        compute_greeks_fn: callable — computes {"delta", "gamma", "vega"} for a Position

        Returns
        -------
        PnLSnapshot
        """
        now = datetime.now()
        positions = []

        for trader in traders:
            for pos in trader.book.positions:
                inst = pos.instrument
                greeks = compute_greeks_fn(pos)

                # Estimate theta from options (simplified: -0.5 * gamma * S^2 * sigma^2 / 252)
                theta = 0.0
                S = getattr(inst, 'spot_source', None)
                if S is not None:
                    S_val = S.value
                    sigma = getattr(inst, 'volatility', 0.2)
                    theta = -0.5 * greeks["gamma"] * S_val ** 2 * sigma ** 2 / 252.0

                positions.append({
                    "trader": trader.name,
                    "instrument": inst.name,
                    "price": inst.value,
                    "quantity": pos.quantity,
                    "delta": greeks["delta"],
                    "gamma": greeks["gamma"],
                    "vega": greeks["vega"],
                    "theta": theta,
                })

        snap = PnLSnapshot(
            timestamp=now.isoformat(),
            positions=positions,
        )

        # Store in Barbara
        if self.barbara:
            date_key = now.strftime("%Y-%m-%d")
            time_key = now.strftime("%H%M%S")
            bk = f"/Risk/pnl_attr/{date_key}/{time_key}"
            self.barbara[bk] = {
                "timestamp": snap.timestamp,
                "positions": positions,
            }

        return snap

    def attribute(self, prev_snapshot, curr_snapshot):
        """
        Attribute P&L between two snapshots using Taylor expansion.

        Parameters
        ----------
        prev_snapshot : PnLSnapshot
        curr_snapshot : PnLSnapshot

        Returns
        -------
        Table — MnTable with columns: trader, instrument, total_pnl, delta_pnl,
                gamma_pnl, vega_pnl, theta_pnl, residual_pnl
        """
        from bank_python.mntable import Table

        # Build lookup from prev snapshot
        prev_map = {}
        for p in prev_snapshot.positions:
            key = (p["trader"], p["instrument"])
            prev_map[key] = p

        rows = []
        for curr_pos in curr_snapshot.positions:
            key = (curr_pos["trader"], curr_pos["instrument"])
            prev_pos = prev_map.get(key)
            if prev_pos is None:
                continue

            dS = curr_pos["price"] - prev_pos["price"]
            qty = curr_pos["quantity"]

            # Use previous Greeks for attribution
            delta = prev_pos["delta"]
            gamma = prev_pos["gamma"]
            vega = prev_pos["vega"]
            theta = prev_pos["theta"]

            # Taylor expansion components
            delta_pnl = delta * dS
            gamma_pnl = 0.5 * gamma * dS ** 2

            # Estimate vol change (simplified: assume flat for now)
            d_sigma = 0.0
            vega_pnl = vega * d_sigma

            # Time decay (assume 1/252 year between snapshots)
            dt = 1.0 / 252.0
            theta_pnl = theta * dt

            # Total actual P&L
            total_pnl = (curr_pos["price"] - prev_pos["price"]) * qty
            # For options, the price is per-share; actual P&L needs multiplier
            from bank_python.dagger import Option
            # Just use raw price diff * qty as the total
            residual_pnl = total_pnl - (delta_pnl + gamma_pnl + vega_pnl + theta_pnl)

            rows.append({
                "trader": curr_pos["trader"],
                "instrument": curr_pos["instrument"],
                "total_pnl": float(total_pnl),
                "delta_pnl": float(delta_pnl),
                "gamma_pnl": float(gamma_pnl),
                "vega_pnl": float(vega_pnl),
                "theta_pnl": float(theta_pnl),
                "residual_pnl": float(residual_pnl),
            })

        result = Table([
            ("trader", str),
            ("instrument", str),
            ("total_pnl", float),
            ("delta_pnl", float),
            ("gamma_pnl", float),
            ("vega_pnl", float),
            ("theta_pnl", float),
            ("residual_pnl", float),
        ], name="pnl_attribution")

        if rows:
            result.extend(rows)

        return result

    def attribute_all(self, traders, state, compute_greeks_fn):
        """
        Take a new snapshot, compare to previous, return attribution table.

        On first call, stores snapshot and returns None (no previous to compare).

        Returns
        -------
        Table or None
        """
        curr = self.snapshot(traders, state, compute_greeks_fn)

        if self._prev_snapshot is None:
            self._prev_snapshot = curr
            return None

        result = self.attribute(self._prev_snapshot, curr)
        self._prev_snapshot = curr
        return result


# ── VolSurface visualization ────────────────────────────────────────────

def render_vol_surface(surface, console):
    """
    Render a vol surface summary as a Rich table.

    Parameters
    ----------
    surface : VolSurface
    console : rich.console.Console
    """
    from rich.panel import Panel
    from rich.table import Table as RichTable

    if not surface.built or not surface._surface_data:
        console.print(f"[yellow]No vol surface data for {surface.ticker}[/yellow]")
        return

    tbl = RichTable(
        title=f"Implied Vol Surface — {surface.ticker}",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Expiry (days)", justify="right", width=14)
    tbl.add_column("Strike", justify="right", width=10)
    tbl.add_column("Moneyness", justify="right", width=12)
    tbl.add_column("IV", justify="right", width=10)
    tbl.add_column("Type", width=6)

    # Group by expiry, show sample
    by_expiry = {}
    for pt in surface._surface_data:
        exp = pt["expiry_days"]
        if exp not in by_expiry:
            by_expiry[exp] = []
        by_expiry[exp].append(pt)

    shown = 0
    for exp_days in sorted(by_expiry.keys()):
        pts = sorted(by_expiry[exp_days], key=lambda x: x["moneyness"])
        # Show ATM region: moneyness 0.9 to 1.1
        atm_pts = [p for p in pts if 0.85 <= p["moneyness"] <= 1.15]
        if not atm_pts:
            atm_pts = pts[:5]

        for pt in atm_pts[:3]:
            tbl.add_row(
                str(pt["expiry_days"]),
                f"${pt['strike']:.0f}",
                f"{pt['moneyness']:.3f}",
                f"{pt['implied_vol']:.1%}",
                pt["option_type"],
            )
            shown += 1
            if shown >= 25:
                break

        if shown >= 25:
            break

    console.print(Panel(
        tbl,
        border_style="cyan",
        subtitle=f"{len(surface._surface_data)} total points",
    ))


def render_pnl_attribution(attr_table, console):
    """
    Render P&L attribution table.

    Parameters
    ----------
    attr_table : Table (MnTable)
    console    : rich.console.Console
    """
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    tbl = RichTable(
        title="P&L Attribution",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Trader", style="bold", width=8)
    tbl.add_column("Instrument", width=12)
    tbl.add_column("Total", justify="right", width=12)
    tbl.add_column("Delta", justify="right", width=12)
    tbl.add_column("Gamma", justify="right", width=12)
    tbl.add_column("Vega", justify="right", width=10)
    tbl.add_column("Theta", justify="right", width=10)
    tbl.add_column("Residual", justify="right", width=12)

    for row in attr_table:
        total_style = "green" if row["total_pnl"] >= 0 else "red"
        tbl.add_row(
            row["trader"],
            row["instrument"],
            Text(f"${row['total_pnl']:+,.0f}", style=total_style),
            f"${row['delta_pnl']:+,.0f}",
            f"${row['gamma_pnl']:+,.0f}",
            f"${row['vega_pnl']:+,.0f}",
            f"${row['theta_pnl']:+,.0f}",
            f"${row['residual_pnl']:+,.0f}",
        )

    console.print(Panel(tbl, border_style="cyan"))

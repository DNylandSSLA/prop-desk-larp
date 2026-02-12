#!/usr/bin/env python3
"""
Monte Carlo Simulation Engine for the Prop Trading Desk.

Correlated GBM path simulation (100K paths, numpy vectorized), portfolio
VaR/CVaR, stress testing, and Greeks validation via bump-and-reprice.

Architecture:
    HistoricalStore -> CovarianceBuilder -> Cholesky L
        -> PathSimulator -> correlated GBM terminal prices
        -> PortfolioRepricer -> P&L distribution
        -> RiskCalculator -> VaR/CVaR (95%/99%) + per-trader decomposition
    StressEngine -> predefined scenario shocks -> per-trader P&L
    GreeksValidator -> bump-and-reprice -> analytical vs numerical delta
    MCVisualizer -> Rich tables, ASCII histogram, comparison panels
    MonteCarloEngine -> orchestrator wiring everything together

Usage:
    python prop_desk.py mc [--paths 100000] [--horizon 1]
    python prop_desk.py stress
"""

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable
from rich.text import Text

from bank_python import (
    Bond,
    Equity,
    FXRate,
    MarketData,
    Option,
    Position,
)
from bank_python.risk_models import HestonProcess, MertonJumpDiffusion

try:
    from mc_engine_rs import simulate_gbm, bs_price_vec, bond_pv_vec
    _HAS_MC_RS = True
except ImportError:
    _HAS_MC_RS = False

logger = logging.getLogger(__name__)

SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


# ── MCConfig ─────────────────────────────────────────────────────────────

@dataclass
class MCConfig:
    """Configuration for the Monte Carlo simulation."""
    n_paths: int = 100_000
    horizon_days: int = 1
    confidence_levels: list = field(default_factory=lambda: [0.95, 0.99])
    historical_period: str = "1y"
    random_seed: int = 42


# ── CovarianceBuilder ────────────────────────────────────────────────────

class CovarianceBuilder:
    """
    Build covariance matrix from historical price data.

    Fetches aligned historical closes, computes log returns, builds
    covariance matrix and Cholesky decomposition for correlated simulation.
    """

    def build(self, mgr, tickers, period="1y"):
        """
        Build covariance structure from historical data.

        Parameters
        ----------
        mgr : MarketDataManager
            Manager with historical data access.
        tickers : list[str]
            Yahoo Finance tickers to include.
        period : str
            Historical period to fetch (e.g. "1y").

        Returns
        -------
        dict with keys:
            S0     : np.ndarray [n_assets] — current spot prices
            mu     : np.ndarray [n_assets] — annualized drift
            sigma  : np.ndarray [n_assets] — annualized volatility
            L      : np.ndarray [n_assets, n_assets] — Cholesky factor
            tickers: list[str] — aligned ticker order
        """
        # Fetch history for each ticker
        price_series = {}
        for ticker in tickers:
            view = mgr.history.query(ticker)
            rows = list(view)
            if len(rows) < 20:
                # Not enough history — try loading
                mgr.load_history(ticker, period)
                view = mgr.history.query(ticker)
                rows = list(view)
            if len(rows) < 2:
                logger.warning(f"Insufficient history for {ticker}, skipping")
                continue
            closes = [r["close"] for r in rows]
            price_series[ticker] = closes

        if not price_series:
            raise ValueError("No historical data available for any ticker")

        # Align series to common length (trim to shortest)
        valid_tickers = list(price_series.keys())
        min_len = min(len(price_series[t]) for t in valid_tickers)
        if min_len < 10:
            raise ValueError(f"Insufficient aligned history: only {min_len} points")

        # Build price matrix [n_days, n_assets]
        prices = np.column_stack([
            np.array(price_series[t][-min_len:], dtype=np.float64)
            for t in valid_tickers
        ])

        # Current prices (last row)
        S0 = prices[-1].copy()

        # Log returns [n_days-1, n_assets]
        log_prices = np.log(np.maximum(prices, 1e-10))
        returns = np.diff(log_prices, axis=0)

        # Annualized statistics
        mu = np.mean(returns, axis=0) * 252
        sigma = np.std(returns, axis=0, ddof=1) * np.sqrt(252)
        sigma = np.maximum(sigma, 1e-6)  # floor vol at tiny value

        # Covariance matrix (annualized)
        cov = np.cov(returns.T) * 252
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        # Regularize for numerical stability
        eps = 1e-8
        cov += eps * np.eye(len(valid_tickers))

        # Cholesky decomposition
        L = np.linalg.cholesky(cov)

        return {
            "S0": S0,
            "mu": mu,
            "sigma": sigma,
            "L": L,
            "tickers": valid_tickers,
            "returns_matrix": returns,
        }


# ── PathSimulator ────────────────────────────────────────────────────────

class PathSimulator:
    """
    Generate correlated GBM terminal prices using vectorized numpy.

    S_T = S0 * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*L@Z)
    """

    def simulate(self, S0, mu, sigma, L, config):
        """
        Simulate terminal asset prices.

        Parameters
        ----------
        S0    : np.ndarray [n_assets]
        mu    : np.ndarray [n_assets]
        sigma : np.ndarray [n_assets]
        L     : np.ndarray [n_assets, n_assets]
        config: MCConfig

        Returns
        -------
        S_terminal : np.ndarray [n_paths, n_assets]
        """
        if _HAS_MC_RS:
            return simulate_gbm(
                np.ascontiguousarray(S0, dtype=np.float64),
                np.ascontiguousarray(mu, dtype=np.float64),
                np.ascontiguousarray(sigma, dtype=np.float64),
                np.ascontiguousarray(L, dtype=np.float64),
                config.n_paths,
                config.horizon_days,
                config.random_seed,
            )

        rng = np.random.default_rng(config.random_seed)
        n_assets = len(S0)
        dt = config.horizon_days / 252.0

        # Independent standard normals [n_paths, n_assets]
        Z = rng.standard_normal((config.n_paths, n_assets))

        # Correlate: Z_corr = (L @ Z.T).T  ->  [n_paths, n_assets]
        Z_corr = (L @ Z.T).T

        # GBM terminal values
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_corr

        S_terminal = S0 * np.exp(drift + diffusion)

        return S_terminal


# ── Vectorized Black-Scholes helpers ─────────────────────────────────────

def _vec_norm_cdf(x):
    """Vectorized normal CDF using erf (no scipy needed)."""
    return 0.5 * (1.0 + _vec_erf(x / np.sqrt(2.0)))


def _vec_erf(x):
    """Vectorized error function via scipy C library (50-100x faster)."""
    from scipy.special import erf
    return erf(x)


def _vec_bs_price(S, K, sigma, T, is_call):
    """
    Vectorized Black-Scholes option price.

    Parameters
    ----------
    S      : np.ndarray [n_paths] — spot prices
    K      : float — strike
    sigma  : float — volatility
    T      : float — time to expiry
    is_call: bool

    Returns
    -------
    np.ndarray [n_paths] — option prices
    """
    if _HAS_MC_RS:
        return bs_price_vec(
            np.ascontiguousarray(S, dtype=np.float64),
            float(K), float(sigma), float(T), bool(is_call),
        )

    if T <= 0 or sigma <= 0:
        if is_call:
            return np.maximum(S - K, 0.0)
        else:
            return np.maximum(K - S, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = _vec_norm_cdf(d1)
    nd2 = _vec_norm_cdf(d2)

    if is_call:
        return S * nd1 - K * nd2
    else:
        return K * (1 - nd2) - S * (1 - nd1)


def _vec_bond_pv(rate, face, coupon_rate, maturity):
    """
    Vectorized bond present value.

    Parameters
    ----------
    rate       : np.ndarray [n_paths] — discount rates
    face       : float
    coupon_rate: float
    maturity   : int

    Returns
    -------
    np.ndarray [n_paths]
    """
    if _HAS_MC_RS:
        return bond_pv_vec(
            np.ascontiguousarray(rate, dtype=np.float64),
            float(face), float(coupon_rate), int(maturity),
        )

    coupon = face * coupon_rate
    t_arr = np.arange(1, maturity + 1, dtype=np.float64)
    # Discount factors: [n_paths, maturity]
    df = 1.0 / (1.0 + rate[:, np.newaxis]) ** t_arr[np.newaxis, :]
    pv = coupon * np.sum(df, axis=1) + face * df[:, -1]
    return pv


# ── PortfolioRepricer ────────────────────────────────────────────────────

class PortfolioRepricer:
    """
    Reprice all trader positions across simulated terminal prices.

    Uses vectorized BS for options, vectorized PV for bonds, and simple
    price * qty for equities/FX.
    """

    def reprice_all(self, traders, S_terminal, ticker_to_idx, state, config):
        """
        Compute P&L distribution for each trader and the desk.

        Parameters
        ----------
        traders      : list[Trader]
        S_terminal   : np.ndarray [n_paths, n_assets]
        ticker_to_idx: dict[str, int]
        state        : dict — desk state from setup_desk()
        config       : MCConfig

        Returns
        -------
        dict with:
            pnl_desk      : np.ndarray [n_paths]
            pnl_by_trader : dict[str, np.ndarray [n_paths]]
        """
        n_paths = S_terminal.shape[0]
        pnl_desk = np.zeros(n_paths)
        pnl_by_trader = {}

        # Map instruments to their current repriceable info
        sofr_md = state.get("sofr_md")
        dt = config.horizon_days / 252.0

        for trader in traders:
            pnl_trader = np.zeros(n_paths)

            for pos in trader.book.positions:
                inst = pos.instrument
                qty = pos.quantity
                current_value = inst.value * qty

                if isinstance(inst, Option):
                    # Find the ticker for the option's spot source
                    spot_name = inst.spot_source.name
                    ticker = self._find_ticker(spot_name, ticker_to_idx, state)
                    if ticker is None:
                        continue
                    idx = ticker_to_idx[ticker]
                    S_paths = S_terminal[:, idx]

                    # Remaining time to expiry after horizon
                    T_remaining = max(inst.time_to_expiry - dt, 0.0)

                    # Vectorized BS reprice
                    option_prices = _vec_bs_price(
                        S_paths, inst.strike, inst.volatility,
                        T_remaining, inst.is_call,
                    )
                    # Options have 100x multiplier in this desk
                    sim_value = option_prices * qty * 100.0
                    current_opt_value = inst.value * qty * 100.0
                    pnl_trader += sim_value - current_opt_value

                elif isinstance(inst, Bond):
                    # Simulate rates with small random walk
                    base_rate = sofr_md.value if sofr_md else 0.05
                    rng = np.random.default_rng(config.random_seed + hash(inst.name) % 10000)
                    rate_shock = rng.normal(0, 0.005, n_paths)  # ~50bp std dev
                    sim_rates = np.maximum(base_rate + rate_shock, 0.001)

                    bond_prices = _vec_bond_pv(
                        sim_rates, inst.face, inst.coupon_rate, inst.maturity,
                    )
                    sim_value = bond_prices * qty
                    pnl_trader += sim_value - current_value

                elif isinstance(inst, (Equity, FXRate)):
                    # Find ticker index
                    if isinstance(inst, Equity):
                        spot_name = inst.spot_source.name
                    else:
                        spot_name = inst.rate_source.name

                    ticker = self._find_ticker(spot_name, ticker_to_idx, state)
                    if ticker is None:
                        continue
                    idx = ticker_to_idx[ticker]
                    S_paths = S_terminal[:, idx]
                    sim_value = S_paths * qty
                    pnl_trader += sim_value - current_value

                else:
                    # MarketData or unknown — skip
                    continue

            pnl_by_trader[trader.name] = pnl_trader
            pnl_desk += pnl_trader

        return {
            "pnl_desk": pnl_desk,
            "pnl_by_trader": pnl_by_trader,
        }

    def _find_ticker(self, spot_name, ticker_to_idx, state):
        """Map a MarketData node name back to a Yahoo Finance ticker."""
        mgr = state.get("mgr")
        if mgr is None:
            return None
        for ticker, node in mgr.registered_tickers.items():
            if node.name == spot_name and ticker in ticker_to_idx:
                return ticker
        return None


# ── RiskCalculator ───────────────────────────────────────────────────────

class RiskCalculator:
    """Compute VaR, CVaR, and risk contribution from P&L distributions."""

    @staticmethod
    def compute_var_cvar(pnl_dist, confidence):
        """
        Compute Value-at-Risk and Conditional VaR (Expected Shortfall).

        Parameters
        ----------
        pnl_dist   : np.ndarray [n_paths] — P&L distribution
        confidence : float — e.g. 0.95 or 0.99

        Returns
        -------
        (VaR, CVaR) : tuple[float, float]
            Both positive numbers representing losses.
        """
        quantile = np.percentile(pnl_dist, (1 - confidence) * 100)
        var = -quantile
        tail = pnl_dist[pnl_dist <= quantile]
        if len(tail) == 0:
            cvar = var
        else:
            cvar = -np.mean(tail)
        return var, cvar

    @staticmethod
    def per_trader_risk(pnl_by_trader, confidence_levels):
        """
        Compute VaR/CVaR for each trader at each confidence level.

        Returns dict[trader_name, dict[conf, (VaR, CVaR)]]
        """
        results = {}
        for name, pnl in pnl_by_trader.items():
            results[name] = {}
            for conf in confidence_levels:
                var, cvar = RiskCalculator.compute_var_cvar(pnl, conf)
                results[name][conf] = (var, cvar)
        return results

    @staticmethod
    def risk_contribution(pnl_by_trader, pnl_desk, confidence=0.95):
        """
        Compute each trader's marginal contribution to desk VaR.

        Uses component VaR: weight_i * cov(pnl_i, pnl_desk) / VaR_desk.

        Returns dict[trader_name, float] — percentage contribution.
        """
        desk_var, _ = RiskCalculator.compute_var_cvar(pnl_desk, confidence)
        if desk_var <= 0:
            return {name: 0.0 for name in pnl_by_trader}

        contributions = {}
        for name, pnl_trader in pnl_by_trader.items():
            # Marginal contribution via covariance with desk
            cov_val = np.cov(pnl_trader, pnl_desk)[0, 1]
            desk_std = np.std(pnl_desk, ddof=1)
            if desk_std > 0:
                contrib = cov_val / (desk_std * desk_var) * desk_var
                contributions[name] = contrib
            else:
                contributions[name] = 0.0

        # Normalize to percentages
        total = sum(abs(v) for v in contributions.values())
        if total > 0:
            contributions = {k: v / total * 100 for k, v in contributions.items()}

        return contributions


# ── StressEngine ─────────────────────────────────────────────────────────

# Predefined stress scenarios
STRESS_SCENARIOS = {
    "2008 GFC": {
        "equity_shock": -0.40,
        "rate_shock_bp": 200,
        "fx_shock": 0.0,
        "vix_level": 80.0,
        "description": "2008 financial crisis replay",
    },
    "COVID": {
        "equity_shock": -0.35,
        "rate_shock_bp": -100,
        "fx_shock": 0.0,
        "vix_level": 65.0,
        "description": "March 2020 COVID crash",
    },
    "Rate Shock": {
        "equity_shock": 0.0,
        "rate_shock_bp": 200,
        "fx_shock": 0.0,
        "vix_level": 35.0,
        "description": "Sudden 200bp rate increase",
    },
    "Vol Spike": {
        "equity_shock": -0.15,
        "rate_shock_bp": 0,
        "fx_shock": 0.0,
        "vix_level": 50.0,
        "description": "Volatility spike with equity selloff",
    },
    "USD Rally": {
        "equity_shock": 0.0,
        "rate_shock_bp": 0,
        "fx_shock": -0.10,
        "vix_level": None,
        "description": "10% USD appreciation (FX pairs drop)",
    },
    "Tech Crash": {
        "equity_shock": -0.25,
        "rate_shock_bp": -50,
        "fx_shock": 0.0,
        "vix_level": 45.0,
        "description": "Tech sector selloff with flight to safety",
    },
    "Stagflation": {
        "equity_shock": -0.20,
        "rate_shock_bp": 300,
        "fx_shock": -0.05,
        "vix_level": 40.0,
        "description": "Rising rates + falling growth + strong dollar",
    },
    "Credit Crunch": {
        "equity_shock": -0.15,
        "rate_shock_bp": 150,
        "fx_shock": 0.0,
        "vix_level": 35.0,
        "description": "Widening spreads, HY selloff, bank stress",
    },
}


class StressEngine:
    """
    Run predefined and custom stress scenarios.

    Applies percentage shocks to current prices, reprices portfolio,
    reports per-trader P&L impact.
    """

    def run_scenario(self, traders, state, scenario_name):
        """
        Run a named predefined stress scenario.

        Returns dict with per-trader and desk P&L impact.
        """
        if scenario_name not in STRESS_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                             f"Available: {list(STRESS_SCENARIOS.keys())}")

        scenario = STRESS_SCENARIOS[scenario_name]
        return self._apply_scenario(traders, state, scenario, scenario_name)

    def run_all_scenarios(self, traders, state):
        """Run all predefined scenarios in parallel. Returns dict[scenario_name, results]."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(STRESS_SCENARIOS))) as pool:
            futures = {
                pool.submit(self.run_scenario, traders, state, name): name
                for name in STRESS_SCENARIOS
            }
            results = {}
            for future in futures:
                results[futures[future]] = future.result()
        return results

    def run_custom(self, traders, state, shocks):
        """
        Run a custom scenario.

        shocks: dict with optional keys: equity_shock, rate_shock_bp,
                fx_shock, vix_level
        """
        return self._apply_scenario(traders, state, shocks, "Custom")

    def _apply_scenario(self, traders, state, scenario, name):
        """
        Apply scenario shocks and compute P&L impact per trader.

        This does NOT modify the real MarketData nodes. Instead, it
        computes what prices would be under the scenario and reprices.
        """
        eq_shock = scenario.get("equity_shock", 0.0)
        rate_bp = scenario.get("rate_shock_bp", 0)
        fx_shock = scenario.get("fx_shock", 0.0)
        vix_level = scenario.get("vix_level")

        # Current state
        sofr_md = state.get("sofr_md")
        spots = state.get("spots", {})
        current_sofr = sofr_md.value if sofr_md else 0.05

        # Stressed rate
        stressed_rate = current_sofr + rate_bp / 10000.0

        results = {"scenario": name, "traders": {}, "desk_pnl": 0.0}

        for trader in traders:
            trader_pnl = 0.0

            for pos in trader.book.positions:
                inst = pos.instrument
                qty = pos.quantity
                current_mv = inst.value * qty

                if isinstance(inst, Option):
                    # Stressed spot
                    S_current = inst.spot_source.value
                    S_stressed = S_current * (1 + eq_shock)

                    # Stressed vol: if vix_level given, scale proportionally
                    stressed_vol = inst.volatility
                    if vix_level is not None:
                        vix_current = state.get("vix_md")
                        if vix_current and vix_current.value > 0:
                            vol_ratio = vix_level / vix_current.value
                            stressed_vol = inst.volatility * vol_ratio

                    # Reprice using scalar BS
                    stressed_price = self._bs_price(
                        S_stressed, inst.strike, stressed_vol,
                        inst.time_to_expiry, inst.is_call,
                    )
                    stressed_mv = stressed_price * qty * 100.0
                    current_opt_mv = inst.value * qty * 100.0
                    trader_pnl += stressed_mv - current_opt_mv

                elif isinstance(inst, Bond):
                    # Reprice bond at stressed rate
                    stressed_pv = self._bond_pv(
                        stressed_rate, inst.face, inst.coupon_rate, inst.maturity,
                    )
                    stressed_mv = stressed_pv * qty
                    trader_pnl += stressed_mv - current_mv

                elif isinstance(inst, FXRate):
                    # FX shock
                    S_stressed = inst.value * (1 + fx_shock)
                    stressed_mv = S_stressed * qty
                    trader_pnl += stressed_mv - current_mv

                elif isinstance(inst, Equity):
                    # Equity shock
                    S_stressed = inst.value * (1 + eq_shock)
                    stressed_mv = S_stressed * qty
                    trader_pnl += stressed_mv - current_mv

                else:
                    continue

            results["traders"][trader.name] = {
                "pnl": trader_pnl,
                "current_mv": trader.book.total_value,
            }
            results["desk_pnl"] += trader_pnl

        return results

    @staticmethod
    def _bs_price(S, K, sigma, T, is_call):
        """Scalar Black-Scholes price."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(0, S - K) if is_call else max(0, K - S)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
        if is_call:
            return S * nd1 - K * nd2
        else:
            return K * (1 - nd2) - S * (1 - nd1)

    @staticmethod
    def _bond_pv(rate, face, coupon_rate, maturity):
        """Scalar bond PV (vectorized over coupon periods)."""
        coupon = face * coupon_rate
        t_arr = np.arange(1, maturity + 1, dtype=np.float64)
        df = 1.0 / (1.0 + rate) ** t_arr
        return float(coupon * np.sum(df) + face * df[-1])


# ── GreeksValidator ──────────────────────────────────────────────────────

class GreeksValidator:
    """
    Validate analytical Greeks via bump-and-reprice.

    For each position with a spot-dependent underlying, bumps the price
    by epsilon, recomputes via the dependency graph, and compares the
    numerical delta to the analytical one from prop_desk.compute_greeks().
    """

    def __init__(self, epsilon=0.01):
        """epsilon: relative bump size (default 1%)."""
        self.epsilon = epsilon

    def validate(self, traders, state, compute_greeks_fn):
        """
        Run bump-and-reprice validation.

        Parameters
        ----------
        traders          : list[Trader]
        state            : dict — desk state
        compute_greeks_fn: callable — from prop_desk.compute_greeks

        Returns
        -------
        list[dict] with keys: trader, instrument, analytical_delta,
                              numerical_delta, discrepancy_pct
        """
        graph = state["graph"]
        results = []

        for trader in traders:
            for pos in trader.book.positions:
                inst = pos.instrument
                qty = pos.quantity

                # Find the spot source for bump
                spot_node = self._get_spot_node(inst)
                if spot_node is None:
                    continue

                S0 = spot_node.value
                if S0 <= 0:
                    continue

                bump = S0 * self.epsilon

                # Analytical delta
                greeks = compute_greeks_fn(pos)
                analytical_delta = greeks["delta"]

                # Bump up
                spot_node.set_price(S0 + bump)
                graph.recalculate(spot_node)
                V_up = inst.value * qty
                if isinstance(inst, Option):
                    V_up = inst.value * qty * 100.0

                # Bump down
                spot_node.set_price(S0 - bump)
                graph.recalculate(spot_node)
                V_down = inst.value * qty
                if isinstance(inst, Option):
                    V_down = inst.value * qty * 100.0

                # Restore
                spot_node.set_price(S0)
                graph.recalculate(spot_node)

                # Numerical delta: dV/dS
                raw_numerical = (V_up - V_down) / (2 * bump)

                # Match analytical convention: compute_greeks returns
                # dollar delta (S*qty) for linears, dV/dS*qty*100 for options.
                # For equities/FX/bonds, analytical delta = value * qty (notional),
                # so scale numerical dV/dS by S0 to get dollar delta.
                if isinstance(inst, Option):
                    numerical_delta = raw_numerical
                else:
                    numerical_delta = raw_numerical * S0

                # Discrepancy
                if abs(analytical_delta) > 1e-6:
                    disc_pct = abs(numerical_delta - analytical_delta) / abs(analytical_delta) * 100
                elif abs(numerical_delta) > 1e-6:
                    disc_pct = 100.0
                else:
                    disc_pct = 0.0

                results.append({
                    "trader": trader.name,
                    "instrument": inst.name,
                    "quantity": qty,
                    "analytical_delta": analytical_delta,
                    "numerical_delta": numerical_delta,
                    "discrepancy_pct": disc_pct,
                })

        return results

    @staticmethod
    def _get_spot_node(inst):
        """Get the MarketData spot source for an instrument."""
        if isinstance(inst, Option):
            return inst.spot_source
        elif isinstance(inst, Equity):
            return inst.spot_source
        elif isinstance(inst, FXRate):
            return inst.rate_source
        elif isinstance(inst, Bond):
            return inst.rate_source
        return None


# ── MCVisualizer ─────────────────────────────────────────────────────────

class MCVisualizer:
    """Rich-based visualization for Monte Carlo results."""

    def __init__(self, console=None):
        self.console = console or Console(width=110)

    def render_pnl_histogram(self, pnl_dist, var_levels=None):
        """
        Render an ASCII histogram of the P&L distribution.

        Parameters
        ----------
        pnl_dist   : np.ndarray [n_paths]
        var_levels : dict[float, float] — confidence -> VaR value (optional)
        """
        n_bins = 50
        counts, bin_edges = np.histogram(pnl_dist, bins=n_bins)
        max_count = max(counts) if max(counts) > 0 else 1
        height = 8  # number of block character levels

        # Scale counts to 0-height
        scaled = (counts / max_count * height).astype(int)

        # Build histogram string
        chars = list(SPARK_CHARS)
        hist_line = ""
        for i, s in enumerate(scaled):
            if s <= 0:
                hist_line += " "
            else:
                idx = min(s, len(chars)) - 1
                hist_line += chars[idx]

        # Mark VaR positions with '|'
        marked_hist = list(hist_line)
        if var_levels:
            for conf, var_val in var_levels.items():
                var_pos = -var_val  # VaR is a loss, mark on the left
                for i in range(len(bin_edges) - 1):
                    if bin_edges[i] <= var_pos <= bin_edges[i + 1]:
                        marked_hist[i] = "|"
                        break

        hist_str = "".join(marked_hist)

        # Labels
        lo = pnl_dist.min()
        hi = pnl_dist.max()
        mean = pnl_dist.mean()
        std = pnl_dist.std()

        lines = [
            f"P&L Distribution ({len(pnl_dist):,} paths)",
            f"[{hist_str}]",
            f"${lo:+,.0f}{'':>20s}${mean:+,.0f}{'':>20s}${hi:+,.0f}",
            f"Mean: ${mean:+,.0f}  Std: ${std:,.0f}",
        ]

        if var_levels:
            for conf, var_val in sorted(var_levels.items()):
                lines.append(f"  VaR {conf:.0%}: ${var_val:,.0f}  (| marker)")

        text = "\n".join(lines)
        self.console.print(Panel(text, title="P&L Histogram", border_style="cyan"))

    def render_var_table(self, desk_risk, trader_risk, contributions):
        """
        Render VaR/CVaR table for desk and per-trader.

        Parameters
        ----------
        desk_risk    : dict[confidence, (VaR, CVaR)]
        trader_risk  : dict[trader, dict[confidence, (VaR, CVaR)]]
        contributions: dict[trader, float] — % contribution to desk VaR
        """
        tbl = RichTable(
            title="Value-at-Risk / Expected Shortfall",
            expand=True,
            title_style="bold white",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Entity", style="bold", width=10)
        tbl.add_column("VaR 95%", justify="right", width=12)
        tbl.add_column("CVaR 95%", justify="right", width=12)
        tbl.add_column("VaR 99%", justify="right", width=12)
        tbl.add_column("CVaR 99%", justify="right", width=12)
        tbl.add_column("Contrib %", justify="right", width=10)

        # Desk row
        v95, c95 = desk_risk.get(0.95, (0, 0))
        v99, c99 = desk_risk.get(0.99, (0, 0))
        tbl.add_row(
            Text("DESK", style="bold"),
            f"${v95:,.0f}", f"${c95:,.0f}",
            f"${v99:,.0f}", f"${c99:,.0f}",
            "100%",
        )

        tbl.add_row("", "", "", "", "", "")

        # Per-trader
        for name in sorted(trader_risk.keys()):
            risk = trader_risk[name]
            v95, c95 = risk.get(0.95, (0, 0))
            v99, c99 = risk.get(0.99, (0, 0))
            contrib = contributions.get(name, 0)
            tbl.add_row(
                name,
                f"${v95:,.0f}", f"${c95:,.0f}",
                f"${v99:,.0f}", f"${c99:,.0f}",
                f"{contrib:.1f}%",
            )

        self.console.print(Panel(tbl, border_style="cyan"))

    def render_stress_table(self, all_results, traders):
        """
        Render stress test results as scenario x trader matrix.

        Parameters
        ----------
        all_results : dict[scenario_name, result_dict]
        traders     : list[Trader]
        """
        tbl = RichTable(
            title="Stress Test Results",
            expand=True,
            title_style="bold white",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Scenario", style="bold", width=14)
        for t in traders:
            tbl.add_column(t.name, justify="right", width=12)
        tbl.add_column("DESK", justify="right", width=14, style="bold")

        for scenario_name, result in all_results.items():
            row = [scenario_name]
            for t in traders:
                pnl = result["traders"].get(t.name, {}).get("pnl", 0)
                style = "green" if pnl >= 0 else "red"
                row.append(Text(f"${pnl:+,.0f}", style=style))

            desk_pnl = result["desk_pnl"]
            style = "green" if desk_pnl >= 0 else "red"
            row.append(Text(f"${desk_pnl:+,.0f}", style=f"bold {style}"))
            tbl.add_row(*row)

        self.console.print(Panel(tbl, border_style="red"))

    def render_greeks_table(self, comparisons):
        """
        Render analytical vs numerical Greeks comparison.

        Parameters
        ----------
        comparisons : list[dict] from GreeksValidator.validate()
        """
        tbl = RichTable(
            title="Greeks Validation (Bump-and-Reprice)",
            expand=True,
            title_style="bold white",
            show_header=True,
            header_style="bold cyan",
        )
        tbl.add_column("Trader", style="bold", width=8)
        tbl.add_column("Instrument", width=12)
        tbl.add_column("Qty", justify="right", width=7)
        tbl.add_column("Analytical \u0394", justify="right", width=14)
        tbl.add_column("Numerical \u0394", justify="right", width=14)
        tbl.add_column("Discrep %", justify="right", width=10)

        for c in comparisons:
            disc = c["discrepancy_pct"]
            if disc > 5.0:
                disc_style = "red bold"
            elif disc > 1.0:
                disc_style = "yellow"
            else:
                disc_style = "green"

            tbl.add_row(
                c["trader"],
                c["instrument"],
                f"{c['quantity']:,}",
                f"{c['analytical_delta']:+,.1f}",
                f"{c['numerical_delta']:+,.1f}",
                Text(f"{disc:.2f}%", style=disc_style),
            )

        self.console.print(Panel(tbl, border_style="cyan"))

    def render_full_report(self, mc_results):
        """Render the complete MC analysis report."""
        self.console.print(Panel(
            "[bold]MONTE CARLO RISK ANALYSIS[/bold]\n"
            f"[dim]{mc_results['config'].n_paths:,} paths, "
            f"{mc_results['config'].horizon_days}d horizon, "
            f"seed={mc_results['config'].random_seed}[/dim]",
            style="cyan",
            expand=False,
        ))

        # P&L histogram
        var_levels = {}
        for conf in mc_results["config"].confidence_levels:
            var, _ = mc_results["desk_risk"].get(conf, (0, 0))
            var_levels[conf] = var
        self.render_pnl_histogram(mc_results["pnl_desk"], var_levels)

        # VaR table
        self.render_var_table(
            mc_results["desk_risk"],
            mc_results["trader_risk"],
            mc_results["risk_contributions"],
        )

        # Stress tests
        if mc_results.get("stress_results"):
            self.render_stress_table(
                mc_results["stress_results"],
                mc_results["traders"],
            )

        # Greeks validation
        if mc_results.get("greeks_validation"):
            self.render_greeks_table(mc_results["greeks_validation"])

        # Summary stats
        pnl = mc_results["pnl_desk"]
        prob_loss = np.mean(pnl < 0) * 100
        self.console.print(Panel(
            f"P(Loss): {prob_loss:.1f}%    "
            f"Mean P&L: ${pnl.mean():+,.0f}    "
            f"Worst: ${pnl.min():+,.0f}    "
            f"Best: ${pnl.max():+,.0f}    "
            f"Assets: {mc_results.get('n_assets', '?')}",
            title="Summary",
            border_style="cyan",
        ))

    def render_var_summary(self, mc_results):
        """
        Render a compact VaR summary for the live dashboard risk panel.

        Returns a list of (label, value_str, style) tuples.
        """
        lines = []
        desk_risk = mc_results.get("desk_risk", {})
        for conf in sorted(desk_risk.keys()):
            var, cvar = desk_risk[conf]
            lines.append((f"VaR {conf:.0%}", f"${var:,.0f}", ""))
            lines.append((f"CVaR {conf:.0%}", f"${cvar:,.0f}", ""))

        pnl = mc_results.get("pnl_desk")
        if pnl is not None:
            prob_loss = np.mean(pnl < 0) * 100
            lines.append(("P(Loss)", f"{prob_loss:.1f}%", "yellow" if prob_loss > 50 else ""))

        return lines


# ── MonteCarloEngine (orchestrator) ──────────────────────────────────────

class MonteCarloEngine:
    """
    Orchestrator that wires together all MC components.

    Usage:
        engine = MonteCarloEngine(traders, state, db, config)
        results = engine.run_full_simulation()
    """

    def __init__(self, traders, state, db, config=None, compute_greeks_fn=None,
                 model="gbm"):
        """
        Parameters
        ----------
        model : str — "gbm" (default), "heston", or "merton"
        """
        self.traders = traders
        self.state = state
        self.db = db
        self.config = config or MCConfig()
        self.compute_greeks_fn = compute_greeks_fn
        self.model = model

        self.cov_builder = CovarianceBuilder()
        self.path_sim = PathSimulator()
        self.repricer = PortfolioRepricer()
        self.risk_calc = RiskCalculator()
        self.stress_engine = StressEngine()
        self.greeks_validator = GreeksValidator()

    def run_full_simulation(self):
        """
        Run the complete Monte Carlo analysis pipeline.

        1. Collect tickers from all trader positions
        2. Build covariance from historical data
        3. Simulate 100K correlated paths
        4. Reprice portfolio across paths
        5. Compute VaR/CVaR desk + per-trader
        6. Run stress tests
        7. Validate Greeks (bump-and-reprice)
        8. Store in Barbara
        9. Return results dict
        """
        t0 = time.time()
        mgr = self.state["mgr"]

        # 1. Collect tickers that have positions
        tickers = self._collect_tickers()
        if not tickers:
            raise ValueError("No tickers found in portfolio positions")

        logger.info(f"MC simulation: {len(tickers)} tickers, "
                    f"{self.config.n_paths} paths")

        # 2. Build covariance
        cov_data = self.cov_builder.build(
            mgr, tickers, period=self.config.historical_period,
        )
        valid_tickers = cov_data["tickers"]
        ticker_to_idx = {t: i for i, t in enumerate(valid_tickers)}

        # 3. Simulate paths (model-dependent)
        if self.model == "heston":
            S_terminal = self._simulate_heston(cov_data)
        elif self.model == "merton":
            S_terminal = self._simulate_merton(cov_data)
        else:
            S_terminal = self.path_sim.simulate(
                cov_data["S0"], cov_data["mu"], cov_data["sigma"],
                cov_data["L"], self.config,
            )

        # 4. Reprice
        reprice_result = self.repricer.reprice_all(
            self.traders, S_terminal, ticker_to_idx, self.state, self.config,
        )
        pnl_desk = reprice_result["pnl_desk"]
        pnl_by_trader = reprice_result["pnl_by_trader"]

        # 5. Compute risk metrics
        desk_risk = {}
        for conf in self.config.confidence_levels:
            desk_risk[conf] = self.risk_calc.compute_var_cvar(pnl_desk, conf)

        trader_risk = self.risk_calc.per_trader_risk(
            pnl_by_trader, self.config.confidence_levels,
        )
        risk_contributions = self.risk_calc.risk_contribution(
            pnl_by_trader, pnl_desk,
        )

        # 6. Stress tests
        stress_results = self.stress_engine.run_all_scenarios(
            self.traders, self.state,
        )

        # 7. Greeks validation
        greeks_validation = None
        if self.compute_greeks_fn:
            greeks_validation = self.greeks_validator.validate(
                self.traders, self.state, self.compute_greeks_fn,
            )

        elapsed = time.time() - t0

        results = {
            "config": self.config,
            "traders": self.traders,
            "pnl_desk": pnl_desk,
            "pnl_by_trader": pnl_by_trader,
            "desk_risk": desk_risk,
            "trader_risk": trader_risk,
            "risk_contributions": risk_contributions,
            "stress_results": stress_results,
            "greeks_validation": greeks_validation,
            "n_assets": len(valid_tickers),
            "tickers": valid_tickers,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        # 8. Store in Barbara
        self._store_results(results)

        logger.info(f"MC simulation complete in {elapsed:.1f}s")
        return results

    def run_stress_only(self):
        """Run stress tests without full MC simulation."""
        return self.stress_engine.run_all_scenarios(self.traders, self.state)

    def _simulate_heston(self, cov_data):
        """Simulate terminal prices using Heston stochastic vol, per-asset."""
        rng = np.random.default_rng(self.config.random_seed)
        n_assets = len(cov_data["S0"])
        dt_total = self.config.horizon_days / 252.0
        n_steps = 50 * self.config.horizon_days
        dt_step = dt_total / max(n_steps, 1)

        heston = HestonProcess()
        cols = []
        for i in range(n_assets):
            S_i, _ = heston.simulate(
                S0=cov_data["S0"][i],
                mu=cov_data["mu"][i],
                n_paths=self.config.n_paths,
                n_steps=n_steps,
                dt=dt_step,
                rng=rng,
            )
            cols.append(S_i)

        S_terminal = np.column_stack(cols)

        # Apply cross-asset correlation via Cholesky on the log-returns
        L = cov_data["L"]
        log_returns = np.log(S_terminal / cov_data["S0"])
        # Standardize, correlate, de-standardize
        stds = np.std(log_returns, axis=0, keepdims=True)
        stds = np.maximum(stds, 1e-10)
        Z = log_returns / stds
        Z_corr = (L @ Z.T).T
        corr_returns = Z_corr * stds
        S_terminal = cov_data["S0"] * np.exp(corr_returns)

        return S_terminal

    def _simulate_merton(self, cov_data):
        """Simulate terminal prices using Merton jump-diffusion, per-asset."""
        rng = np.random.default_rng(self.config.random_seed)
        n_assets = len(cov_data["S0"])
        dt = self.config.horizon_days / 252.0

        merton = MertonJumpDiffusion()
        cols = []
        for i in range(n_assets):
            S_i = merton.simulate(
                S0=cov_data["S0"][i],
                mu=cov_data["mu"][i],
                sigma=cov_data["sigma"][i],
                n_paths=self.config.n_paths,
                dt=dt,
                rng=rng,
            )
            cols.append(S_i)

        S_terminal = np.column_stack(cols)

        # Apply cross-asset correlation via Cholesky
        L = cov_data["L"]
        log_returns = np.log(S_terminal / cov_data["S0"])
        stds = np.std(log_returns, axis=0, keepdims=True)
        stds = np.maximum(stds, 1e-10)
        Z = log_returns / stds
        Z_corr = (L @ Z.T).T
        corr_returns = Z_corr * stds
        S_terminal = cov_data["S0"] * np.exp(corr_returns)

        return S_terminal

    def _collect_tickers(self):
        """Collect Yahoo Finance tickers referenced by trader positions."""
        mgr = self.state["mgr"]
        # Build reverse map: MarketData node name -> ticker
        node_to_ticker = {}
        for ticker, node in mgr.registered_tickers.items():
            node_to_ticker[node.name] = ticker

        tickers = set()
        for trader in self.traders:
            for pos in trader.book.positions:
                inst = pos.instrument
                if isinstance(inst, Option):
                    name = inst.spot_source.name
                elif isinstance(inst, Equity):
                    name = inst.spot_source.name
                elif isinstance(inst, FXRate):
                    name = inst.rate_source.name
                else:
                    continue
                if name in node_to_ticker:
                    tickers.add(node_to_ticker[name])

        return list(tickers)

    def _store_results(self, results):
        """Store MC results summary in Barbara."""
        now = datetime.now()
        summary = {
            "timestamp": results["timestamp"],
            "n_paths": self.config.n_paths,
            "horizon_days": self.config.horizon_days,
            "n_assets": results["n_assets"],
            "tickers": results["tickers"],
            "elapsed_seconds": results["elapsed_seconds"],
            "desk_risk": {
                str(conf): {"var": float(var), "cvar": float(cvar)}
                for conf, (var, cvar) in results["desk_risk"].items()
            },
            "trader_risk": {
                name: {
                    str(conf): {"var": float(var), "cvar": float(cvar)}
                    for conf, (var, cvar) in risk.items()
                }
                for name, risk in results["trader_risk"].items()
            },
            "risk_contributions": {
                name: float(pct) for name, pct in results["risk_contributions"].items()
            },
            "pnl_stats": {
                "mean": float(results["pnl_desk"].mean()),
                "std": float(results["pnl_desk"].std()),
                "min": float(results["pnl_desk"].min()),
                "max": float(results["pnl_desk"].max()),
                "prob_loss": float(np.mean(results["pnl_desk"] < 0)),
            },
        }

        bk = f"/Risk/mc/{now.strftime('%Y-%m-%d')}/{now.strftime('%H%M%S')}"
        self.db[bk] = summary

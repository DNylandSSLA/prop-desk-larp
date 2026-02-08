"""
Hedging Error Simulator — Based on Chapters 24, 26 of Taleb's
"Statistical Consequences of Fat Tails".

Demonstrates that delta hedging fails under fat tails. Simulates
hedging portfolio pi = -C + delta*S under different return distributions.

Expected: Gaussian errors tight (kurtosis ~3), Student-T(df=3) much wider
(kurtosis >>3), power law explosive.
"""

import math
from dataclasses import dataclass

import numpy as np

from bank_python.risk_models import _norm_cdf, _norm_pdf


@dataclass
class HedgingErrorResult:
    """Result of hedging error simulation for one distribution."""
    distribution: str
    mean_error: float
    std_error: float
    kurtosis: float
    var_95: float
    n_paths: int
    errors: np.ndarray


class HedgingErrorSimulator:
    """
    Simulate delta-hedging errors under different return distributions.

    At each rebalance step, compute BSM delta, track hedge P&L.
    Under fat tails, hedge errors explode because BSM delta assumes
    Gaussian returns.
    """

    def __init__(self, S0=100.0, K=100.0, sigma=0.20, T=1/12, n_steps=20):
        """
        Parameters
        ----------
        S0      : float — initial spot price
        K       : float — strike price
        sigma   : float — BSM vol for delta computation
        T       : float — time to expiry (years)
        n_steps : int — number of rebalance steps
        """
        self.S0 = S0
        self.K = K
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps

    def simulate(self, distribution="gaussian", n_paths=10000, df=3, alpha=3.0,
                 seed=None):
        """
        Simulate hedging errors under a given return distribution.

        Parameters
        ----------
        distribution : str — "gaussian", "student_t", or "power_law"
        n_paths      : int — number of MC paths
        df           : int — degrees of freedom for Student-T
        alpha        : float — tail index for power law
        seed         : int or None — random seed

        Returns
        -------
        HedgingErrorResult
        """
        rng = np.random.default_rng(seed)
        dt = self.T / self.n_steps
        sqrt_dt = math.sqrt(dt)

        # Generate returns for each path and step
        returns = self._generate_returns(
            distribution, n_paths, self.n_steps, dt, rng, df=df, alpha=alpha
        )

        # Simulate hedging
        errors = np.zeros(n_paths)
        S = np.full(n_paths, self.S0)
        hedge_pnl = np.zeros(n_paths)

        for step in range(self.n_steps):
            t_remaining = self.T - step * dt
            if t_remaining <= 0:
                break

            # Compute BSM delta (assuming Gaussian — this is the key flaw)
            delta = self._bsm_delta(S, self.K, self.sigma, t_remaining)

            # Price move
            S_new = S * np.exp(returns[:, step])

            # Option value change (true)
            # We track the hedging error: actual option P&L vs hedge P&L
            dS = S_new - S
            hedge_pnl += delta * dS

            S = S_new

        # Terminal option payoff
        option_payoff = np.maximum(S - self.K, 0.0)

        # Initial option price (BSM)
        C0 = self._bsm_call(self.S0, self.K, self.sigma, self.T)

        # Hedging error = option payoff - initial premium - hedge gains
        errors = option_payoff - C0 - hedge_pnl

        # Statistics
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        n = len(errors)
        m4 = float(np.mean((errors - mean_err) ** 4))
        m2 = float(np.mean((errors - mean_err) ** 2))
        kurtosis = m4 / max(m2 ** 2, 1e-15) if m2 > 0 else 3.0

        sorted_errors = np.sort(errors)
        var_95 = float(sorted_errors[int(0.05 * n)])

        return HedgingErrorResult(
            distribution=distribution,
            mean_error=mean_err,
            std_error=std_err,
            kurtosis=kurtosis,
            var_95=var_95,
            n_paths=n_paths,
            errors=errors,
        )

    def compare_all(self, n_paths=10000, seed=42):
        """
        Compare hedging errors across all three distributions.

        Returns
        -------
        list[HedgingErrorResult]
        """
        results = []
        for dist in ("gaussian", "student_t", "power_law"):
            results.append(self.simulate(
                distribution=dist, n_paths=n_paths, seed=seed,
            ))
        return results

    def _generate_returns(self, distribution, n_paths, n_steps, dt, rng,
                          df=3, alpha=3.0):
        """Generate return matrix [n_paths, n_steps] for given distribution."""
        sqrt_dt = math.sqrt(dt)

        if distribution == "gaussian":
            return rng.standard_normal((n_paths, n_steps)) * self.sigma * sqrt_dt

        elif distribution == "student_t":
            # Student-T via Z/sqrt(V/df) — no scipy needed
            # V = sum of df independent standard normals squared ~ chi2(df)
            Z = rng.standard_normal((n_paths, n_steps))
            # Generate chi-squared via sum of squared normals
            chi2_samples = np.zeros((n_paths, n_steps))
            for _ in range(df):
                chi2_samples += rng.standard_normal((n_paths, n_steps)) ** 2
            V = chi2_samples / df
            T_samples = Z / np.sqrt(np.maximum(V, 1e-10))
            # Scale to match vol * sqrt(dt), adjust for T variance = df/(df-2)
            scale = self.sigma * sqrt_dt
            if df > 2:
                scale *= math.sqrt((df - 2.0) / df)
            return T_samples * scale

        elif distribution == "power_law":
            # Pareto inverse CDF: u^{-1/alpha} - 1, symmetrized
            U = rng.uniform(0.01, 1.0, (n_paths, n_steps))
            pareto = np.power(U, -1.0 / alpha) - 1.0
            signs = rng.choice([-1.0, 1.0], size=(n_paths, n_steps))
            # Scale to approximate same center volatility
            raw = pareto * signs
            # Normalize to target vol * sqrt(dt)
            raw_std = np.std(raw)
            if raw_std > 0:
                raw = raw * (self.sigma * sqrt_dt / raw_std)
            return raw

        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

    @staticmethod
    def _bsm_delta(S, K, sigma, T):
        """Vectorized BSM call delta."""
        S = np.asarray(S, dtype=np.float64)
        valid = (S > 0) & (T > 0) & (sigma > 0)
        delta = np.where(S > K, 1.0, 0.0)  # default: intrinsic delta

        if np.any(valid):
            sqrt_T = math.sqrt(T)
            d1 = np.where(
                valid,
                (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T),
                0.0,
            )
            # Vectorized norm CDF
            delta = np.where(valid, 0.5 * (1.0 + _erf_vec(d1 / math.sqrt(2.0))), delta)

        return delta

    @staticmethod
    def _bsm_call(S, K, sigma, T):
        """Scalar BSM call price."""
        if T <= 0 or sigma <= 0:
            return max(0, S - K)
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def _erf_vec(x):
    """Vectorized error function using numpy."""
    # numpy doesn't have erf, but we can use the approximation
    # or iterate with math.erf
    result = np.empty_like(x, dtype=np.float64)
    flat = x.ravel()
    out = result.ravel()
    for i in range(len(flat)):
        out[i] = math.erf(flat[i])
    return result

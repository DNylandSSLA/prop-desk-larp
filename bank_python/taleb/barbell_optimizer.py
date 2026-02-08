"""
Barbell Optimizer — VaR/CVaR-constrained portfolio construction.

Based on Chapter 30 of Taleb's "Statistical Consequences of Fat Tails".
Under max entropy, the optimal portfolio is a barbell: weight w in safe
numeraire, (1-w) in a risky basket. Replace Markowitz variance constraint
with VaR/CVaR constraints.

Key formulas (Proposition 30.1):
    B(eps) = phi(Phi^{-1}(eps)) / eps
    mu = (v_- + K*B(eps)) / (1 + B(eps))
    sigma = (K - v_-) / (eta(eps) * (1 + B(eps)))
"""

import math
from datetime import datetime

import numpy as np

from bank_python.optimizer import OptimalPortfolio, MeanVarianceOptimizer
from bank_python.risk_models import _norm_cdf, _norm_pdf


def _norm_ppf(p, tol=1e-8, max_iter=100):
    """
    Inverse standard normal CDF via bisection.

    Same pattern as IV inversion in risk_models.py — no scipy needed.

    Parameters
    ----------
    p : float — probability in (0, 1)

    Returns
    -------
    float — quantile value
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if abs(p - 0.5) < 1e-12:
        return 0.0

    lo, hi = -8.0, 8.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        cdf_mid = _norm_cdf(mid)
        if abs(cdf_mid - p) < tol:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return mid
    return (lo + hi) / 2.0


class BarbellOptimizer:
    """
    Barbell portfolio optimizer with VaR constraints.

    Splits capital between a safe numeraire (risk-free) and a risky basket.
    The safe weight is determined by the VaR constraint: at confidence (1-eps),
    the portfolio should not lose more than max_loss_pct of value.

    The risky basket is optimized via MV max-Sharpe within the risky allocation.

    Parameters
    ----------
    cov_data       : dict — from CovarianceBuilder.build()
    risk_free_rate : float — annualized risk-free rate
    epsilon        : float — VaR confidence tail probability (e.g. 0.05 for 95%)
    max_loss_pct   : float — maximum acceptable loss fraction (e.g. 0.10 for 10%)
    """

    def __init__(self, cov_data, risk_free_rate=0.05, epsilon=0.05, max_loss_pct=0.10):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.epsilon = epsilon
        self.max_loss_pct = max_loss_pct
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)
        self.L = cov_data["L"]
        self.Sigma = self.L @ self.L.T
        self.mu = cov_data["mu"]

    def optimize(self, risky_method="max_sharpe"):
        """
        Compute the barbell portfolio.

        1. Compute max-entropy parameters (Prop 30.1)
        2. Solve safe weight from VaR constraint
        3. Allocate within risky basket via MV max-Sharpe

        Parameters
        ----------
        risky_method : str — how to allocate within risky basket

        Returns
        -------
        OptimalPortfolio
        """
        eps = self.epsilon
        K = -self.max_loss_pct  # loss threshold (negative)

        # Max-entropy parameters (Prop 30.1)
        z_eps = _norm_ppf(eps)  # negative for small eps
        phi_z = _norm_pdf(z_eps)
        B_eps = phi_z / max(eps, 1e-10)

        # Risky asset parameters: use portfolio-level stats
        risky_mu = float(np.mean(self.mu))
        risky_sigma = float(np.sqrt(np.mean(np.diag(self.Sigma))))

        # VaR of risky basket at confidence 1-eps
        risky_var = risky_mu + z_eps * risky_sigma  # negative for losses

        # Safe weight from VaR constraint:
        # w_safe * rf + (1 - w_safe) * risky_var >= K
        # w_safe * (rf - risky_var) >= K - risky_var
        denom = self.rf - risky_var
        if abs(denom) < 1e-10:
            w_safe = 0.5
        else:
            w_safe = (K - risky_var) / denom

        # Clamp to reasonable range
        w_safe = max(0.05, min(0.95, w_safe))

        # Optimize within risky basket using MV
        mv = MeanVarianceOptimizer(self.cov_data, risk_free_rate=self.rf)
        risky_result = mv.optimize(long_only=True, risk_aversion=0.5)
        risky_weights = risky_result.weights

        # Combine: [w_safe, (1-w_safe) * risky_weights...]
        all_weights = np.zeros(self.n + 1)
        all_weights[0] = w_safe
        all_weights[1:] = (1.0 - w_safe) * risky_weights

        # Portfolio statistics
        risky_port_ret = float(self.mu @ risky_weights)
        risky_port_vol = float(np.sqrt(risky_weights @ self.Sigma @ risky_weights))

        port_ret = w_safe * self.rf + (1.0 - w_safe) * risky_port_ret
        port_vol = (1.0 - w_safe) * risky_port_vol
        sharpe = (port_ret - self.rf) / max(port_vol, 1e-10)

        all_tickers = ["SAFE"] + list(self.tickers)

        return OptimalPortfolio(
            weights=all_weights,
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe_ratio=sharpe,
            method="taleb_barbell",
            tickers=all_tickers,
            timestamp=datetime.now().isoformat(),
        )

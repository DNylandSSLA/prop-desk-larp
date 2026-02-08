"""
Riskfolio-Lib wrappers — CVaR optimization and Risk Budgeting.

Provides capabilities beyond the native optimizer: tail-risk minimization
and custom risk allocation across assets.
"""

import numpy as np
import pandas as pd
from datetime import datetime

import riskfolio as rp

from bank_python.optimizer import OptimalPortfolio


class RiskfolioCVaROptimizer:
    """
    CVaR (Conditional Value-at-Risk) optimization via Riskfolio-Lib.

    Minimizes the expected loss in the worst alpha-percentile of outcomes,
    producing portfolios with better tail-risk properties than MV.
    """

    def __init__(self, cov_data, risk_free_rate=0.05, **kwargs):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

    def optimize(self, alpha=0.05, **kwargs):
        """
        Minimize CVaR at the given alpha level.

        Parameters
        ----------
        alpha : float — tail probability (default 0.05 = 5% worst outcomes)

        Returns
        -------
        OptimalPortfolio
        """
        returns_matrix = self.cov_data.get("returns_matrix")
        if returns_matrix is None:
            raise ValueError("CVaR optimization requires returns_matrix in cov_data. "
                             "Use CovarianceBuilder.build() which now provides it.")

        returns_df = pd.DataFrame(returns_matrix, columns=self.tickers)

        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu="hist", method_cov="hist")
        port.alpha = alpha

        w = port.optimization(
            model="Classic",
            rm="CVaR",
            obj="MinRisk",
            rf=self.rf,
            hist=True,
        )

        if w is None or w.empty:
            raise RuntimeError("Riskfolio CVaR optimization failed to find a solution")

        weights = w.values.flatten()

        # Compute portfolio stats
        L = self.cov_data["L"]
        Sigma = L @ L.T
        mu = self.cov_data["mu"]

        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(weights @ Sigma @ weights))
        sharpe = (exp_ret - self.rf) / max(exp_vol, 1e-10)

        return OptimalPortfolio(
            weights=weights,
            expected_return=exp_ret,
            expected_vol=exp_vol,
            sharpe_ratio=sharpe,
            method="riskfolio_cvar",
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )


class RiskfolioRiskBudgeting:
    """
    Risk budgeting optimization via Riskfolio-Lib.

    Allocates portfolio weights so that each asset contributes a specified
    fraction of total portfolio risk. Equal budgets approximates risk parity.
    """

    def __init__(self, cov_data, risk_free_rate=0.05, **kwargs):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

    def optimize(self, risk_budgets=None, risk_measure="MV", **kwargs):
        """
        Run risk budgeting optimization.

        Parameters
        ----------
        risk_budgets : dict[str, float] or None — ticker -> budget fraction.
                       If None, uses equal budgets (risk parity).
        risk_measure : str — "MV", "CVaR", or "CDaR"

        Returns
        -------
        OptimalPortfolio
        """
        returns_matrix = self.cov_data.get("returns_matrix")
        if returns_matrix is None:
            raise ValueError("Risk budgeting requires returns_matrix in cov_data. "
                             "Use CovarianceBuilder.build() which now provides it.")

        returns_df = pd.DataFrame(returns_matrix, columns=self.tickers)

        port = rp.Portfolio(returns=returns_df)
        port.assets_stats(method_mu="hist", method_cov="hist")

        # Build budget vector — riskfolio expects a 2D column vector
        if risk_budgets is not None:
            b = np.array([risk_budgets.get(t, 1.0 / self.n) for t in self.tickers])
            b = b / b.sum()  # normalize
        else:
            b = np.ones(self.n) / self.n
        b = b.reshape(-1, 1)

        w = port.rp_optimization(
            model="Classic",
            rm=risk_measure,
            rf=self.rf,
            b=b,
            hist=True,
        )

        if w is None or w.empty:
            raise RuntimeError("Riskfolio risk budgeting optimization failed")

        weights = w.values.flatten()

        # Compute portfolio stats
        L = self.cov_data["L"]
        Sigma = L @ L.T
        mu = self.cov_data["mu"]

        exp_ret = float(mu @ weights)
        exp_vol = float(np.sqrt(weights @ Sigma @ weights))
        sharpe = (exp_ret - self.rf) / max(exp_vol, 1e-10)

        return OptimalPortfolio(
            weights=weights,
            expected_return=exp_ret,
            expected_vol=exp_vol,
            sharpe_ratio=sharpe,
            method=f"riskfolio_risk_budget_{risk_measure.lower()}",
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )

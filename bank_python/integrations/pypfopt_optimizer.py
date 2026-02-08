"""
PyPortfolioOpt wrappers — MeanVariance, HRP, and Black-Litterman.

All classes consume ``cov_data`` dicts from CovarianceBuilder.build() and
return ``OptimalPortfolio`` dataclasses, so they slot into the existing
prop desk infrastructure without changes.
"""

import numpy as np
import pandas as pd
from datetime import datetime

from pypfopt import (
    EfficientFrontier,
    HRPOpt,
    BlackLittermanModel as BLModel,
    expected_returns,
    risk_models,
)

from bank_python.optimizer import OptimalPortfolio


class PyPfOptMeanVariance:
    """
    Mean-variance optimization via PyPortfolioOpt's EfficientFrontier.

    Supports max_sharpe, min_volatility, and target_return objectives.
    """

    def __init__(self, cov_data, risk_free_rate=0.05):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

        # Build pandas structures from cov_data
        self.mu_series = pd.Series(cov_data["mu"], index=self.tickers)
        L = cov_data["L"]
        cov_matrix = L @ L.T
        self.cov_df = pd.DataFrame(cov_matrix, index=self.tickers, columns=self.tickers)

    def optimize(self, objective="max_sharpe", target_return=None, long_only=True, **kwargs):
        """
        Run mean-variance optimization.

        Parameters
        ----------
        objective     : str — "max_sharpe", "min_volatility", or "target_return"
        target_return : float — required when objective="target_return"
        long_only     : bool — if True, weight_bounds=(0, 1)

        Returns
        -------
        OptimalPortfolio
        """
        bounds = (0, 1) if long_only else (-1, 1)
        ef = EfficientFrontier(self.mu_series, self.cov_df, weight_bounds=bounds)

        if objective == "max_sharpe":
            ef.max_sharpe(risk_free_rate=self.rf)
            method = "pypfopt_max_sharpe"
        elif objective == "min_volatility":
            ef.min_volatility()
            method = "pypfopt_min_volatility"
        elif objective == "target_return" and target_return is not None:
            ef.efficient_return(target_return)
            method = "pypfopt_target_return"
        else:
            ef.max_sharpe(risk_free_rate=self.rf)
            method = "pypfopt_max_sharpe"

        cleaned = ef.clean_weights()
        weights = np.array([cleaned.get(t, 0.0) for t in self.tickers])

        perf = ef.portfolio_performance(risk_free_rate=self.rf)
        exp_ret, exp_vol, sharpe = perf

        return OptimalPortfolio(
            weights=weights,
            expected_return=exp_ret,
            expected_vol=exp_vol,
            sharpe_ratio=sharpe,
            method=method,
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )


class PyPfOptHRP:
    """
    Hierarchical Risk Parity via PyPortfolioOpt's HRPOpt.

    Uses a dendrogram-based clustering approach that does not require
    covariance matrix inversion, making it robust to estimation error.
    """

    def __init__(self, cov_data, risk_free_rate=0.05):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

    def optimize(self, **kwargs):
        """
        Run HRP optimization.

        Returns
        -------
        OptimalPortfolio
        """
        returns_matrix = self.cov_data.get("returns_matrix")
        if returns_matrix is None:
            raise ValueError("HRP requires returns_matrix in cov_data. "
                             "Use CovarianceBuilder.build() which now provides it.")

        returns_df = pd.DataFrame(returns_matrix, columns=self.tickers)
        hrp = HRPOpt(returns_df)
        hrp.optimize()
        cleaned = hrp.clean_weights()

        weights = np.array([cleaned.get(t, 0.0) for t in self.tickers])

        # Compute portfolio stats from weights
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
            method="pypfopt_hrp",
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )


class PyPfOptBlackLitterman:
    """
    Black-Litterman model via PyPortfolioOpt.

    Same add_view() API as the native BlackLittermanModel, but uses
    PyPortfolioOpt's BL implementation internally.
    """

    def __init__(self, cov_data, market_cap_weights=None, delta=2.5, tau=0.05,
                 risk_free_rate=0.05):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)
        self.delta = delta
        self.tau = tau

        L = cov_data["L"]
        self.cov_matrix = L @ L.T
        self.cov_df = pd.DataFrame(self.cov_matrix, index=self.tickers, columns=self.tickers)

        if market_cap_weights is not None:
            self.w_mkt = np.asarray(market_cap_weights, dtype=np.float64)
        else:
            self.w_mkt = np.ones(self.n) / self.n

        self._views_P = []
        self._views_Q = []
        self._views_omega = []

    def add_view(self, view_type, assets, returns, confidence):
        """
        Add a view (same API as the native BlackLittermanModel).

        Parameters
        ----------
        view_type  : str — "absolute" or "relative"
        assets     : list[str]
        returns    : float
        confidence : float (0 to 1)
        """
        P = np.zeros(self.n)

        if view_type == "absolute":
            for asset in assets:
                if asset in self.tickers:
                    idx = self.tickers.index(asset)
                    P[idx] = 1.0
        elif view_type == "relative":
            if len(assets) >= 2:
                idx1 = self.tickers.index(assets[0]) if assets[0] in self.tickers else None
                idx2 = self.tickers.index(assets[1]) if assets[1] in self.tickers else None
                if idx1 is not None and idx2 is not None:
                    P[idx1] = 1.0
                    P[idx2] = -1.0

        omega = (1.0 - confidence) * self.tau * (P @ self.cov_matrix @ P) + 1e-10

        self._views_P.append(P)
        self._views_Q.append(returns)
        self._views_omega.append(omega)

    def optimize(self, long_only=True, **kwargs):
        """
        Compute BL posterior and optimize.

        Returns
        -------
        OptimalPortfolio
        """
        pi = self.delta * self.cov_matrix @ self.w_mkt

        if self._views_P:
            P = np.array(self._views_P)
            Q = np.array(self._views_Q)
            omega = np.diag(self._views_omega)

            bl = BLModel(
                self.cov_df,
                pi=pd.Series(pi, index=self.tickers),
                P=P, Q=Q,
                omega=omega,
            )
            bl_returns = bl.bl_returns()
        else:
            bl_returns = pd.Series(pi, index=self.tickers)

        bounds = (0, 1) if long_only else (-1, 1)
        ef = EfficientFrontier(bl_returns, self.cov_df, weight_bounds=bounds)
        try:
            ef.max_sharpe(risk_free_rate=self.rf)
        except ValueError:
            # All returns below risk-free rate — fall back to min volatility
            ef = EfficientFrontier(bl_returns, self.cov_df, weight_bounds=bounds)
            ef.min_volatility()
        cleaned = ef.clean_weights()

        weights = np.array([cleaned.get(t, 0.0) for t in self.tickers])
        perf = ef.portfolio_performance(risk_free_rate=self.rf)
        exp_ret, exp_vol, sharpe = perf

        return OptimalPortfolio(
            weights=weights,
            expected_return=exp_ret,
            expected_vol=exp_vol,
            sharpe_ratio=sharpe,
            method="pypfopt_black_litterman",
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )

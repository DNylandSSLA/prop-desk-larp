"""
Correlation Fragility Analyzer — Based on Chapter 29 of Taleb's
"Statistical Consequences of Fat Tails".

Shows that MV portfolios are hypersensitive to correlation estimates
that swing wildly in practice. Small perturbations in the correlation
matrix produce large swings in optimal weights.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from bank_python.optimizer import MeanVarianceOptimizer


@dataclass
class CorrelationFragilityResult:
    """Result of correlation fragility analysis."""
    base_weights: np.ndarray
    tickers: list
    perturbation_deltas: list
    perturbed_weights: list  # list of np.ndarray per delta
    weight_sensitivities: np.ndarray  # per-asset sensitivity
    rolling_corr_mean: float
    rolling_corr_std: float
    delta_statistic: float
    p_value: float


class CorrelationFragilityAnalyzer:
    """
    Analyze sensitivity of MV portfolios to correlation estimation error.

    1. Compute base MV weights from observed covariance
    2. Perturb off-diagonal correlation by delta, re-optimize at each
    3. Compute weight sensitivity: (w(rho+d) - w(rho-d)) / (2d)
    4. Rolling correlation stats from returns_matrix
    5. Delta_{n,m} statistic for rolling correlation variance
    """

    def __init__(self, cov_data, risk_free_rate=0.05):
        """
        Parameters
        ----------
        cov_data       : dict — from CovarianceBuilder.build()
        risk_free_rate : float
        """
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)
        self.L = cov_data["L"]
        self.Sigma = self.L @ self.L.T
        self.mu = cov_data["mu"]
        self.returns_matrix = cov_data.get("returns_matrix")

    def analyze(self, perturbation_deltas=None, rolling_window=20):
        """
        Full correlation fragility analysis.

        Parameters
        ----------
        perturbation_deltas : list[float] or None — defaults to [-0.3, ..., +0.3]
        rolling_window      : int — window for rolling correlation

        Returns
        -------
        CorrelationFragilityResult
        """
        if perturbation_deltas is None:
            perturbation_deltas = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]

        # Base optimization
        mv = MeanVarianceOptimizer(self.cov_data, risk_free_rate=self.rf)
        base_result = mv.optimize(long_only=True)
        base_weights = base_result.weights

        # Perturb correlation and re-optimize at each delta
        perturbed_weights = []
        for delta in perturbation_deltas:
            perturbed_Sigma = self._perturb_correlation(self.Sigma, delta)
            perturbed_cov_data = self._make_cov_data(perturbed_Sigma)
            try:
                mv_p = MeanVarianceOptimizer(perturbed_cov_data, risk_free_rate=self.rf)
                result_p = mv_p.optimize(long_only=True)
                perturbed_weights.append(result_p.weights)
            except Exception:
                perturbed_weights.append(base_weights.copy())

        # Weight sensitivity: (w(rho+d) - w(rho-d)) / (2d) using widest perturbation
        sensitivities = np.zeros(self.n)
        if len(perturbation_deltas) >= 2:
            d_max = max(abs(d) for d in perturbation_deltas if d != 0)
            # Find +d and -d indices
            pos_idx = None
            neg_idx = None
            for i, d in enumerate(perturbation_deltas):
                if abs(d - d_max) < 1e-10:
                    pos_idx = i
                if abs(d + d_max) < 1e-10:
                    neg_idx = i
            if pos_idx is not None and neg_idx is not None and d_max > 0:
                sensitivities = (perturbed_weights[pos_idx] - perturbed_weights[neg_idx]) / (2 * d_max)

        # Rolling correlation stats
        rolling_corr_mean = 0.0
        rolling_corr_std = 0.0
        delta_stat = 0.0
        p_value = 0.5

        if self.returns_matrix is not None and len(self.returns_matrix) > rolling_window:
            rolling_corrs = self._rolling_correlations(self.returns_matrix, rolling_window)
            if len(rolling_corrs) > 1:
                rolling_corr_mean = float(np.mean(rolling_corrs))
                rolling_corr_std = float(np.std(rolling_corrs))
                # Delta statistic: second moment of rolling correlations
                delta_stat = float(np.mean(rolling_corrs ** 2))
                # Approximate p-value via chi-squared: ndf under null
                ndf = len(rolling_corrs)
                # Under null of constant correlation, delta_stat * ndf ~ chi2(ndf)
                chi2_stat = delta_stat * ndf
                # Simple chi-squared p-value approximation
                p_value = self._chi2_pvalue_approx(chi2_stat, ndf)

        return CorrelationFragilityResult(
            base_weights=base_weights,
            tickers=self.tickers,
            perturbation_deltas=perturbation_deltas,
            perturbed_weights=perturbed_weights,
            weight_sensitivities=sensitivities,
            rolling_corr_mean=rolling_corr_mean,
            rolling_corr_std=rolling_corr_std,
            delta_statistic=delta_stat,
            p_value=p_value,
        )

    def _perturb_correlation(self, Sigma, delta):
        """
        Perturb off-diagonal correlations by delta.

        1. Extract correlation matrix from covariance
        2. Add delta to off-diagonal elements
        3. Clamp to [-1, 1]
        4. Project to nearest PSD via eigenvalue clipping
        5. Convert back to covariance
        """
        n = len(Sigma)
        stds = np.sqrt(np.diag(Sigma))
        stds_outer = np.outer(stds, stds)
        stds_outer = np.maximum(stds_outer, 1e-15)

        # Correlation matrix
        corr = Sigma / stds_outer

        # Perturb off-diagonal
        perturbed_corr = np.clip(corr + delta, -0.99, 0.99)
        np.fill_diagonal(perturbed_corr, 1.0)

        # Project to nearest PSD via eigenvalue clipping
        perturbed_corr = self._project_psd(perturbed_corr)

        # Convert back to covariance
        return perturbed_corr * stds_outer

    @staticmethod
    def _project_psd(matrix, min_eigenvalue=1e-6):
        """Project matrix to nearest PSD via eigenvalue clipping."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Ensure diagonal is 1 for correlation matrix
        d = np.sqrt(np.diag(result))
        d_outer = np.outer(d, d)
        d_outer = np.maximum(d_outer, 1e-15)
        return result / d_outer

    def _make_cov_data(self, Sigma):
        """Create a cov_data dict from a covariance matrix."""
        # Cholesky with regularization
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma_reg = Sigma + 1e-6 * np.eye(self.n)
            L = np.linalg.cholesky(Sigma_reg)

        return {
            "S0": self.cov_data.get("S0", {}),
            "mu": self.mu,
            "sigma": np.sqrt(np.diag(Sigma)),
            "L": L,
            "tickers": self.tickers,
        }

    @staticmethod
    def _rolling_correlations(returns_matrix, window):
        """Compute rolling average pairwise correlation."""
        n_obs, n_assets = returns_matrix.shape
        if n_assets < 2:
            return np.array([0.0])

        corr_values = []
        for start in range(0, n_obs - window + 1, max(1, window // 2)):
            end = start + window
            chunk = returns_matrix[start:end]
            if len(chunk) < window:
                break
            corr = np.corrcoef(chunk.T)
            # Average off-diagonal
            mask = ~np.eye(n_assets, dtype=bool)
            avg_corr = np.mean(corr[mask])
            corr_values.append(avg_corr)

        return np.array(corr_values) if corr_values else np.array([0.0])

    @staticmethod
    def _chi2_pvalue_approx(x, k):
        """
        Approximate chi-squared p-value using Wilson-Hilferty transformation.

        P(chi2 > x) ~ 1 - Phi(z) where z = ((x/k)^{1/3} - (1 - 2/(9k))) / sqrt(2/(9k))
        """
        if k <= 0:
            return 0.5
        from bank_python.risk_models import _norm_cdf
        a = 2.0 / (9.0 * k)
        z = ((x / k) ** (1.0 / 3.0) - (1.0 - a)) / math.sqrt(a)
        return max(0.0, min(1.0, 1.0 - _norm_cdf(z)))

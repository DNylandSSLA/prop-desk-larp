"""
Tail Diagnostics — Hill estimator, kappa metric, max-to-sum ratio.

Based on Chapters 4, 10, 27 of Taleb's "Statistical Consequences of Fat Tails".
These diagnostics determine how fat-tailed a return distribution is:
- Hill alpha < 4: variance may not exist; alpha < 2: mean may not exist
- Kappa deviates from sqrt(2/pi) for non-Gaussian tails
- Max-to-sum ratio converges to 0 for thin tails, stays large for fat tails
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TailDiagnosticsResult:
    """Result of tail analysis on a return series."""
    alpha: float
    alpha_se: float
    kappa: float
    kappa_gaussian: float
    max_to_sum: float
    n_observations: int
    k_tail: int


class TailDiagnostics:
    """
    Fat-tail diagnostics for return series.

    Provides Hill estimator for tail index, kappa metric for
    deviation from Gaussian, and max-to-sum ratio.
    """

    @staticmethod
    def hill_estimator(sorted_abs_returns, k=None):
        """
        Hill estimator for the tail index alpha.

        Parameters
        ----------
        sorted_abs_returns : array-like
            Absolute returns sorted in DESCENDING order.
        k : int or None
            Number of tail observations to use. If None, uses sqrt(n).

        Returns
        -------
        (alpha, se) : tuple[float, float]
            Estimated tail index and its standard error.
        """
        x = np.asarray(sorted_abs_returns, dtype=np.float64)
        n = len(x)
        if n < 10:
            return (float("inf"), float("inf"))

        if k is None:
            k = max(5, int(math.sqrt(n)))
        k = min(k, n - 1)

        # x should be sorted descending; x[0] is largest
        # Hill: alpha = k / sum_{i=0}^{k-1} ln(x[i] / x[k])
        threshold = x[k]
        if threshold <= 0:
            return (float("inf"), float("inf"))

        top_k = x[:k]
        positive = top_k[top_k > 0]
        log_sum = float(np.sum(np.log(positive / threshold)))

        if log_sum <= 0:
            return (float("inf"), float("inf"))

        alpha = k / log_sum
        se = alpha / math.sqrt(k)
        return (alpha, se)

    @staticmethod
    def kappa_metric(returns):
        """
        Kappa metric: E[|X - mu|] / std(X).

        For a Gaussian, kappa = sqrt(2/pi) ~ 0.7979.
        Fat tails produce kappa < sqrt(2/pi).

        Parameters
        ----------
        returns : array-like

        Returns
        -------
        float
        """
        r = np.asarray(returns, dtype=np.float64)
        if len(r) < 3:
            return float("nan")
        mu = np.mean(r)
        mad = np.mean(np.abs(r - mu))
        std = np.std(r, ddof=0)
        if std < 1e-15:
            return float("nan")
        return float(mad / std)

    @staticmethod
    def max_to_sum_ratio(returns):
        """
        Max-to-sum ratio: max(|x_i|) / sum(|x_i|).

        Converges to 0 for thin-tailed distributions.
        Stays large for fat-tailed distributions (single observation dominates).

        Parameters
        ----------
        returns : array-like

        Returns
        -------
        float
        """
        r = np.asarray(returns, dtype=np.float64)
        absvals = np.abs(r)
        total = np.sum(absvals)
        if total < 1e-15:
            return 0.0
        return float(np.max(absvals) / total)

    @classmethod
    def analyze(cls, returns):
        """
        Full tail diagnostic analysis.

        Parameters
        ----------
        returns : array-like

        Returns
        -------
        TailDiagnosticsResult
        """
        r = np.asarray(returns, dtype=np.float64)
        n = len(r)
        absvals = np.sort(np.abs(r))[::-1]  # descending
        k = max(5, int(math.sqrt(n)))

        alpha, alpha_se = cls.hill_estimator(absvals, k=k)
        kappa = cls.kappa_metric(r)
        mts = cls.max_to_sum_ratio(r)

        return TailDiagnosticsResult(
            alpha=alpha,
            alpha_se=alpha_se,
            kappa=kappa,
            kappa_gaussian=math.sqrt(2.0 / math.pi),
            max_to_sum=mts,
            n_observations=n,
            k_tail=k,
        )

"""
Portfolio Optimizer — Mean-variance, risk parity, Black-Litterman, efficient frontier.

Phase 2B of the prop desk platform build. Uses covariance data from
CovarianceBuilder (mc_engine.py) and provides optimal portfolio construction.

Classes:
    OptimalPortfolio      — Dataclass for optimization results
    MeanVarianceOptimizer — Classic Markowitz with projected gradient descent
    RiskParityOptimizer   — Equal risk contribution via iterative adjustment
    BlackLittermanModel   — Market equilibrium + subjective views
    EfficientFrontier     — Sweep target returns, compute frontier
    Rebalancer            — Compute trades to reach target weights
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── OptimalPortfolio ────────────────────────────────────────────────────

@dataclass
class OptimalPortfolio:
    """Result of a portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    expected_vol: float
    sharpe_ratio: float
    method: str
    tickers: list
    timestamp: str = ""


# ── Simplex projection ─────────────────────────────────────────────────

def _project_simplex(v):
    """
    Project vector v onto the unit simplex (sum=1, all >= 0).

    O(n log n) sorting algorithm.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.max(np.where(u * np.arange(1, n + 1) > cssv)[0]) + 1 if np.any(u * np.arange(1, n + 1) > cssv) else n
    theta = cssv[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


# ── MeanVarianceOptimizer ───────────────────────────────────────────────

class MeanVarianceOptimizer:
    """
    Classic Markowitz mean-variance optimization.

    Uses projected gradient descent on the simplex (no scipy required).
    Takes cov_data from CovarianceBuilder.build() which provides Cholesky L.
    """

    def __init__(self, cov_data, risk_free_rate=0.05):
        """
        Parameters
        ----------
        cov_data       : dict with keys S0, mu, sigma, L, tickers
        risk_free_rate : float — annualized risk-free rate for Sharpe
        """
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.L = cov_data["L"]
        self.Sigma = self.L @ self.L.T
        self.mu = cov_data["mu"]
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

    def optimize(self, target_return=None, risk_aversion=1.0, long_only=True,
                 max_iter=1000, lr=0.01, tol=1e-8):
        """
        Find optimal portfolio weights.

        If target_return is given, minimize variance subject to E[r] >= target.
        Otherwise, maximize: E[r] - (risk_aversion/2) * w' Sigma w.

        Parameters
        ----------
        target_return  : float or None
        risk_aversion  : float — higher = more conservative
        long_only      : bool — if True, constrain weights >= 0
        max_iter       : int
        lr             : float — learning rate
        tol            : float — convergence tolerance

        Returns
        -------
        OptimalPortfolio
        """
        w = np.ones(self.n) / self.n  # equal weight start

        for i in range(max_iter):
            # Gradient of objective: maximize mu'w - (ra/2) * w'Σw
            grad = self.mu - risk_aversion * (self.Sigma @ w)

            if target_return is not None:
                # Penalize if return below target
                port_ret = self.mu @ w
                if port_ret < target_return:
                    grad += 10.0 * self.mu  # push toward higher return

            w_new = w + lr * grad

            if long_only:
                w_new = _project_simplex(w_new)
            else:
                w_new = w_new / np.sum(np.abs(w_new))  # normalize

            # Check convergence
            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                break
            w = w_new

        # Ensure sum to 1 for long-only
        if long_only:
            w = np.maximum(w, 0.0)
            s = w.sum()
            if s > 0:
                w /= s

        port_ret = float(self.mu @ w)
        port_vol = float(np.sqrt(w @ self.Sigma @ w))
        sharpe = (port_ret - self.rf) / max(port_vol, 1e-10)

        return OptimalPortfolio(
            weights=w,
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe_ratio=sharpe,
            method="mean_variance",
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )


# ── RiskParityOptimizer ────────────────────────────────────────────────

class RiskParityOptimizer:
    """
    Equal risk contribution portfolio.

    Iterative proportional adjustment: w_i <- 1 / MRC_i, then normalize.
    MRC_i = (Sigma @ w)_i * w_i / (w' Sigma w)
    """

    def __init__(self, cov_data, risk_free_rate=0.05):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.L = cov_data["L"]
        self.Sigma = self.L @ self.L.T
        self.mu = cov_data["mu"]
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

    def optimize(self, max_iter=500, tol=1e-8):
        """
        Find risk parity weights.

        Returns
        -------
        OptimalPortfolio
        """
        w = np.ones(self.n) / self.n

        for _ in range(max_iter):
            Sw = self.Sigma @ w
            port_var = w @ Sw

            if port_var < 1e-20:
                break

            # Marginal risk contribution
            mrc = Sw * w / port_var

            # Inverse MRC weighting
            inv_mrc = np.where(mrc > 1e-20, 1.0 / mrc, 1.0)
            w_new = inv_mrc / inv_mrc.sum()

            if np.max(np.abs(w_new - w)) < tol:
                w = w_new
                break
            w = w_new

        w = np.maximum(w, 0.0)
        s = w.sum()
        if s > 0:
            w /= s

        port_ret = float(self.mu @ w)
        port_vol = float(np.sqrt(w @ self.Sigma @ w))
        sharpe = (port_ret - self.rf) / max(port_vol, 1e-10)

        return OptimalPortfolio(
            weights=w,
            expected_return=port_ret,
            expected_vol=port_vol,
            sharpe_ratio=sharpe,
            method="risk_parity",
            tickers=self.tickers,
            timestamp=datetime.now().isoformat(),
        )


# ── BlackLittermanModel ────────────────────────────────────────────────

class BlackLittermanModel:
    """
    Black-Litterman model: market equilibrium + trader views.

    Combines market-implied returns with subjective views using
    Bayesian updating to produce posterior expected returns.
    """

    def __init__(self, cov_data, market_cap_weights=None, delta=2.5, tau=0.05,
                 risk_free_rate=0.05):
        """
        Parameters
        ----------
        cov_data          : dict from CovarianceBuilder
        market_cap_weights: np.ndarray or None — if None, uses equal weights
        delta             : float — risk aversion coefficient
        tau               : float — scaling factor for uncertainty in prior
        risk_free_rate    : float
        """
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.L = cov_data["L"]
        self.Sigma = self.L @ self.L.T
        self.mu = cov_data["mu"]
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)
        self.delta = delta
        self.tau = tau

        if market_cap_weights is not None:
            self.w_mkt = np.asarray(market_cap_weights, dtype=np.float64)
        else:
            self.w_mkt = np.ones(self.n) / self.n

        # Implied equilibrium returns: pi = delta * Sigma * w_mkt
        self.pi = self.delta * self.Sigma @ self.w_mkt

        self._views_P = []  # pick matrices
        self._views_Q = []  # expected returns
        self._views_omega = []  # view confidence (variance)

    def add_view(self, view_type, assets, returns, confidence):
        """
        Add a view to the model.

        Parameters
        ----------
        view_type  : str — "absolute" or "relative"
        assets     : list[str] — ticker names
        returns    : float — expected return
        confidence : float — confidence level (0 to 1, higher = more confident)
        """
        P = np.zeros(self.n)

        if view_type == "absolute":
            for asset in assets:
                if asset in self.tickers:
                    idx = self.tickers.index(asset)
                    P[idx] = 1.0

        elif view_type == "relative":
            if len(assets) >= 2:
                # First asset outperforms second
                idx1 = self.tickers.index(assets[0]) if assets[0] in self.tickers else None
                idx2 = self.tickers.index(assets[1]) if assets[1] in self.tickers else None
                if idx1 is not None and idx2 is not None:
                    P[idx1] = 1.0
                    P[idx2] = -1.0

        # View uncertainty: inverse confidence
        omega = (1.0 - confidence) * self.tau * (P @ self.Sigma @ P) + 1e-10

        self._views_P.append(P)
        self._views_Q.append(returns)
        self._views_omega.append(omega)

    def compute_posterior(self):
        """
        Compute posterior expected returns and covariance.

        Returns
        -------
        (mu_BL, Sigma_BL) : tuple[np.ndarray, np.ndarray]
        """
        if not self._views_P:
            return self.pi, self.Sigma

        P = np.array(self._views_P)  # [k, n]
        Q = np.array(self._views_Q)  # [k]
        Omega = np.diag(self._views_omega)  # [k, k]

        tau_Sigma = self.tau * self.Sigma

        # BL formula: mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
        #                    * [(tau*Sigma)^-1 * pi + P'*Omega^-1*Q]
        inv_tau_Sigma = np.linalg.solve(tau_Sigma, np.eye(self.n))
        inv_Omega = np.linalg.solve(Omega, np.eye(len(Q)))

        M = inv_tau_Sigma + P.T @ inv_Omega @ P
        mu_BL = np.linalg.solve(M, inv_tau_Sigma @ self.pi + P.T @ inv_Omega @ Q)

        # Posterior covariance
        Sigma_BL = np.linalg.solve(M, np.eye(self.n)) + self.Sigma

        return mu_BL, Sigma_BL

    def optimize(self, risk_aversion=None, long_only=True):
        """
        Optimize portfolio using BL posterior.

        Returns
        -------
        OptimalPortfolio
        """
        mu_BL, Sigma_BL = self.compute_posterior()
        ra = risk_aversion if risk_aversion is not None else self.delta

        # Create a cov_data-like dict for MV optimizer
        # Use Cholesky of posterior covariance
        try:
            L_BL = np.linalg.cholesky(Sigma_BL)
        except np.linalg.LinAlgError:
            # Add regularization if not PD
            Sigma_BL += 1e-6 * np.eye(self.n)
            L_BL = np.linalg.cholesky(Sigma_BL)

        cov_data_bl = {
            "S0": self.cov_data["S0"],
            "mu": mu_BL,
            "sigma": np.sqrt(np.diag(Sigma_BL)),
            "L": L_BL,
            "tickers": self.tickers,
        }

        mv = MeanVarianceOptimizer(cov_data_bl, risk_free_rate=self.rf)
        result = mv.optimize(risk_aversion=ra, long_only=long_only)
        result.method = "black_litterman"
        return result


# ── EfficientFrontier ──────────────────────────────────────────────────

class EfficientFrontier:
    """
    Compute the efficient frontier by sweeping target returns.

    Solves MV optimization at each target return level to trace
    the minimum variance frontier.
    """

    def __init__(self, cov_data, risk_free_rate=0.05):
        self.cov_data = cov_data
        self.rf = risk_free_rate
        self.mu = cov_data["mu"]
        self.tickers = cov_data["tickers"]
        self.n = len(self.tickers)

    def compute(self, n_points=50, long_only=True):
        """
        Compute the efficient frontier.

        Parameters
        ----------
        n_points  : int — number of points on the frontier
        long_only : bool

        Returns
        -------
        Table — MnTable with (target_return, portfolio_vol, sharpe) + weight columns
        """
        from bank_python.mntable import Table

        schema = [
            ("target_return", float),
            ("portfolio_vol", float),
            ("sharpe", float),
        ]
        for t in self.tickers:
            schema.append((f"w_{t}", float))

        result = Table(schema, name="efficient_frontier")

        mu_min = float(np.min(self.mu))
        mu_max = float(np.max(self.mu))
        # Extend range slightly
        targets = np.linspace(mu_min * 0.5, mu_max * 1.5, n_points)

        mv = MeanVarianceOptimizer(self.cov_data, risk_free_rate=self.rf)

        for target in targets:
            port = mv.optimize(target_return=target, long_only=long_only,
                               max_iter=500, lr=0.005)
            row = {
                "target_return": float(target),
                "portfolio_vol": port.expected_vol,
                "sharpe": port.sharpe_ratio,
            }
            for i, t in enumerate(self.tickers):
                row[f"w_{t}"] = float(port.weights[i])
            result.append(row)

        return result

    def render_ascii(self, console, frontier_table=None, n_points=30):
        """
        Render an ASCII scatter plot of the efficient frontier.

        Parameters
        ----------
        console        : rich.console.Console
        frontier_table : Table or None — if None, computes it
        n_points       : int
        """
        from rich.panel import Panel

        if frontier_table is None:
            frontier_table = self.compute(n_points=n_points)

        rows = list(frontier_table)
        if not rows:
            console.print("[yellow]No frontier data[/yellow]")
            return

        vols = [r["portfolio_vol"] for r in rows]
        rets = [r["target_return"] for r in rows]

        vol_min, vol_max = min(vols), max(vols)
        ret_min, ret_max = min(rets), max(rets)

        # ASCII grid
        width = 60
        height = 20
        grid = [[" "] * width for _ in range(height)]

        vol_range = max(vol_max - vol_min, 1e-10)
        ret_range = max(ret_max - ret_min, 1e-10)

        for v, r in zip(vols, rets):
            x = int((v - vol_min) / vol_range * (width - 1))
            y = int((r - ret_min) / ret_range * (height - 1))
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            grid[height - 1 - y][x] = "*"

        lines = ["Efficient Frontier (Risk vs Return)"]
        lines.append(f"  Return ↑  ({ret_min:.1%} to {ret_max:.1%})")
        for row in grid:
            lines.append("  " + "".join(row))
        lines.append(f"  Risk →    ({vol_min:.1%} to {vol_max:.1%})")

        # Find max Sharpe portfolio
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["sharpe"])
        best = rows[best_idx]
        lines.append(f"  Max Sharpe: vol={best['portfolio_vol']:.1%}, "
                     f"ret={best['target_return']:.1%}, "
                     f"sharpe={best['sharpe']:.2f}")

        console.print(Panel("\n".join(lines), border_style="cyan",
                            title="Efficient Frontier"))


# ── Rebalancer ──────────────────────────────────────────────────────────

class Rebalancer:
    """
    Compute trades needed to reach target portfolio weights.

    Accounts for round lots and minimum trade thresholds.
    Optionally executes through TradingEngine.
    """

    def __init__(self, min_trade_value=100.0, round_lot=1):
        self.min_trade_value = min_trade_value
        self.round_lot = round_lot

    def compute_trades(self, current_positions, target_weights, total_value, prices):
        """
        Compute rebalancing trades.

        Parameters
        ----------
        current_positions : dict[str, float] — ticker -> current quantity
        target_weights    : dict[str, float] — ticker -> target weight (0-1)
        total_value       : float — total portfolio value
        prices            : dict[str, float] — ticker -> current price

        Returns
        -------
        Table — MnTable with (ticker, current_qty, target_qty, trade_qty,
                trade_value, current_weight, target_weight)
        """
        from bank_python.mntable import Table

        result = Table([
            ("ticker", str),
            ("current_qty", float),
            ("target_qty", float),
            ("trade_qty", float),
            ("trade_value", float),
            ("current_weight", float),
            ("target_weight", float),
        ], name="rebalance_trades")

        all_tickers = set(list(current_positions.keys()) + list(target_weights.keys()))

        for ticker in sorted(all_tickers):
            curr_qty = current_positions.get(ticker, 0.0)
            tgt_weight = target_weights.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)

            if price <= 0:
                continue

            curr_value = curr_qty * price
            curr_weight = curr_value / max(total_value, 1.0)
            tgt_value = tgt_weight * total_value
            tgt_qty = tgt_value / price

            # Round to lot size
            if self.round_lot > 0:
                tgt_qty = round(tgt_qty / self.round_lot) * self.round_lot

            trade_qty = tgt_qty - curr_qty
            trade_value = trade_qty * price

            # Skip tiny trades
            if abs(trade_value) < self.min_trade_value:
                trade_qty = 0.0
                trade_value = 0.0

            result.append({
                "ticker": ticker,
                "current_qty": float(curr_qty),
                "target_qty": float(tgt_qty),
                "trade_qty": float(trade_qty),
                "trade_value": float(trade_value),
                "current_weight": float(curr_weight),
                "target_weight": float(tgt_weight),
            })

        return result

    def execute_rebalance(self, trades_table, trader, trading_engine):
        """
        Execute rebalancing trades through the TradingEngine.

        Parameters
        ----------
        trades_table   : Table — from compute_trades()
        trader         : Trader object
        trading_engine : TradingEngine
        """
        orders = []
        for row in trades_table:
            qty = row["trade_qty"]
            if abs(qty) < 1:
                continue

            side = "BUY" if qty > 0 else "SELL"
            order = trading_engine.submit_order(
                trader=trader,
                instrument_name=row["ticker"],
                side=side,
                quantity=abs(qty),
            )
            orders.append(order)

        return orders


# ── Rendering helpers ──────────────────────────────────────────────────

def render_optimal_portfolio(portfolio, console):
    """Render an OptimalPortfolio as a Rich panel."""
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    tbl = RichTable(
        title=f"Optimal Portfolio ({portfolio.method})",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Ticker", style="bold", width=10)
    tbl.add_column("Weight", justify="right", width=12)

    for i, ticker in enumerate(portfolio.tickers):
        w = portfolio.weights[i]
        if w > 0.001:
            tbl.add_row(ticker, f"{w:.1%}")

    tbl.add_row("", "")
    tbl.add_row(Text("E[Return]", style="bold"), f"{portfolio.expected_return:.2%}")
    tbl.add_row(Text("Volatility", style="bold"), f"{portfolio.expected_vol:.2%}")
    tbl.add_row(Text("Sharpe", style="bold"), f"{portfolio.sharpe_ratio:.2f}")

    console.print(Panel(tbl, border_style="cyan"))


def create_optimizer(cov_data, method="mv", backend="native", risk_free_rate=0.05, **kwargs):
    """
    Factory function that dispatches to the appropriate optimizer backend.

    Parameters
    ----------
    cov_data       : dict from CovarianceBuilder.build()
    method         : str — "mv", "rp", "bl", "hrp", "cvar", "risk_budget"
    backend        : str — "native", "pypfopt", "riskfolio"
    risk_free_rate : float
    **kwargs       : passed through to the optimizer constructor

    Returns
    -------
    An optimizer instance with an .optimize() method that returns OptimalPortfolio.
    """
    if backend == "native":
        if method == "mv":
            return MeanVarianceOptimizer(cov_data, risk_free_rate=risk_free_rate)
        elif method == "rp":
            return RiskParityOptimizer(cov_data, risk_free_rate=risk_free_rate)
        elif method == "bl":
            return BlackLittermanModel(cov_data, risk_free_rate=risk_free_rate, **kwargs)
        else:
            raise ValueError(f"Native backend does not support method={method!r}. "
                             f"Use 'mv', 'rp', or 'bl'.")

    elif backend == "pypfopt":
        from bank_python.integrations import require_pypfopt
        require_pypfopt()
        from bank_python.integrations.pypfopt_optimizer import (
            PyPfOptMeanVariance, PyPfOptHRP, PyPfOptBlackLitterman,
        )
        if method in ("mv", "max_sharpe", "min_volatility"):
            return PyPfOptMeanVariance(cov_data, risk_free_rate=risk_free_rate)
        elif method == "hrp":
            return PyPfOptHRP(cov_data, risk_free_rate=risk_free_rate)
        elif method == "bl":
            return PyPfOptBlackLitterman(cov_data, risk_free_rate=risk_free_rate, **kwargs)
        else:
            raise ValueError(f"PyPortfolioOpt backend does not support method={method!r}. "
                             f"Use 'mv', 'max_sharpe', 'min_volatility', 'hrp', or 'bl'.")

    elif backend == "riskfolio":
        from bank_python.integrations import require_riskfolio
        require_riskfolio()
        from bank_python.integrations.riskfolio_optimizer import (
            RiskfolioCVaROptimizer, RiskfolioRiskBudgeting,
        )
        if method == "cvar":
            return RiskfolioCVaROptimizer(cov_data, risk_free_rate=risk_free_rate, **kwargs)
        elif method == "risk_budget":
            return RiskfolioRiskBudgeting(cov_data, risk_free_rate=risk_free_rate, **kwargs)
        else:
            raise ValueError(f"Riskfolio backend does not support method={method!r}. "
                             f"Use 'cvar' or 'risk_budget'.")

    else:
        raise ValueError(f"Unknown backend={backend!r}. Use 'native', 'pypfopt', or 'riskfolio'.")


def render_rebalance_trades(trades_table, console):
    """Render rebalancing trades as a Rich table."""
    from rich.panel import Panel
    from rich.table import Table as RichTable
    from rich.text import Text

    tbl = RichTable(
        title="Rebalancing Trades",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Ticker", style="bold", width=8)
    tbl.add_column("Current", justify="right", width=10)
    tbl.add_column("Target", justify="right", width=10)
    tbl.add_column("Trade", justify="right", width=10)
    tbl.add_column("Value", justify="right", width=12)
    tbl.add_column("Curr Wt", justify="right", width=8)
    tbl.add_column("Tgt Wt", justify="right", width=8)

    for row in trades_table:
        trade_qty = row["trade_qty"]
        if abs(trade_qty) < 0.5:
            continue
        style = "green" if trade_qty > 0 else "red"
        tbl.add_row(
            row["ticker"],
            f"{row['current_qty']:,.0f}",
            f"{row['target_qty']:,.0f}",
            Text(f"{trade_qty:+,.0f}", style=style),
            Text(f"${row['trade_value']:+,.0f}", style=style),
            f"{row['current_weight']:.1%}",
            f"{row['target_weight']:.1%}",
        )

    console.print(Panel(tbl, border_style="cyan"))

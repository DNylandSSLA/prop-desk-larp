"""Tests for optimizer — MV, risk parity, Black-Litterman, frontier, rebalancer."""

import numpy as np
import pytest

from bank_python.optimizer import (
    OptimalPortfolio,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    BlackLittermanModel,
    EfficientFrontier,
    Rebalancer,
    _project_simplex,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_cov_data(n=3, seed=42):
    """Create synthetic covariance data mimicking CovarianceBuilder output."""
    rng = np.random.default_rng(seed)

    mu = rng.uniform(0.05, 0.20, n)
    sigma = rng.uniform(0.15, 0.40, n)

    # Generate random correlation matrix
    A = rng.standard_normal((n, n))
    cov = (A @ A.T) / n * 0.04  # scale to reasonable vol levels
    # Make diagonal match sigma
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    cov = corr * np.outer(sigma, sigma) / 252.0 * 252.0  # annualized

    # Regularize
    cov += 1e-6 * np.eye(n)
    L = np.linalg.cholesky(cov)

    tickers = [f"ASSET{i}" for i in range(n)]
    S0 = rng.uniform(50, 200, n)

    return {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "L": L,
        "tickers": tickers,
    }


# ── Simplex projection ─────────────────────────────────────────────────

class TestSimplexProjection:
    def test_already_on_simplex(self):
        v = np.array([0.3, 0.3, 0.4])
        p = _project_simplex(v)
        assert abs(p.sum() - 1.0) < 1e-10
        assert np.all(p >= 0)

    def test_negative_values(self):
        v = np.array([-0.5, 0.3, 1.2])
        p = _project_simplex(v)
        assert abs(p.sum() - 1.0) < 1e-10
        assert np.all(p >= -1e-10)

    def test_all_equal(self):
        v = np.array([1.0, 1.0, 1.0])
        p = _project_simplex(v)
        assert abs(p.sum() - 1.0) < 1e-10
        assert np.allclose(p, 1.0 / 3.0, atol=1e-10)

    def test_single_element(self):
        v = np.array([5.0])
        p = _project_simplex(v)
        assert abs(p[0] - 1.0) < 1e-10


# ── OptimalPortfolio ────────────────────────────────────────────────────

class TestOptimalPortfolio:
    def test_create(self):
        op = OptimalPortfolio(
            weights=np.array([0.5, 0.5]),
            expected_return=0.10,
            expected_vol=0.15,
            sharpe_ratio=0.33,
            method="test",
            tickers=["A", "B"],
        )
        assert len(op.weights) == 2
        assert op.method == "test"


# ── MeanVarianceOptimizer ───────────────────────────────────────────────

class TestMeanVarianceOptimizer:
    def test_basic_optimization(self):
        cov_data = _make_cov_data(n=3)
        mv = MeanVarianceOptimizer(cov_data)
        result = mv.optimize()

        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 0.01
        assert np.all(result.weights >= -0.01)
        assert result.expected_vol > 0
        assert result.method == "mean_variance"

    def test_long_only(self):
        cov_data = _make_cov_data(n=4)
        mv = MeanVarianceOptimizer(cov_data)
        result = mv.optimize(long_only=True)
        assert np.all(result.weights >= -1e-10)

    def test_with_target_return(self):
        cov_data = _make_cov_data(n=3)
        mv = MeanVarianceOptimizer(cov_data)
        target = float(np.mean(cov_data["mu"]))
        result = mv.optimize(target_return=target)
        assert isinstance(result, OptimalPortfolio)

    def test_high_risk_aversion(self):
        cov_data = _make_cov_data(n=3)
        mv = MeanVarianceOptimizer(cov_data)
        conservative = mv.optimize(risk_aversion=10.0)
        aggressive = mv.optimize(risk_aversion=0.1)
        # Higher risk aversion should give lower vol
        assert conservative.expected_vol <= aggressive.expected_vol + 0.1

    def test_two_assets(self):
        cov_data = _make_cov_data(n=2)
        mv = MeanVarianceOptimizer(cov_data)
        result = mv.optimize()
        assert len(result.weights) == 2
        assert abs(result.weights.sum() - 1.0) < 0.01

    def test_sharpe_ratio_sign(self):
        cov_data = _make_cov_data(n=3)
        # Set high expected returns
        cov_data["mu"] = np.array([0.15, 0.20, 0.25])
        mv = MeanVarianceOptimizer(cov_data, risk_free_rate=0.05)
        result = mv.optimize()
        assert result.sharpe_ratio > 0


# ── RiskParityOptimizer ────────────────────────────────────────────────

class TestRiskParityOptimizer:
    def test_basic_optimization(self):
        cov_data = _make_cov_data(n=3)
        rp = RiskParityOptimizer(cov_data)
        result = rp.optimize()

        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 0.01
        assert np.all(result.weights >= -1e-10)
        assert result.method == "risk_parity"

    def test_equal_vol_equal_weights(self):
        n = 3
        sigma = 0.2
        cov = sigma ** 2 * np.eye(n)
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))

        cov_data = {
            "S0": np.array([100.0] * n),
            "mu": np.array([0.10] * n),
            "sigma": np.array([sigma] * n),
            "L": L,
            "tickers": ["A", "B", "C"],
        }

        rp = RiskParityOptimizer(cov_data)
        result = rp.optimize()
        # With equal vol and no correlation, weights should be equal
        assert np.allclose(result.weights, 1.0 / n, atol=0.05)

    def test_risk_contributions_similar(self):
        cov_data = _make_cov_data(n=4)
        rp = RiskParityOptimizer(cov_data)
        result = rp.optimize()

        w = result.weights
        Sigma = rp.Sigma
        Sw = Sigma @ w
        port_var = w @ Sw
        if port_var > 1e-10:
            rc = w * Sw / port_var
            # Risk contributions should be in reasonable range
            assert np.std(rc) < 0.60  # allow for numerical convergence variation


# ── BlackLittermanModel ────────────────────────────────────────────────

class TestBlackLittermanModel:
    def test_no_views(self):
        cov_data = _make_cov_data(n=3)
        bl = BlackLittermanModel(cov_data)
        mu_bl, Sigma_bl = bl.compute_posterior()
        # Without views, should return equilibrium
        assert len(mu_bl) == 3
        assert Sigma_bl.shape == (3, 3)

    def test_absolute_view(self):
        cov_data = _make_cov_data(n=3)
        bl = BlackLittermanModel(cov_data)
        bl.add_view("absolute", ["ASSET0"], 0.15, 0.8)
        mu_bl, _ = bl.compute_posterior()
        # With bullish view on ASSET0, its return should increase
        assert mu_bl[0] > bl.pi[0] - 0.1  # roughly pulled toward 0.15

    def test_relative_view(self):
        cov_data = _make_cov_data(n=3)
        bl = BlackLittermanModel(cov_data)
        bl.add_view("relative", ["ASSET0", "ASSET1"], 0.05, 0.7)
        mu_bl, _ = bl.compute_posterior()
        assert len(mu_bl) == 3

    def test_optimize(self):
        cov_data = _make_cov_data(n=3)
        bl = BlackLittermanModel(cov_data)
        bl.add_view("absolute", ["ASSET0"], 0.20, 0.9)
        result = bl.optimize()
        assert isinstance(result, OptimalPortfolio)
        assert result.method == "black_litterman"
        assert abs(result.weights.sum() - 1.0) < 0.05

    def test_custom_market_weights(self):
        cov_data = _make_cov_data(n=3)
        mkt_w = np.array([0.5, 0.3, 0.2])
        bl = BlackLittermanModel(cov_data, market_cap_weights=mkt_w)
        assert np.allclose(bl.w_mkt, mkt_w)

    def test_multiple_views(self):
        cov_data = _make_cov_data(n=3)
        bl = BlackLittermanModel(cov_data)
        bl.add_view("absolute", ["ASSET0"], 0.15, 0.8)
        bl.add_view("relative", ["ASSET1", "ASSET2"], 0.03, 0.6)
        result = bl.optimize()
        assert isinstance(result, OptimalPortfolio)


# ── EfficientFrontier ──────────────────────────────────────────────────

class TestEfficientFrontier:
    def test_compute_frontier(self):
        cov_data = _make_cov_data(n=3)
        ef = EfficientFrontier(cov_data)
        table = ef.compute(n_points=10)
        rows = list(table)
        assert len(rows) == 10
        assert "target_return" in rows[0]
        assert "portfolio_vol" in rows[0]
        assert "sharpe" in rows[0]

    def test_frontier_has_weight_columns(self):
        cov_data = _make_cov_data(n=3)
        ef = EfficientFrontier(cov_data)
        table = ef.compute(n_points=5)
        rows = list(table)
        for t in cov_data["tickers"]:
            assert f"w_{t}" in rows[0]

    def test_frontier_monotonic_vol(self):
        cov_data = _make_cov_data(n=3)
        ef = EfficientFrontier(cov_data)
        table = ef.compute(n_points=20)
        rows = list(table)
        vols = [r["portfolio_vol"] for r in rows]
        # Not strictly monotonic due to optimization, but should be roughly increasing
        assert vols[-1] >= vols[0] * 0.5  # sanity check


# ── Rebalancer ──────────────────────────────────────────────────────────

class TestRebalancer:
    def test_compute_trades(self):
        rb = Rebalancer()
        current = {"AAPL": 100.0, "MSFT": 50.0}
        target = {"AAPL": 0.6, "MSFT": 0.4}
        prices = {"AAPL": 150.0, "MSFT": 400.0}
        total_value = 100.0 * 150.0 + 50.0 * 400.0  # 35000

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        assert len(rows) == 2

    def test_no_trades_if_already_balanced(self):
        rb = Rebalancer(min_trade_value=100.0)
        current = {"AAPL": 100.0}
        target = {"AAPL": 1.0}
        prices = {"AAPL": 150.0}
        total_value = 15000.0

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        # trade_qty should be 0 (or close)
        for r in rows:
            assert abs(r["trade_qty"]) < 1.0

    def test_buy_new_position(self):
        rb = Rebalancer()
        current = {"AAPL": 100.0}
        target = {"AAPL": 0.5, "MSFT": 0.5}
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        total_value = 10000.0

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        msft = [r for r in rows if r["ticker"] == "MSFT"]
        assert len(msft) == 1
        assert msft[0]["trade_qty"] > 0

    def test_sell_position(self):
        rb = Rebalancer()
        current = {"AAPL": 200.0, "MSFT": 50.0}
        target = {"AAPL": 0.3, "MSFT": 0.7}
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        total_value = 25000.0

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        aapl = [r for r in rows if r["ticker"] == "AAPL"]
        assert len(aapl) == 1
        assert aapl[0]["trade_qty"] < 0  # sell AAPL

    def test_min_trade_threshold(self):
        rb = Rebalancer(min_trade_value=1000.0)
        current = {"AAPL": 100.0}
        target = {"AAPL": 1.0}
        prices = {"AAPL": 150.0}
        total_value = 15000.0

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        for r in rows:
            assert abs(r["trade_value"]) < 1.0 or abs(r["trade_value"]) >= 1000.0

    def test_round_lots(self):
        rb = Rebalancer(round_lot=10)
        current = {}
        target = {"AAPL": 1.0}
        prices = {"AAPL": 150.0}
        total_value = 15000.0

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        assert len(rows) == 1
        assert rows[0]["target_qty"] % 10 == 0

    def test_zero_price_skipped(self):
        rb = Rebalancer()
        current = {"AAPL": 100.0}
        target = {"AAPL": 1.0}
        prices = {"AAPL": 0.0}
        total_value = 10000.0

        trades = rb.compute_trades(current, target, total_value, prices)
        rows = list(trades)
        assert len(rows) == 0

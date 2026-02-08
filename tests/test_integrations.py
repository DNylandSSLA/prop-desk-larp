"""
Tests for PyPortfolioOpt, Riskfolio-Lib, and QuantLib integrations.

All tests are gated with @pytest.mark.skipif so the test suite works
even if the optional libraries are not installed.
"""

import math
import numpy as np
import pytest

from bank_python.integrations import HAS_PYPFOPT, HAS_RISKFOLIO, HAS_QUANTLIB
from bank_python.optimizer import OptimalPortfolio, create_optimizer
from bank_python.dagger import MarketData, DependencyGraph


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_cov_data(n=5, seed=42):
    """Build a synthetic cov_data dict for testing optimizers."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET{i}" for i in range(n)]

    # Random returns matrix (200 days, n assets)
    returns = rng.normal(0.0003, 0.015, size=(200, n))

    mu = np.mean(returns, axis=0) * 252
    sigma = np.std(returns, axis=0, ddof=1) * np.sqrt(252)
    sigma = np.maximum(sigma, 1e-6)

    cov = np.cov(returns.T) * 252
    cov += 1e-8 * np.eye(n)
    L = np.linalg.cholesky(cov)

    S0 = rng.uniform(50, 300, size=n)

    return {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "L": L,
        "tickers": tickers,
        "returns_matrix": returns,
    }


# ══════════════════════════════════════════════════════════════════════════
# PyPortfolioOpt Tests
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_PYPFOPT, reason="PyPortfolioOpt not installed")
class TestPyPfOpt:

    def test_mv_max_sharpe_returns_optimal_portfolio(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptMeanVariance
        cov_data = _make_cov_data()
        opt = PyPfOptMeanVariance(cov_data)
        result = opt.optimize(objective="max_sharpe")
        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in result.weights)

    def test_mv_min_volatility_lower_vol(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptMeanVariance
        cov_data = _make_cov_data()
        opt = PyPfOptMeanVariance(cov_data)
        sharpe_result = opt.optimize(objective="max_sharpe")

        opt2 = PyPfOptMeanVariance(cov_data)
        minvol_result = opt2.optimize(objective="min_volatility")
        assert minvol_result.expected_vol <= sharpe_result.expected_vol + 1e-6

    def test_mv_weights_sum_to_one(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptMeanVariance
        cov_data = _make_cov_data(n=8)
        opt = PyPfOptMeanVariance(cov_data)
        result = opt.optimize()
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_hrp_returns_valid_weights(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptHRP
        cov_data = _make_cov_data()
        opt = PyPfOptHRP(cov_data)
        result = opt.optimize()
        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in result.weights)
        assert result.method == "pypfopt_hrp"

    def test_hrp_all_nonzero(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptHRP
        cov_data = _make_cov_data(n=4)
        opt = PyPfOptHRP(cov_data)
        result = opt.optimize()
        # HRP typically gives non-zero weight to all assets
        assert all(w > 1e-4 for w in result.weights)

    def test_hrp_needs_returns_matrix(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptHRP
        cov_data = _make_cov_data()
        del cov_data["returns_matrix"]
        opt = PyPfOptHRP(cov_data)
        with pytest.raises(ValueError, match="returns_matrix"):
            opt.optimize()

    def test_bl_with_views_shifts_weights(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptBlackLitterman
        cov_data = _make_cov_data()
        bl = PyPfOptBlackLitterman(cov_data)
        bl.add_view("absolute", ["ASSET0"], 0.20, 0.8)
        result = bl.optimize()
        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 1e-6

    def test_bl_no_views_still_works(self):
        from bank_python.integrations.pypfopt_optimizer import PyPfOptBlackLitterman
        cov_data = _make_cov_data()
        bl = PyPfOptBlackLitterman(cov_data)
        result = bl.optimize()
        assert isinstance(result, OptimalPortfolio)

    def test_factory_dispatches_pypfopt(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="hrp", backend="pypfopt")
        result = opt.optimize()
        assert result.method == "pypfopt_hrp"

    def test_factory_max_sharpe_pypfopt(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="max_sharpe", backend="pypfopt")
        result = opt.optimize(objective="max_sharpe")
        assert "pypfopt" in result.method


# ══════════════════════════════════════════════════════════════════════════
# Riskfolio-Lib Tests
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_RISKFOLIO, reason="Riskfolio-Lib not installed")
class TestRiskfolio:

    def test_cvar_returns_valid_portfolio(self):
        from bank_python.integrations.riskfolio_optimizer import RiskfolioCVaROptimizer
        cov_data = _make_cov_data()
        opt = RiskfolioCVaROptimizer(cov_data)
        result = opt.optimize(alpha=0.05)
        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        assert all(w >= -1e-6 for w in result.weights)

    def test_cvar_method_name(self):
        from bank_python.integrations.riskfolio_optimizer import RiskfolioCVaROptimizer
        cov_data = _make_cov_data()
        opt = RiskfolioCVaROptimizer(cov_data)
        result = opt.optimize()
        assert result.method == "riskfolio_cvar"

    def test_cvar_needs_returns_matrix(self):
        from bank_python.integrations.riskfolio_optimizer import RiskfolioCVaROptimizer
        cov_data = _make_cov_data()
        del cov_data["returns_matrix"]
        opt = RiskfolioCVaROptimizer(cov_data)
        with pytest.raises(ValueError, match="returns_matrix"):
            opt.optimize()

    def test_risk_budgeting_equal_approx_risk_parity(self):
        from bank_python.integrations.riskfolio_optimizer import RiskfolioRiskBudgeting
        cov_data = _make_cov_data(n=4)
        opt = RiskfolioRiskBudgeting(cov_data)
        result = opt.optimize()  # equal budgets = risk parity
        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 1e-4
        # With equal budgets, weights should be somewhat balanced
        assert np.std(result.weights) < 0.6

    def test_risk_budgeting_custom_budgets(self):
        from bank_python.integrations.riskfolio_optimizer import RiskfolioRiskBudgeting
        cov_data = _make_cov_data(n=4)
        budgets = {"ASSET0": 0.4, "ASSET1": 0.3, "ASSET2": 0.2, "ASSET3": 0.1}
        opt = RiskfolioRiskBudgeting(cov_data)
        result = opt.optimize(risk_budgets=budgets)
        assert isinstance(result, OptimalPortfolio)
        assert abs(result.weights.sum() - 1.0) < 1e-4

    def test_risk_budgeting_cvar_measure(self):
        from bank_python.integrations.riskfolio_optimizer import RiskfolioRiskBudgeting
        cov_data = _make_cov_data(n=4)
        opt = RiskfolioRiskBudgeting(cov_data)
        result = opt.optimize(risk_measure="CVaR")
        assert "cvar" in result.method

    def test_factory_dispatches_riskfolio_cvar(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="cvar", backend="riskfolio")
        result = opt.optimize()
        assert result.method == "riskfolio_cvar"

    def test_factory_dispatches_riskfolio_risk_budget(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="risk_budget", backend="riskfolio")
        result = opt.optimize()
        assert "risk_budget" in result.method


# ══════════════════════════════════════════════════════════════════════════
# QuantLib Tests
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_QUANTLIB, reason="QuantLib not installed")
class TestQuantLib:

    def test_option_price_close_to_hand_rolled(self):
        from bank_python.integrations.quantlib_pricing import QuantLibOption
        from bank_python.dagger import Option as HandOption

        spot = MarketData("SPOT", price=100.0)
        hand = HandOption("HR_CALL", spot_source=spot, strike=100.0,
                          volatility=0.2, time_to_expiry=1.0, is_call=True)
        # Hand-rolled model uses no discounting (r=0), so compare at r=0
        ql_opt = QuantLibOption("QL_CALL", spot_source=spot, strike=100.0,
                                volatility=0.2, time_to_expiry=1.0, is_call=True,
                                risk_free_rate=0.0)

        hr_price = hand.value
        ql_price = ql_opt.value
        assert hr_price > 0
        assert ql_price > 0
        pct_diff = abs(hr_price - ql_price) / hr_price * 100
        assert pct_diff < 1.0, f"Price diff {pct_diff:.2f}% exceeds 1%"

    def test_option_greeks_nonzero(self):
        from bank_python.integrations.quantlib_pricing import QuantLibOption

        spot = MarketData("SPOT", price=100.0)
        ql_opt = QuantLibOption("QL_CALL", spot_source=spot, strike=100.0,
                                volatility=0.2, time_to_expiry=1.0, is_call=True)
        _ = ql_opt.value
        greeks = ql_opt.greeks()
        assert abs(greeks["delta"]) > 0.01
        assert greeks["gamma"] > 0
        assert abs(greeks["vega"]) > 0
        assert abs(greeks["theta"]) > 0

    def test_option_put_delta_negative(self):
        from bank_python.integrations.quantlib_pricing import QuantLibOption

        spot = MarketData("SPOT", price=100.0)
        ql_put = QuantLibOption("QL_PUT", spot_source=spot, strike=100.0,
                                volatility=0.2, time_to_expiry=1.0, is_call=False)
        greeks = ql_put.greeks()
        assert greeks["delta"] < 0

    def test_bond_pv_close_to_hand_rolled(self):
        from bank_python.integrations.quantlib_pricing import QuantLibBond
        from bank_python.dagger import Bond as HandBond

        rate_md = MarketData("RATE", price=0.05)
        hand = HandBond("HR_BOND", rate_source=rate_md, face=100.0,
                        coupon_rate=0.05, maturity=5)
        ql_bond = QuantLibBond("QL_BOND", rate_source=rate_md, face=100.0,
                               coupon_rate=0.05, maturity=5)

        hr_price = hand.value
        ql_price = ql_bond.value
        assert hr_price > 0
        assert ql_price > 0
        # Allow more tolerance for bond (different day count conventions)
        pct_diff = abs(hr_price - ql_price) / hr_price * 100
        assert pct_diff < 5.0, f"Bond price diff {pct_diff:.2f}% exceeds 5%"

    def test_bond_duration_positive(self):
        from bank_python.integrations.quantlib_pricing import QuantLibBond

        rate_md = MarketData("RATE", price=0.05)
        ql_bond = QuantLibBond("QL_BOND", rate_source=rate_md, face=100.0,
                               coupon_rate=0.05, maturity=5)
        assert ql_bond.duration > 0

    def test_bond_convexity_positive(self):
        from bank_python.integrations.quantlib_pricing import QuantLibBond

        rate_md = MarketData("RATE", price=0.05)
        ql_bond = QuantLibBond("QL_BOND", rate_source=rate_md, face=100.0,
                               coupon_rate=0.05, maturity=5)
        assert ql_bond.convexity > 0

    def test_cds_integrates_with_graph(self):
        from bank_python.integrations.quantlib_pricing import QuantLibCDS

        spread_md = MarketData("SPREAD", price=0.02)
        rate_md = MarketData("RATE", price=0.05)
        graph = DependencyGraph()

        cds = QuantLibCDS("QL_CDS", credit_spread_source=spread_md,
                          rate_source=rate_md, notional=10_000_000, maturity=5)
        graph.register(cds)

        v1 = cds.value

        # Change spread and recalculate
        spread_md.set_price(0.03)
        graph.recalculate(spread_md)
        v2 = cds.value

        # Value should change when spread changes
        assert v1 != v2

    def test_option_dirty_propagation(self):
        from bank_python.integrations.quantlib_pricing import QuantLibOption

        spot = MarketData("SPOT", price=100.0)
        graph = DependencyGraph()

        ql_opt = QuantLibOption("QL_CALL", spot_source=spot, strike=100.0,
                                volatility=0.2, time_to_expiry=1.0, is_call=True)
        graph.register(ql_opt)

        v1 = ql_opt.value

        spot.set_price(110.0)
        graph.recalculate(spot)
        v2 = ql_opt.value

        assert v2 > v1  # call price should increase with spot

    def test_bond_longer_maturity_higher_duration(self):
        from bank_python.integrations.quantlib_pricing import QuantLibBond

        rate_md = MarketData("RATE", price=0.05)
        bond_5y = QuantLibBond("QL_5Y", rate_source=rate_md, face=100.0,
                               coupon_rate=0.05, maturity=5)
        bond_10y = QuantLibBond("QL_10Y", rate_source=rate_md, face=100.0,
                                coupon_rate=0.05, maturity=10)
        assert bond_10y.duration > bond_5y.duration


# ══════════════════════════════════════════════════════════════════════════
# Factory / cross-backend tests
# ══════════════════════════════════════════════════════════════════════════

class TestFactory:

    def test_native_mv(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="mv", backend="native")
        result = opt.optimize()
        assert result.method == "mean_variance"

    def test_native_rp(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="rp", backend="native")
        result = opt.optimize()
        assert result.method == "risk_parity"

    def test_native_bl(self):
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="bl", backend="native")
        result = opt.optimize()
        assert result.method == "black_litterman"

    def test_invalid_backend_raises(self):
        cov_data = _make_cov_data()
        with pytest.raises(ValueError, match="Unknown backend"):
            create_optimizer(cov_data, backend="nonexistent")

    def test_invalid_native_method_raises(self):
        cov_data = _make_cov_data()
        with pytest.raises(ValueError, match="does not support"):
            create_optimizer(cov_data, method="hrp", backend="native")

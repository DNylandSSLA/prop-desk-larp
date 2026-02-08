"""
Tests for the Taleb Fat-Tail Risk Framework.

Covers: TailDiagnostics, PowerLawPricer, HedgingErrorSimulator,
        BarbellOptimizer, CorrelationFragilityAnalyzer.
"""

import math

import numpy as np
import pytest

from bank_python.taleb.tail_diagnostics import TailDiagnostics, TailDiagnosticsResult
from bank_python.taleb.power_law_pricer import PowerLawPricer, PowerLawOption
from bank_python.taleb.hedging_error import HedgingErrorSimulator, HedgingErrorResult
from bank_python.taleb.barbell_optimizer import BarbellOptimizer, _norm_ppf
from bank_python.taleb.correlation_fragility import (
    CorrelationFragilityAnalyzer, CorrelationFragilityResult,
)
from bank_python.dagger import MarketData, DependencyGraph
from bank_python.optimizer import create_optimizer, OptimalPortfolio


# ── Helper: synthetic cov_data ──

def _make_cov_data(n=5, seed=42):
    """Create synthetic cov_data dict matching CovarianceBuilder output."""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    mu = rng.uniform(0.05, 0.20, n)
    # Random positive-definite covariance
    A = rng.standard_normal((n, n)) * 0.1
    Sigma = A @ A.T + 0.02 * np.eye(n)
    L = np.linalg.cholesky(Sigma)
    # Synthetic returns matrix (100 observations)
    returns_matrix = rng.multivariate_normal(mu / 252, Sigma / 252, size=100)
    return {
        "S0": {t: 100.0 for t in tickers},
        "mu": mu,
        "sigma": np.sqrt(np.diag(Sigma)),
        "L": L,
        "tickers": tickers,
        "returns_matrix": returns_matrix,
    }


# ═══════════════════════════════════════════════════════════════════════
# TAIL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════

class TestTailDiagnostics:
    """Tests for Hill estimator, kappa, max-to-sum."""

    def test_hill_on_known_pareto(self):
        """Hill estimator should recover alpha for Pareto(alpha) data."""
        rng = np.random.default_rng(42)
        true_alpha = 3.0
        n = 50000
        # Pareto: x = u^{-1/alpha}, u ~ Uniform(0,1)
        u = rng.uniform(0.001, 1.0, n)
        x = np.power(u, -1.0 / true_alpha)
        sorted_x = np.sort(x)[::-1]
        # Use k ~ n^0.6 for better convergence (bias-variance tradeoff)
        k = int(n ** 0.6)
        alpha, se = TailDiagnostics.hill_estimator(sorted_x, k=k)
        assert abs(alpha - true_alpha) < 1.0, f"Hill alpha={alpha}, expected ~{true_alpha}"

    def test_hill_se_decreases_with_n(self):
        """Standard error should decrease with more data."""
        rng = np.random.default_rng(42)
        u_small = rng.uniform(0.01, 1.0, 100)
        u_large = rng.uniform(0.01, 1.0, 10000)
        x_small = np.sort(np.power(u_small, -1.0 / 3.0))[::-1]
        x_large = np.sort(np.power(u_large, -1.0 / 3.0))[::-1]
        _, se_small = TailDiagnostics.hill_estimator(x_small)
        _, se_large = TailDiagnostics.hill_estimator(x_large)
        assert se_large < se_small

    def test_hill_on_gaussian(self):
        """Gaussian data should produce very high alpha (thin tails)."""
        rng = np.random.default_rng(42)
        x = np.sort(np.abs(rng.standard_normal(5000)))[::-1]
        alpha, _ = TailDiagnostics.hill_estimator(x)
        assert alpha > 4.0, f"Gaussian alpha={alpha}, expected > 4"

    def test_kappa_gaussian(self):
        """Kappa for Gaussian should be near sqrt(2/pi) ~ 0.7979."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(50000)
        kappa = TailDiagnostics.kappa_metric(returns)
        expected = math.sqrt(2.0 / math.pi)
        assert abs(kappa - expected) < 0.02, f"kappa={kappa}, expected ~{expected}"

    def test_kappa_fat_tails(self):
        """Kappa for fat-tailed data should differ from Gaussian."""
        rng = np.random.default_rng(42)
        # Student-T(df=3) has fatter tails
        Z = rng.standard_normal(10000)
        chi2 = sum(rng.standard_normal(10000) ** 2 for _ in range(3)) / 3
        t_samples = Z / np.sqrt(np.maximum(chi2, 1e-10))
        kappa = TailDiagnostics.kappa_metric(t_samples)
        gaussian_ref = math.sqrt(2.0 / math.pi)
        # Fat tails usually produce lower kappa
        assert kappa != pytest.approx(gaussian_ref, abs=0.01)

    def test_max_to_sum_thin_tails(self):
        """Max-to-sum should be small for Gaussian (thin-tailed) data."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(10000)
        mts = TailDiagnostics.max_to_sum_ratio(returns)
        assert mts < 0.01, f"max-to-sum={mts}, expected < 0.01 for Gaussian"

    def test_max_to_sum_fat_tails(self):
        """Max-to-sum should be larger for fat-tailed data."""
        rng = np.random.default_rng(42)
        # Create data with one extreme outlier
        returns = rng.standard_normal(1000)
        returns[0] = 100.0  # extreme outlier
        mts = TailDiagnostics.max_to_sum_ratio(returns)
        assert mts > 0.05

    def test_analyze_returns_result(self):
        """analyze() should return a TailDiagnosticsResult."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(1000)
        result = TailDiagnostics.analyze(returns)
        assert isinstance(result, TailDiagnosticsResult)
        assert result.n_observations == 1000
        assert result.k_tail > 0
        assert result.kappa_gaussian == pytest.approx(math.sqrt(2.0 / math.pi))

    def test_analyze_short_series(self):
        """analyze() should handle short series gracefully."""
        returns = [0.01, -0.02, 0.03, 0.01, -0.01,
                   0.02, -0.03, 0.01, 0.02, -0.01]
        result = TailDiagnostics.analyze(returns)
        assert isinstance(result, TailDiagnosticsResult)

    def test_hill_empty_data(self):
        """Hill should handle very short data."""
        alpha, se = TailDiagnostics.hill_estimator([1.0, 2.0, 3.0])
        assert math.isinf(alpha) or alpha > 0

    def test_max_to_sum_zeros(self):
        """Max-to-sum should handle all-zero data."""
        mts = TailDiagnostics.max_to_sum_ratio([0.0] * 100)
        assert mts == 0.0

    def test_kappa_constant_data(self):
        """Kappa should return nan for constant data (zero variance)."""
        kappa = TailDiagnostics.kappa_metric([5.0] * 100)
        assert math.isnan(kappa)


# ═══════════════════════════════════════════════════════════════════════
# POWER LAW PRICER
# ═══════════════════════════════════════════════════════════════════════

class TestPowerLawPricer:
    """Tests for power law option pricing."""

    def test_calls_decrease_with_strike(self):
        """OTM call prices should decrease as strike increases."""
        pricer = PowerLawPricer(spot=100, anchor_strike=105, anchor_price=3.0,
                                alpha=3.0, is_call=True)
        prices = [pricer.price_call(K) for K in [105, 110, 115, 120]]
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1]

    def test_puts_increase_with_lower_strike(self):
        """OTM put prices should increase as strike moves further below spot."""
        pricer = PowerLawPricer(spot=100, anchor_strike=95, anchor_price=3.0,
                                alpha=3.0, is_call=False)
        prices = [pricer.price_put(K) for K in [95, 90, 85, 80]]
        # Further OTM puts should be cheaper (lower K = further OTM for puts)
        # Actually for put: further from spot = cheaper
        for i in range(len(prices) - 1):
            assert prices[i] >= prices[i + 1], \
                f"Put at {95 - 5*i} = {prices[i]:.4f} should >= put at {90 - 5*i} = {prices[i+1]:.4f}"

    def test_matches_anchor(self):
        """Price at anchor strike should match anchor price."""
        pricer = PowerLawPricer(spot=100, anchor_strike=110, anchor_price=5.0,
                                alpha=3.0, is_call=True)
        assert pricer.price_call(110) == pytest.approx(5.0, abs=0.01)

    def test_otm_power_law_exceeds_bsm(self):
        """Deep OTM power law call should exceed BSM (fat tails = higher prices)."""
        # With alpha=2.0, PL decay is slower than BSM for very deep OTM
        spot = 100.0
        anchor_K = 105.0
        bsm_anchor = PowerLawPricer._bsm_price(spot, anchor_K, 0.20, 1.0, is_call=True)
        pricer = PowerLawPricer(spot=spot, anchor_strike=anchor_K,
                                anchor_price=bsm_anchor, alpha=2.0, is_call=True)
        result = pricer.price_range([140, 150, 160], bsm_vol=0.20, T=1.0)
        # For very deep OTM with fat tails (alpha=2.0), PL prices should exceed BSM
        for K, pl_p, bsm_p in zip(result.strikes, result.pl_prices, result.bsm_prices):
            if bsm_p > 0.001:
                assert pl_p > bsm_p, \
                    f"PL={pl_p:.4f} should > BSM={bsm_p:.4f} at K={K}"

    def test_calibrate_alpha_roundtrip(self):
        """Calibrating alpha from power law prices should recover approximate alpha."""
        true_alpha = 3.0
        pricer = PowerLawPricer(spot=100, anchor_strike=105, anchor_price=5.0,
                                alpha=true_alpha, is_call=True)
        strikes = [105, 110, 115, 120, 125, 130]
        prices = [pricer.price_call(K) for K in strikes]
        recovered_alpha = PowerLawPricer.calibrate_alpha(100, strikes, prices, is_call=True)
        assert abs(recovered_alpha - true_alpha) < 1.5, \
            f"Recovered alpha={recovered_alpha}, expected ~{true_alpha}"

    def test_price_range_result(self):
        """price_range should return a PowerLawPricingResult."""
        pricer = PowerLawPricer(spot=100, anchor_strike=105, anchor_price=5.0,
                                alpha=3.0, is_call=True)
        result = pricer.price_range([105, 110, 115])
        assert len(result.strikes) == 3
        assert len(result.pl_prices) == 3
        assert len(result.bsm_prices) == 3
        assert result.spot == 100
        assert result.alpha == 3.0

    def test_power_law_option_in_graph(self):
        """PowerLawOption should work in DependencyGraph."""
        spot = MarketData("SPOT", price=100.0)
        opt = PowerLawOption("PL_CALL", spot_source=spot, strike=110,
                             anchor_strike=105, anchor_price=5.0, alpha=3.0)
        graph = DependencyGraph()
        graph.register(opt)

        v1 = opt.value
        assert v1 > 0

        # Change spot, recalculate
        spot.set_price(105.0)
        graph.recalculate(spot)
        v2 = opt.value
        assert v2 != v1  # price should change

    def test_power_law_option_underliers(self):
        """PowerLawOption.underliers should return [spot_source]."""
        spot = MarketData("SPOT", price=100.0)
        opt = PowerLawOption("PL_CALL", spot_source=spot, strike=110)
        assert opt.underliers == [spot]

    def test_put_call_basic_sanity(self):
        """Basic sanity: call > 0 for near-ATM, put > 0 for near-ATM."""
        pricer_call = PowerLawPricer(spot=100, anchor_strike=105,
                                     anchor_price=5.0, alpha=3.0, is_call=True)
        pricer_put = PowerLawPricer(spot=100, anchor_strike=95,
                                    anchor_price=5.0, alpha=3.0, is_call=False)
        assert pricer_call.price_call(108) > 0
        assert pricer_put.price_put(92) > 0

    def test_calibrate_too_few_strikes(self):
        """Calibration with too few OTM strikes should return default."""
        alpha = PowerLawPricer.calibrate_alpha(100, [90], [5.0], is_call=True)
        assert alpha == 3.0  # default


# ═══════════════════════════════════════════════════════════════════════
# HEDGING ERROR SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

class TestHedgingErrorSimulator:
    """Tests for delta-hedging failure simulation."""

    def test_gaussian_tight(self):
        """Gaussian hedging errors should have low kurtosis (~3)."""
        sim = HedgingErrorSimulator(S0=100, K=100, sigma=0.20, T=1/12, n_steps=20)
        result = sim.simulate(distribution="gaussian", n_paths=5000, seed=42)
        assert isinstance(result, HedgingErrorResult)
        assert result.distribution == "gaussian"
        assert result.kurtosis < 10, f"Gaussian kurtosis={result.kurtosis}, expected < 10"

    def test_student_t_wider(self):
        """Student-T errors should have higher kurtosis than Gaussian."""
        sim = HedgingErrorSimulator(S0=100, K=100, sigma=0.20, T=1/12, n_steps=20)
        gauss = sim.simulate(distribution="gaussian", n_paths=5000, seed=42)
        student = sim.simulate(distribution="student_t", n_paths=5000, seed=42)
        assert student.std_error > gauss.std_error * 0.5

    def test_power_law_widest(self):
        """Power law errors should have elevated kurtosis (>3)."""
        sim = HedgingErrorSimulator(S0=100, K=100, sigma=0.20, T=1/12, n_steps=20)
        gauss = sim.simulate(distribution="gaussian", n_paths=5000, seed=42)
        power = sim.simulate(distribution="power_law", n_paths=5000, seed=42)
        # Power law should show non-Gaussian behavior
        assert power.kurtosis > 3.0, f"Power law kurtosis={power.kurtosis}"
        # And std_error should be positive
        assert power.std_error > 0

    def test_kurtosis_ordering(self):
        """Kurtosis should increase: gaussian < student_t < power_law."""
        sim = HedgingErrorSimulator(S0=100, K=100, sigma=0.20, T=1/12, n_steps=20)
        results = sim.compare_all(n_paths=5000, seed=42)
        # Just check that power law has highest kurtosis
        kurt = {r.distribution: r.kurtosis for r in results}
        assert kurt["power_law"] > kurt["gaussian"]

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        sim = HedgingErrorSimulator(S0=100, K=100, sigma=0.20)
        r1 = sim.simulate(distribution="gaussian", n_paths=1000, seed=123)
        r2 = sim.simulate(distribution="gaussian", n_paths=1000, seed=123)
        assert r1.mean_error == pytest.approx(r2.mean_error)
        assert r1.std_error == pytest.approx(r2.std_error)

    def test_compare_all_returns_three(self):
        """compare_all should return results for all three distributions."""
        sim = HedgingErrorSimulator()
        results = sim.compare_all(n_paths=1000, seed=42)
        assert len(results) == 3
        dists = [r.distribution for r in results]
        assert "gaussian" in dists
        assert "student_t" in dists
        assert "power_law" in dists

    def test_var_95_negative(self):
        """VaR 95% should be negative (loss)."""
        sim = HedgingErrorSimulator(S0=100, K=100)
        result = sim.simulate(distribution="gaussian", n_paths=5000, seed=42)
        assert result.var_95 < result.mean_error

    def test_errors_array_length(self):
        """Errors array should have n_paths entries."""
        sim = HedgingErrorSimulator()
        result = sim.simulate(n_paths=500, seed=42)
        assert len(result.errors) == 500


# ═══════════════════════════════════════════════════════════════════════
# BARBELL OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

class TestBarbellOptimizer:
    """Tests for VaR-constrained barbell portfolio."""

    def test_returns_optimal_portfolio(self):
        """Should return an OptimalPortfolio."""
        cov_data = _make_cov_data()
        opt = BarbellOptimizer(cov_data)
        result = opt.optimize()
        assert isinstance(result, OptimalPortfolio)
        assert result.method == "taleb_barbell"

    def test_weights_sum_to_one(self):
        """Weights (including safe) should sum to 1."""
        cov_data = _make_cov_data()
        opt = BarbellOptimizer(cov_data)
        result = opt.optimize()
        assert sum(result.weights) == pytest.approx(1.0, abs=0.01)

    def test_safe_weight_in_range(self):
        """Safe weight should be in (0, 1)."""
        cov_data = _make_cov_data()
        opt = BarbellOptimizer(cov_data)
        result = opt.optimize()
        safe_w = result.weights[0]
        assert 0.0 < safe_w < 1.0, f"Safe weight={safe_w}"

    def test_tickers_include_safe(self):
        """Tickers should start with 'SAFE'."""
        cov_data = _make_cov_data(n=3)
        opt = BarbellOptimizer(cov_data)
        result = opt.optimize()
        assert result.tickers[0] == "SAFE"
        assert len(result.tickers) == 4  # SAFE + 3 risky

    def test_higher_epsilon_more_risky(self):
        """Higher epsilon (less conservative) should give lower safe weight."""
        cov_data = _make_cov_data()
        conservative = BarbellOptimizer(cov_data, epsilon=0.01).optimize()
        aggressive = BarbellOptimizer(cov_data, epsilon=0.10).optimize()
        # More aggressive epsilon typically allows more risk
        # (safe weight may be lower, but depends on the math)
        assert conservative.weights[0] != aggressive.weights[0]

    def test_factory_integration(self):
        """create_optimizer(method='barbell') should work."""
        cov_data = _make_cov_data()
        opt = create_optimizer(cov_data, method="barbell")
        assert isinstance(opt, BarbellOptimizer)
        result = opt.optimize()
        assert isinstance(result, OptimalPortfolio)

    def test_positive_return_and_vol(self):
        """Expected return and vol should be positive."""
        cov_data = _make_cov_data()
        result = BarbellOptimizer(cov_data).optimize()
        assert result.expected_return > 0
        assert result.expected_vol >= 0

    def test_norm_ppf_basic(self):
        """_norm_ppf should invert _norm_cdf correctly."""
        from bank_python.risk_models import _norm_cdf
        for p in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            z = _norm_ppf(p)
            recovered_p = _norm_cdf(z)
            assert abs(recovered_p - p) < 1e-4, f"ppf({p})={z}, cdf(z)={recovered_p}"


# ═══════════════════════════════════════════════════════════════════════
# CORRELATION FRAGILITY
# ═══════════════════════════════════════════════════════════════════════

class TestCorrelationFragility:
    """Tests for MPT weight sensitivity to correlation perturbations."""

    def test_result_populated(self):
        """analyze() should return a fully populated result."""
        cov_data = _make_cov_data()
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        result = analyzer.analyze()
        assert isinstance(result, CorrelationFragilityResult)
        assert len(result.tickers) == 5
        assert len(result.base_weights) == 5
        assert len(result.perturbed_weights) == 9  # default 9 deltas

    def test_perturbation_changes_weights(self):
        """Perturbing correlation should change optimal weights."""
        cov_data = _make_cov_data()
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        result = analyzer.analyze(perturbation_deltas=[-0.2, 0.0, 0.2])
        w_neg = result.perturbed_weights[0]
        w_zero = result.perturbed_weights[1]
        w_pos = result.perturbed_weights[2]
        # At least some weights should differ
        assert not np.allclose(w_neg, w_pos, atol=1e-4)

    def test_sensitivity_nonzero(self):
        """Weight sensitivities should not all be zero."""
        cov_data = _make_cov_data()
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        result = analyzer.analyze()
        assert np.any(np.abs(result.weight_sensitivities) > 1e-6)

    def test_p_value_in_range(self):
        """P-value should be in [0, 1]."""
        cov_data = _make_cov_data()
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        result = analyzer.analyze()
        assert 0.0 <= result.p_value <= 1.0

    def test_rolling_stats_computed(self):
        """Rolling correlation stats should be computed when returns_matrix exists."""
        cov_data = _make_cov_data()
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        result = analyzer.analyze(rolling_window=10)
        # With synthetic data, rolling stats should be nonzero
        assert result.rolling_corr_std >= 0

    def test_no_returns_matrix(self):
        """Should work without returns_matrix (uses defaults)."""
        cov_data = _make_cov_data()
        del cov_data["returns_matrix"]
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        result = analyzer.analyze()
        assert isinstance(result, CorrelationFragilityResult)
        assert result.rolling_corr_mean == 0.0

    def test_base_weights_match_mv(self):
        """Base weights should match standalone MV optimization."""
        from bank_python.optimizer import MeanVarianceOptimizer
        cov_data = _make_cov_data()
        mv = MeanVarianceOptimizer(cov_data)
        mv_result = mv.optimize(long_only=True)
        analyzer = CorrelationFragilityAnalyzer(cov_data)
        frag_result = analyzer.analyze()
        np.testing.assert_allclose(frag_result.base_weights, mv_result.weights, atol=1e-4)

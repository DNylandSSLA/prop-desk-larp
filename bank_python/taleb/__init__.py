"""
Taleb Fat-Tail Risk Framework — Based on Part VII of
"Statistical Consequences of Fat Tails" (Taleb, 3rd ed 2025).

Core thesis: BSM dynamic hedging fails under fat tails. Options should be
priced by expectation (power law scaling), not BSM. Portfolios should use
VaR/CVaR constraints (barbell), not variance (Markowitz). Correlation is
unreliable for portfolio construction.

Modules:
    tail_diagnostics      — Hill estimator, kappa, max-to-sum ratio
    power_law_pricer      — Power law option pricing + DependencyGraph instrument
    hedging_error         — Delta-hedging failure simulation
    barbell_optimizer     — VaR/CVaR-constrained barbell portfolio
    correlation_fragility — MPT instability analysis
    rendering             — Rich panels for all components
"""

from .tail_diagnostics import TailDiagnostics, TailDiagnosticsResult
from .power_law_pricer import PowerLawPricer, PowerLawPricingResult, PowerLawOption
from .hedging_error import HedgingErrorSimulator, HedgingErrorResult
from .barbell_optimizer import BarbellOptimizer
from .correlation_fragility import CorrelationFragilityAnalyzer, CorrelationFragilityResult

__all__ = [
    "TailDiagnostics", "TailDiagnosticsResult",
    "PowerLawPricer", "PowerLawPricingResult", "PowerLawOption",
    "HedgingErrorSimulator", "HedgingErrorResult",
    "BarbellOptimizer",
    "CorrelationFragilityAnalyzer", "CorrelationFragilityResult",
]

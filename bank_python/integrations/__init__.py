"""
Integrations — Optional wrappers around PyPortfolioOpt, Riskfolio-Lib, and QuantLib.

All imports are conditional; the rest of bank_python works without these installed.
Use HAS_PYPFOPT / HAS_RISKFOLIO / HAS_QUANTLIB flags to check availability at runtime.
"""

# ── PyPortfolioOpt ────────────────────────────────────────────────────────
HAS_PYPFOPT = False
try:
    import pypfopt  # noqa: F401
    HAS_PYPFOPT = True
except ImportError:
    pass

# ── Riskfolio-Lib ─────────────────────────────────────────────────────────
HAS_RISKFOLIO = False
try:
    import riskfolio  # noqa: F401
    HAS_RISKFOLIO = True
except ImportError:
    pass

# ── QuantLib ──────────────────────────────────────────────────────────────
HAS_QUANTLIB = False
try:
    import QuantLib  # noqa: F401
    HAS_QUANTLIB = True
except ImportError:
    pass


def require_pypfopt():
    """Raise ImportError with a helpful message if PyPortfolioOpt is missing."""
    if not HAS_PYPFOPT:
        raise ImportError(
            "PyPortfolioOpt is required for this feature. "
            "Install it with: uv pip install pyportfolioopt"
        )


def require_riskfolio():
    """Raise ImportError with a helpful message if Riskfolio-Lib is missing."""
    if not HAS_RISKFOLIO:
        raise ImportError(
            "Riskfolio-Lib is required for this feature. "
            "Install it with: uv pip install riskfolio-lib"
        )


def require_quantlib():
    """Raise ImportError with a helpful message if QuantLib is missing."""
    if not HAS_QUANTLIB:
        raise ImportError(
            "QuantLib is required for this feature. "
            "Install it with: uv pip install QuantLib"
        )


# ── Conditional re-exports ───────────────────────────────────────────────

if HAS_PYPFOPT:
    from .pypfopt_optimizer import (
        PyPfOptMeanVariance,
        PyPfOptHRP,
        PyPfOptBlackLitterman,
    )

if HAS_RISKFOLIO:
    from .riskfolio_optimizer import (
        RiskfolioCVaROptimizer,
        RiskfolioRiskBudgeting,
    )

if HAS_QUANTLIB:
    from .quantlib_pricing import (
        QuantLibOption,
        QuantLibBond,
        QuantLibCDS,
    )

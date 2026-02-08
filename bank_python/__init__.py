"""
bank_python — A simplified implementation of the four core components
found inside investment bank proprietary Python ecosystems.

Components:
- MnTable: Column-oriented table backed by in-memory SQLite
- Barbara: Hierarchical key-value store with ring-based namespaces
- Dagger: Reactive dependency graph for instrument pricing
- Walpole: Job orchestrator with scheduling and dependencies
- Risk Models: Heston, Merton, vol surface, P&L attribution
- Trading Engine: Order management, execution, signals, audit trail
- Optimizer: Mean-variance, risk parity, Black-Litterman, frontier
- Backtester: Historical replay, strategies, tearsheets
"""

from .mntable import Table, LazyView
from .barbara import BarbaraDB, Ring
from .dagger import (
    Instrument,
    MarketData,
    Bond,
    CreditDefaultSwap,
    Option,
    Position,
    Book,
    DependencyGraph,
    CycleError,
)
from .walpole import JobRunner, JobConfig, JobMode, JobStatus
from .market_data import (
    Equity,
    FXRate,
    MarketDataManager,
    MarketDataSnapshot,
    MarketDataFetcher,
    HistoricalStore,
    MarketDataCache,
)
from .risk_models import (
    HestonProcess,
    MertonJumpDiffusion,
    VolSurface,
    PnLAttribution,
    PnLSnapshot,
)
from .trading_engine import (
    Order,
    Signal,
    ExecutionModel,
    PositionLimits,
    OrderBook,
    SignalGenerator,
    MomentumSignal,
    VolArbSignal,
    StatArbSignal,
    MacroSignal,
    AuditTrail,
    TradingEngine,
)
from .optimizer import (
    OptimalPortfolio,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    BlackLittermanModel,
    EfficientFrontier,
    Rebalancer,
    create_optimizer,
)
from .backtester import (
    HistoricalDataLoader,
    BacktestContext,
    BacktestStrategy,
    MomentumBacktest,
    VolArbBacktest,
    StatArbBacktest,
    MacroBacktest,
    EquityCurve,
    Tearsheet,
    BacktestResult,
    Backtester,
)

__all__ = [
    # MnTable
    "Table", "LazyView",
    # Barbara
    "BarbaraDB", "Ring",
    # Dagger
    "Instrument", "MarketData", "Bond", "CreditDefaultSwap", "Option",
    "Position", "Book", "DependencyGraph", "CycleError",
    # Walpole
    "JobRunner", "JobConfig", "JobMode", "JobStatus",
    # Market Data
    "Equity", "FXRate", "MarketDataManager", "MarketDataSnapshot",
    "MarketDataFetcher", "HistoricalStore", "MarketDataCache",
    # Risk Models
    "HestonProcess", "MertonJumpDiffusion", "VolSurface",
    "PnLAttribution", "PnLSnapshot",
    # Trading Engine
    "Order", "Signal", "ExecutionModel", "PositionLimits", "OrderBook",
    "SignalGenerator", "MomentumSignal", "VolArbSignal", "StatArbSignal",
    "MacroSignal", "AuditTrail", "TradingEngine",
    # Optimizer
    "OptimalPortfolio", "MeanVarianceOptimizer", "RiskParityOptimizer",
    "BlackLittermanModel", "EfficientFrontier", "Rebalancer", "create_optimizer",
    # Backtester
    "HistoricalDataLoader", "BacktestContext", "BacktestStrategy",
    "MomentumBacktest", "VolArbBacktest", "StatArbBacktest", "MacroBacktest",
    "EquityCurve", "Tearsheet", "BacktestResult", "Backtester",
]

# ── Conditional integration exports ──────────────────────────────────────
try:
    from .integrations import HAS_PYPFOPT, HAS_RISKFOLIO, HAS_QUANTLIB
    if HAS_PYPFOPT:
        from .integrations import PyPfOptMeanVariance, PyPfOptHRP, PyPfOptBlackLitterman
        __all__ += ["PyPfOptMeanVariance", "PyPfOptHRP", "PyPfOptBlackLitterman"]
    if HAS_RISKFOLIO:
        from .integrations import RiskfolioCVaROptimizer, RiskfolioRiskBudgeting
        __all__ += ["RiskfolioCVaROptimizer", "RiskfolioRiskBudgeting"]
    if HAS_QUANTLIB:
        from .integrations import QuantLibOption, QuantLibBond, QuantLibCDS
        __all__ += ["QuantLibOption", "QuantLibBond", "QuantLibCDS"]
except ImportError:
    pass

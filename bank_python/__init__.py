"""
bank_python — A simplified implementation of the four core components
found inside investment bank proprietary Python ecosystems.

Components:
- MnTable: Column-oriented table backed by in-memory SQLite
- Barbara: Hierarchical key-value store with ring-based namespaces
- Dagger: Reactive dependency graph for instrument pricing
- Walpole: Job orchestrator with scheduling and dependencies
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

__all__ = [
    "Table", "LazyView",
    "BarbaraDB", "Ring",
    "Instrument", "MarketData", "Bond", "CreditDefaultSwap", "Option",
    "Position", "Book", "DependencyGraph", "CycleError",
    "JobRunner", "JobConfig", "JobMode", "JobStatus",
    "Equity", "FXRate", "MarketDataManager", "MarketDataSnapshot",
    "MarketDataFetcher", "HistoricalStore", "MarketDataCache",
]

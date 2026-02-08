# Bank Python

A from-scratch quantitative trading platform inspired by [Cal Paterson's article](https://calpaterson.com/bank-python.html) about the strange, proprietary Python ecosystems inside investment banks.

Built on four core engines — a reactive dependency graph, a hierarchical key-value store, lazy column-oriented tables, and a job orchestrator — then layered with live market data, Monte Carlo risk, multi-strategy trading, portfolio optimization, backtesting, and a Taleb fat-tail framework. No scipy, no cvxpy. Projected gradient descent for mean-variance, bisection for implied vol, Cholesky for correlated paths — everything from numpy and math.

One optional dependency for live data (`yfinance`). Three optional integrations for comparison benchmarks (`PyPortfolioOpt`, `Riskfolio-Lib`, `QuantLib`). Core platform works without any of them.

```
14 traders  ·  180+ equities  ·  6 FX pairs  ·  options, bonds, CDS
3 MC models  ·  8 stress scenarios  ·  314 tests  ·  ~14,500 lines
```

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USER/bank-python.git
cd bank-python
uv pip install -r requirements.txt   # or: pip install -r requirements.txt

# Run the 23-scene showcase
uv run python showcase.py             # Rich panels, color, pauses
uv run python showcase.py --plain     # No color, single pass
uv run python showcase.py --dense 5   # JSON lines, 5 iterations of intraday dynamics

# Run tests
uv run python -m pytest tests/ -v     # 314 tests

# Live trading desk dashboard
uv run python prop_desk.py --live      # 5-panel Rich dashboard, refreshes every 2s
```

---

## Architecture

```
showcase.py / demo.py / prop_desk.py           ← Entry points
         │
         ├── mc_engine.py                      ← Monte Carlo pipeline
         │     GBM · Heston · Merton
         │     VaR/CVaR · Stress · Greeks
         │
         ├── trade_tracker.py                  ← Talebian options tracker
         ├── trade_analysis.py                 ← Retrospective trade analysis
         │
         └── bank_python/                      ← Core package
               ├── mntable.py                  Column-oriented tables (SQLite-backed)
               ├── barbara.py                  Hierarchical KV store (ring namespaces)
               ├── dagger.py                   Reactive dependency graph
               ├── walpole.py                  Job orchestrator (periodic, one-shot, service)
               ├── market_data.py              yfinance integration + Equity/FXRate
               ├── risk_models.py              Heston, Merton, vol surface, P&L attribution
               ├── trading_engine.py           Signals → orders → fills → audit trail
               ├── optimizer.py                MV, risk parity, Black-Litterman, frontier
               ├── backtester.py               Historical replay + strategy tearsheets
               ├── integrations.py             PyPortfolioOpt, Riskfolio-Lib, QuantLib
               └── taleb/                      Fat-tail risk framework (Ch 24-30)
                     ├── tail_diagnostics.py   Hill estimator, kappa, max-to-sum
                     ├── power_law_pricer.py   Power law option pricing
                     ├── hedging_error.py      Delta-hedging failure simulation
                     ├── barbell_optimizer.py   VaR-constrained barbell portfolio
                     ├── correlation_fragility.py  MPT instability analysis
                     └── rendering.py          Rich panels for all components
```

---

## The Four Engines

### MnTable — Column-Oriented Tables

SQLite-backed lazy tables with relational algebra. Queries compile to SQL and only execute on iteration.

```python
from bank_python import Table

t = Table([("counterparty", str), ("quantity", float), ("price", float)])
t.extend([
    {"counterparty": "GS",  "quantity": 1000, "price": 104.33},
    {"counterparty": "JPM", "quantity": 500,  "price": 104.33},
    {"counterparty": "GS",  "quantity": 300,  "price": 12.95},
])

# Lazy views — SQL only runs on iteration
gs_trades = t.restrict(counterparty="GS").project("quantity", "price")
totals = t.aggregate("counterparty", {"quantity": "sum"})
```

### Barbara — Hierarchical Key-Value Store

Ring-based namespace overlay backed by SQLite. Reads cascade through rings, writes go to the first.

```python
from bank_python import BarbaraDB

db = BarbaraDB.open("trading_desk;default")
db["/Instruments/VODA_BOND"] = bond_obj
db["/Config/currency"] = "USD"

# Desk-level values shadow defaults
keys = db.keys("/Instruments/")
```

### Dagger — Reactive Dependency Graph

When a leaf node changes, dirty flags propagate via BFS and all dependent instruments recompute.

```python
from bank_python import MarketData, Bond, Option, DependencyGraph

sofr = MarketData("SOFR", price=0.05)
bond = Bond("VODA_BOND", rate_source=sofr, face=100, coupon_rate=0.06, maturity=5)

graph = DependencyGraph()
graph.register(bond)          # auto-registers underliers

sofr.set_price(0.07)          # shock rates
graph.recalculate(sofr)       # bond revalues automatically
```

Built-in instruments: `Bond`, `CreditDefaultSwap`, `Option`, `Equity`, `FXRate`, `Position`, `Book`, `PowerLawOption`.

### Walpole — Job Orchestrator

Background thread scheduler with dependencies, retries, and three modes.

```python
from bank_python import JobRunner, JobConfig, JobMode

runner = JobRunner(barbara=db)
runner.add_job(JobConfig(
    name="market_data", callable=fetch_prices,
    mode=JobMode.PERIODIC, interval=5.0,
))
runner.add_job(JobConfig(
    name="revalue", callable=revalue_book,
    mode=JobMode.PERIODIC, interval=10.0,
    depends_on=["market_data"],
))
runner.start()
```

---

## Market Data

Live equity/FX prices from Yahoo Finance. SOFR and credit spreads simulated (not available on Yahoo).

```python
from bank_python import MarketData, Equity, MarketDataManager, DependencyGraph, BarbaraDB

graph = DependencyGraph()
db = BarbaraDB.open("desk;default")

aapl_spot = MarketData("AAPL_SPOT", price=0.0)
eq = Equity("AAPL", spot_source=aapl_spot)
graph.register(eq)

mgr = MarketDataManager(graph, db)
mgr.register_ticker("AAPL", aapl_spot)
mgr.update_all()              # fetches live price, updates graph
print(eq.value)               # real AAPL price

mgr.load_history("AAPL", period="1mo")
for row in mgr.history.query("AAPL"):
    print(row)                # OHLCV rows from MnTable
```

---

## Risk Models

Hand-rolled stochastic processes, vol surfaces, and P&L attribution — no QuantLib required.

- **HestonProcess** — stochastic volatility with mean-reversion and leverage
- **MertonJumpDiffusion** — Poisson jumps + normal diffusion for crash modeling
- **VolSurface** — strike/tenor grid with SVI parameterization, sticky-strike interpolation
- **PnLAttribution** — decompose daily P&L into delta, gamma, vega, theta, rho, unexplained

---

## Trading Engine

Signal generation, order management, position limits, and full audit trail.

```python
from bank_python import TradingEngine, MomentumSignal, PositionLimits, ExecutionModel

engine = TradingEngine(
    execution=ExecutionModel(slippage_bps=2.0),
    limits=PositionLimits(max_position=1e6, max_concentration=0.4, book_value=1e7),
)

signal = MomentumSignal(tickers=["AAPL", "MSFT", "NVDA"], lookback=20)
orders = signal.generate(market_data)
for order in orders:
    fill = engine.submit(order)
```

Four signal generators: `MomentumSignal`, `StatArbSignal`, `MacroSignal`, `VolArbSignal`.

---

## Portfolio Optimization

All from scratch — projected gradient descent for mean-variance, bisection for implied vol, eigenvalue clipping for PSD projection.

```python
from bank_python import create_optimizer

# Mean-variance
opt = create_optimizer("mean_variance", cov_data, risk_aversion=2.0)
result = opt.optimize()   # → OptimalPortfolio(weights, tickers, return, vol, sharpe)

# Risk parity
opt = create_optimizer("risk_parity", cov_data)

# Black-Litterman with views
opt = create_optimizer("black_litterman", cov_data, views={"AAPL": 0.15, "TSLA": -0.05})

# Taleb barbell (VaR-constrained safe/risky split)
opt = create_optimizer("barbell", cov_data, epsilon=0.05, max_loss_pct=0.10)
```

Also: `EfficientFrontier` (trace the frontier), `Rebalancer` (drift-based rebalancing).

---

## Backtester

Historical replay engine with per-bar strategy execution and tearsheet generation.

```python
from bank_python import Backtester, MomentumBacktest

bt = Backtester(
    strategy=MomentumBacktest(tickers=["AAPL", "MSFT", "NVDA"], lookback=20),
    start="2023-01-01", end="2024-01-01",
)
result = bt.run()
tearsheet = result.tearsheet()   # sharpe, max_dd, calmar, win_rate, avg_trade, ...
```

Four strategies: `MomentumBacktest`, `VolArbBacktest`, `StatArbBacktest`, `MacroBacktest`.

---

## Monte Carlo Engine

Full MC pipeline with correlated multi-asset simulation, three models, and 8 stress scenarios.

```python
from mc_engine import MonteCarloEngine, MCConfig

engine = MonteCarloEngine(traders, graph, db)
results = engine.run(MCConfig(
    n_paths=100_000,
    model="heston",        # or "gbm", "merton"
    random_seed=42,
))

# results: VaR/CVaR at 95%/99%, per-trader risk contribution,
#          stress scenario P&L, Greeks validation
```

**Models:** GBM (correlated log-normal), Heston (stochastic vol), Merton (jump-diffusion)

**Stress scenarios:** 2008 GFC, COVID crash, rate shock, vol spike, USD rally, tech crash, stagflation, credit crunch

---

## Taleb Fat-Tail Framework

Based on Part VII of *Statistical Consequences of Fat Tails* (Taleb, 3rd ed 2025). The core thesis: BSM dynamic hedging fails under fat tails, and Markowitz portfolio theory breaks when correlations are unreliable.

### Tail Diagnostics

```python
from bank_python import TailDiagnostics
import numpy as np

returns = np.random.standard_t(df=3, size=5000) * 0.02
result = TailDiagnostics.analyze(returns)
# result.alpha    → Hill tail index (< 4 means fat tails)
# result.kappa    → kappa metric (< sqrt(2/pi) means fat tails)
# result.max_to_sum → max-to-sum ratio (> 0 in limit means infinite variance)
```

### Power Law Option Pricing

Options priced by power law scaling instead of BSM. Only needs one anchor option price + tail index. No vol surface, no dynamic hedging assumption.

```python
from bank_python import PowerLawPricer

pricer = PowerLawPricer(spot=100, anchor_strike=105, anchor_price=3.50, alpha=2.5)
price = pricer.price_call(120)          # OTM call via power law
result = pricer.price_range(            # compare across strikes
    strikes=[100, 105, 110, 120, 130, 140, 150],
    bsm_vol=0.20,
)
# result.pl_prices vs result.bsm_prices — power law > BSM for deep OTM
```

### Hedging Error Simulation

Demonstrates that delta hedging fails under fat tails.

```python
from bank_python import HedgingErrorSimulator

sim = HedgingErrorSimulator(spot=100, strike=100, vol=0.20, T=1.0, n_steps=252)
results = sim.compare_all(n_paths=50_000, seed=42)
# Gaussian: tight errors (kurtosis ~3)
# Student-T: wider (kurtosis >> 3)
# Power law: explosive
```

### Barbell Portfolio

Replace Markowitz variance constraint with VaR/CVaR. The optimal portfolio is a barbell: safe numeraire + risky basket.

```python
from bank_python import BarbellOptimizer

opt = BarbellOptimizer(cov_data, risk_free_rate=0.05, epsilon=0.05, max_loss_pct=0.10)
result = opt.optimize()
# result.tickers  → ["SAFE", "AAPL", "MSFT", ...]
# result.weights  → [0.25, 0.12, 0.08, ...]  (safe + risky basket)
```

### Correlation Fragility

Shows that mean-variance portfolios are hypersensitive to correlation estimates.

```python
from bank_python import CorrelationFragilityAnalyzer

analyzer = CorrelationFragilityAnalyzer(cov_data)
result = analyzer.analyze(perturbation_deltas=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
# result.weight_sensitivity  → how much weights shift per unit correlation change
# result.rolling_corr_std    → empirical instability of rolling correlations
# result.delta_statistic     → chi-squared test for correlation stationarity
```

---

## Optional Integrations

Three third-party libraries for comparison benchmarks. Each is auto-detected at import time.

| Library | What it adds |
|---------|-------------|
| `PyPortfolioOpt` | HRP (Hierarchical Risk Parity), mean-variance, Black-Litterman |
| `Riskfolio-Lib` | CVaR minimization, risk budgeting |
| `QuantLib` | Option/bond/CDS pricing for comparison against hand-rolled models |

```bash
uv pip install pyportfolioopt riskfolio-lib QuantLib   # all optional
```

Showcase scenes for these libraries appear only when installed.

---

## Showcase

The 23-scene showcase is the main demo. It wires up all 14 traders with live market data, runs every component, and renders Rich panels.

```
 1. DESK OVERVIEW                          14. EFFICIENT FRONTIER
 2. MONTE CARLO — GBM                      15. PORTFOLIO REBALANCE
 3. MONTE CARLO — Heston                   16. BACKTEST — Momentum
 4. MONTE CARLO — Merton                   17. BACKTEST — All Strategies
 5. STRESS TEST SCENARIOS                  18. TALEB: TAIL DIAGNOSTICS
 6. TRADING SIGNALS                        19. TALEB: POWER LAW PRICING
 7. ORDER EXECUTION                        20. TALEB: HEDGING ERRORS
 8. PORTFOLIO OPT — Mean-Variance          21. TALEB: BARBELL PORTFOLIO
 9. PORTFOLIO OPT — Risk Parity            22. TALEB: CORRELATION FRAGILITY
10. PORTFOLIO OPT — Black-Litterman        23. TEARSHEET
11. PORTFOLIO OPT — HRP (PyPortfolioOpt)
12. PORTFOLIO OPT — CVaR (Riskfolio-Lib)
13. QUANTLIB PRICING COMPARISON
```

**Dense mode** (`--dense N`) outputs machine-readable JSON lines and runs N iterations of intraday market dynamics — correlated equity moves, VIX mean-reversion, SOFR drift, credit spread blowouts, dynamic option vol scaling.

---

## Prop Desk

`prop_desk.py` is a live trading desk dashboard with 14 concurrent traders.

| Trader | Strategy | Focus |
|--------|----------|-------|
| Alice | momentum | Tech/semis trend following |
| Bob | vol_arb | Short options premium, delta hedged |
| Charlie | stat_arb | Mean-reversion pairs (GOOGL/AMZN, JPM/GS) |
| Diana | macro | Rates, FX, broad indices |
| Nero | credit_arb | CDS basis, HY exposure |
| Tony | tech_momentum | Heavy conviction longs |
| Vivian | gamma_scalp | Long straddles, delta hedged |
| Marcus | pairs_trade | Sector pairs (GOOGL/MSFT, XOM/CVX) |
| Elena | carry_trade | FX carry, yield curve |
| Raj | contrarian | Short momentum, long value |
| Sophia | index_arb | Short ETFs vs long components |
| Kenji | vol_spread | Long/short options across strikes |
| Zara | event_driven | Catalyst bets (TSLA, META, COIN) |
| Oscar | multi_asset | Diversified across all classes |

The live dashboard runs 7 Walpole background jobs: market data fetch, book revaluation, risk checks, Barbara snapshots, MC simulation, signal generation, and vol surface refresh.

---

## Tests

```
tests/test_mntable.py ........... 13    tests/test_optimizer.py ......... 30
tests/test_barbara.py ........... 13    tests/test_trading_engine.py .... 46
tests/test_dagger.py ............ 16    tests/test_backtester.py ........ 38
tests/test_walpole.py ...........  8    tests/test_integrations.py ...... 32
tests/test_market_data.py ....... 32    tests/test_taleb.py ............. 45
tests/test_risk_models.py ....... 41
                                        TOTAL .......................... 314
```

```bash
uv run python -m pytest tests/ -v
```

---

## What This Is (and Isn't)

This is a learning project. The goal is to capture the interesting architectural ideas from real bank Python ecosystems — data-first design, reactive computation, centralized persistent stores, layered namespaces — using normal Python conventions. It fetches real market data and does real math, but it's not production banking software. Don't trade with it.

---

## Inspired By

- [Cal Paterson — Bank Python](https://calpaterson.com/bank-python.html)
- Nassim Nicholas Taleb — *Statistical Consequences of Fat Tails* (3rd ed, 2025)

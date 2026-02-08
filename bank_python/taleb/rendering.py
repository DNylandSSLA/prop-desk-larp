"""
Rendering — Rich panels for all Taleb fat-tail components.
"""

import math

from rich.panel import Panel
from rich.table import Table as RichTable
from rich.text import Text


def render_tail_diagnostics(result, console, ticker="Portfolio"):
    """
    Render tail diagnostics as a Rich panel.

    Color-codes alpha and kappa relative to Gaussian reference values.
    """
    tbl = RichTable(
        title=f"Tail Diagnostics — {ticker}",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Metric", style="bold", width=20)
    tbl.add_column("Value", justify="right", width=12)
    tbl.add_column("Reference", justify="right", width=12)
    tbl.add_column("Assessment", width=30)

    # Hill alpha
    alpha = result.alpha
    if alpha < 2:
        alpha_style = "bold red"
        alpha_assess = "EXTREME: mean may not exist"
    elif alpha < 3:
        alpha_style = "red"
        alpha_assess = "Very fat: variance undefined"
    elif alpha < 4:
        alpha_style = "yellow"
        alpha_assess = "Fat-tailed: kurtosis undefined"
    else:
        alpha_style = "green"
        alpha_assess = "Moderate tails"

    tbl.add_row(
        "Hill alpha",
        Text(f"{alpha:.2f}", style=alpha_style),
        "inf (Gaussian)",
        Text(alpha_assess, style=alpha_style),
    )
    tbl.add_row(
        "alpha SE",
        f"{result.alpha_se:.2f}",
        "",
        f"k={result.k_tail} tail obs",
    )

    # Kappa
    kappa = result.kappa
    kappa_ref = result.kappa_gaussian
    kappa_dev = abs(kappa - kappa_ref) / kappa_ref
    if kappa_dev < 0.05:
        kappa_style = "green"
        kappa_assess = "Near Gaussian"
    elif kappa_dev < 0.15:
        kappa_style = "yellow"
        kappa_assess = "Moderate deviation"
    else:
        kappa_style = "red"
        kappa_assess = "Strong non-Gaussian"

    tbl.add_row(
        "Kappa",
        Text(f"{kappa:.4f}", style=kappa_style),
        f"{kappa_ref:.4f}",
        Text(kappa_assess, style=kappa_style),
    )

    # Max-to-sum
    mts = result.max_to_sum
    if mts < 0.01:
        mts_style = "green"
        mts_assess = "Thin-tailed (converges to 0)"
    elif mts < 0.05:
        mts_style = "yellow"
        mts_assess = "Some concentration"
    else:
        mts_style = "red"
        mts_assess = "Single obs dominates sum"

    tbl.add_row(
        "Max-to-Sum",
        Text(f"{mts:.4f}", style=mts_style),
        "~0 (thin)",
        Text(mts_assess, style=mts_style),
    )

    tbl.add_row("Observations", str(result.n_observations), "", "")

    console.print(Panel(tbl, border_style="cyan"))


def render_power_law_pricing(result, console):
    """Render power law vs BSM pricing comparison."""
    tbl = RichTable(
        title=f"Power Law vs BSM Pricing (alpha={result.alpha:.1f})",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Strike", justify="right", width=10)
    tbl.add_column("PL Price", justify="right", width=12)
    tbl.add_column("BSM Price", justify="right", width=12)
    tbl.add_column("Diff %", justify="right", width=10)
    tbl.add_column("Tail Effect", width=20)

    for K, pl_p, bsm_p in zip(result.strikes, result.pl_prices, result.bsm_prices):
        if bsm_p > 0.01:
            diff_pct = (pl_p - bsm_p) / bsm_p * 100
        else:
            diff_pct = 0.0

        if diff_pct > 20:
            effect_style = "bold red"
            effect = "PL >> BSM (fat tail)"
        elif diff_pct > 5:
            effect_style = "yellow"
            effect = "PL > BSM"
        elif diff_pct < -5:
            effect_style = "dim"
            effect = "PL < BSM"
        else:
            effect_style = "green"
            effect = "Similar"

        tbl.add_row(
            f"${K:.0f}",
            f"${pl_p:.4f}",
            f"${bsm_p:.4f}",
            Text(f"{diff_pct:+.1f}%", style=effect_style),
            Text(effect, style=effect_style),
        )

    tbl.add_row("", "", "", "", "")
    tbl.add_row(
        Text("Anchor", style="bold"),
        f"${result.anchor_strike:.0f}",
        f"${result.anchor_price:.4f}",
        "",
        f"Spot: ${result.spot:.2f}",
    )

    console.print(Panel(tbl, border_style="cyan"))


def render_hedging_errors(results, console):
    """Render hedging error comparison across distributions."""
    tbl = RichTable(
        title="Delta Hedging Errors — BSM Hedge Under Different Tails",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Distribution", style="bold", width=14)
    tbl.add_column("Mean Error", justify="right", width=12)
    tbl.add_column("Std Error", justify="right", width=12)
    tbl.add_column("Kurtosis", justify="right", width=10)
    tbl.add_column("VaR 95%", justify="right", width=12)
    tbl.add_column("Paths", justify="right", width=8)

    for r in results:
        if r.kurtosis > 10:
            kurt_style = "bold red"
        elif r.kurtosis > 5:
            kurt_style = "yellow"
        else:
            kurt_style = "green"

        tbl.add_row(
            r.distribution,
            f"${r.mean_error:+.2f}",
            f"${r.std_error:.2f}",
            Text(f"{r.kurtosis:.1f}", style=kurt_style),
            f"${r.var_95:+.2f}",
            str(r.n_paths),
        )

    console.print(Panel(tbl, border_style="cyan",
                        subtitle="Kurtosis >>3 = BSM hedge failure"))


def render_barbell_portfolio(portfolio, console):
    """Render barbell portfolio with safe/risky split."""
    tbl = RichTable(
        title="Taleb Barbell Portfolio — VaR-Constrained",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Component", style="bold", width=12)
    tbl.add_column("Weight", justify="right", width=10)
    tbl.add_column("Role", width=20)

    safe_weight = portfolio.weights[0]
    risky_weight = 1.0 - safe_weight

    tbl.add_row(
        Text("SAFE", style="bold green"),
        Text(f"{safe_weight:.1%}", style="green"),
        "Risk-free numeraire",
    )
    tbl.add_row("", "", "")
    tbl.add_row(
        Text("RISKY", style="bold yellow"),
        Text(f"{risky_weight:.1%}", style="yellow"),
        "Max-Sharpe basket:",
    )

    for i in range(1, len(portfolio.tickers)):
        w = portfolio.weights[i]
        if w > 0.001:
            tbl.add_row(
                f"  {portfolio.tickers[i]}",
                f"{w:.1%}",
                "",
            )

    tbl.add_row("", "", "")
    tbl.add_row(Text("E[Return]", style="bold"), f"{portfolio.expected_return:.2%}", "")
    tbl.add_row(Text("Volatility", style="bold"), f"{portfolio.expected_vol:.2%}", "")
    tbl.add_row(Text("Sharpe", style="bold"), f"{portfolio.sharpe_ratio:.2f}", "")

    console.print(Panel(tbl, border_style="cyan"))


def render_correlation_fragility(result, console):
    """Render correlation fragility analysis."""
    tbl = RichTable(
        title="Correlation Fragility — MPT Weight Sensitivity",
        expand=True,
        title_style="bold white",
        show_header=True,
        header_style="bold cyan",
    )
    tbl.add_column("Ticker", style="bold", width=8)
    tbl.add_column("Base Weight", justify="right", width=12)
    tbl.add_column("Sensitivity", justify="right", width=12)
    tbl.add_column("Fragility", width=20)

    for i, ticker in enumerate(result.tickers):
        w = result.base_weights[i]
        sens = abs(result.weight_sensitivities[i])

        if sens > 1.0:
            frag_style = "bold red"
            frag = "HIGHLY FRAGILE"
        elif sens > 0.5:
            frag_style = "yellow"
            frag = "Moderate fragility"
        elif sens > 0.1:
            frag_style = "dim"
            frag = "Mild sensitivity"
        else:
            frag_style = "green"
            frag = "Stable"

        tbl.add_row(
            ticker,
            f"{w:.1%}",
            Text(f"{result.weight_sensitivities[i]:+.3f}", style=frag_style),
            Text(frag, style=frag_style),
        )

    # Rolling correlation stats
    tbl.add_row("", "", "", "")
    tbl.add_row(
        Text("Rolling Corr", style="bold"),
        f"{result.rolling_corr_mean:.3f}",
        f"std={result.rolling_corr_std:.3f}",
        "",
    )
    tbl.add_row(
        Text("Delta Stat", style="bold"),
        f"{result.delta_statistic:.4f}",
        f"p={result.p_value:.3f}",
        "H0: constant correlation" if result.p_value > 0.05 else Text("REJECT H0", style="red"),
    )

    console.print(Panel(tbl, border_style="cyan",
                        subtitle="Sensitivity = dw/d(rho) at max perturbation"))

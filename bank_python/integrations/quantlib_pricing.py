"""
QuantLib pricing wrappers — Option, Bond, and CDS subclasses.

These extend the existing dagger.py instrument classes, overriding compute()
to use QuantLib's pricing engines while preserving the same underliers property
so that DependencyGraph reactive propagation works unchanged.
"""

import math
from datetime import datetime, timedelta

import QuantLib as ql

from bank_python.dagger import Option, Bond, CreditDefaultSwap


class QuantLibOption(Option):
    """
    European option priced via QuantLib's AnalyticEuropeanEngine (BSM).

    Extends the hand-rolled Option class with proper risk-free rate,
    dividend yield, and full Greeks computation.
    """

    def __init__(self, name, spot_source, strike=100.0, volatility=0.2,
                 time_to_expiry=1.0, is_call=True,
                 risk_free_rate=0.05, dividend_yield=0.0):
        super().__init__(name, spot_source, strike, volatility, time_to_expiry, is_call)
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self._greeks = {}

    def compute(self):
        """Reprice using QuantLib's AnalyticEuropeanEngine."""
        S = self.spot_source.value
        K = self.strike
        sigma = self.volatility
        T = self.time_to_expiry
        r = self.risk_free_rate
        q = self.dividend_yield

        if T <= 0 or sigma <= 0 or S <= 0:
            self._value = max(0, S - K) if self.is_call else max(0, K - S)
            self._greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                            "theta": 0.0, "rho": 0.0}
            return

        # QuantLib date setup
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today
        expiry_days = max(1, int(T * 365))
        expiry_date = today + ql.Period(expiry_days, ql.Days)

        # Option type
        option_type = ql.Option.Call if self.is_call else ql.Option.Put

        # Payoff and exercise
        payoff = ql.PlainVanillaPayoff(option_type, K)
        exercise = ql.EuropeanExercise(expiry_date)
        option = ql.VanillaOption(payoff, exercise)

        # Market data handles
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(r)),
                           ql.Actual365Fixed())
        )
        div_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(q)),
                           ql.Actual365Fixed())
        )
        vol_handle = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(),
                                ql.QuoteHandle(ql.SimpleQuote(sigma)),
                                ql.Actual365Fixed())
        )

        # BSM process + engine
        bsm_process = ql.BlackScholesMertonProcess(
            spot_handle, div_handle, rate_handle, vol_handle
        )
        engine = ql.AnalyticEuropeanEngine(bsm_process)
        option.setPricingEngine(engine)

        self._value = round(option.NPV(), 4)

        # Cache Greeks
        try:
            self._greeks = {
                "delta": option.delta(),
                "gamma": option.gamma(),
                "vega": option.vega() / 100.0,  # per 1% vol move
                "theta": option.theta() / 365.0,  # per-day
                "rho": option.rho() / 100.0,  # per 1% rate move
            }
        except RuntimeError:
            self._greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0,
                            "theta": 0.0, "rho": 0.0}

    def greeks(self):
        """Return cached Greeks from the last compute()."""
        if self._dirty:
            self.compute()
            self._dirty = False
        return dict(self._greeks)


class QuantLibBond(Bond):
    """
    Fixed-rate bond priced via QuantLib's DiscountingBondEngine.

    Extends the hand-rolled Bond class with proper duration and convexity
    analytics via ql.BondFunctions.
    """

    def __init__(self, name, rate_source, face=100.0, coupon_rate=0.05, maturity=5):
        super().__init__(name, rate_source, face, coupon_rate, maturity)
        self._duration = 0.0
        self._convexity = 0.0

    def compute(self):
        """Reprice using QuantLib's DiscountingBondEngine."""
        rate = self.rate_source.value

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        # Yield curve
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(rate)),
                           ql.Actual365Fixed())
        )

        # Build fixed-rate bond
        issue_date = today
        maturity_date = today + ql.Period(self.maturity, ql.Years)
        schedule = ql.Schedule(
            issue_date, maturity_date,
            ql.Period(ql.Semiannual),
            ql.NullCalendar(),
            ql.Unadjusted, ql.Unadjusted,
            ql.DateGeneration.Backward, False,
        )

        bond = ql.FixedRateBond(
            0,  # settlement days
            self.face,
            schedule,
            [self.coupon_rate],
            ql.Actual365Fixed(),
        )

        engine = ql.DiscountingBondEngine(rate_handle)
        bond.setPricingEngine(engine)

        self._value = round(bond.NPV(), 4)

        # Duration and convexity
        try:
            yield_rate = ql.InterestRate(
                rate, ql.Actual365Fixed(), ql.Compounded, ql.Semiannual
            )
            self._duration = ql.BondFunctions.duration(
                bond, yield_rate, ql.Duration.Modified
            )
            self._convexity = ql.BondFunctions.convexity(
                bond, yield_rate
            )
        except RuntimeError:
            self._duration = 0.0
            self._convexity = 0.0

    @property
    def duration(self):
        """Modified duration from the last compute()."""
        if self._dirty:
            self.compute()
            self._dirty = False
        return self._duration

    @property
    def convexity(self):
        """Convexity from the last compute()."""
        if self._dirty:
            self.compute()
            self._dirty = False
        return self._convexity


class QuantLibCDS(CreditDefaultSwap):
    """
    CDS priced via QuantLib's MidPointCdsEngine with flat hazard rate.

    Extends the hand-rolled CreditDefaultSwap class with proper
    survival probability-based pricing and standard 40% recovery.
    """

    def __init__(self, name, credit_spread_source, rate_source,
                 notional=10_000_000, maturity=5, recovery_rate=0.4):
        super().__init__(name, credit_spread_source, rate_source, notional, maturity)
        self.recovery_rate = recovery_rate

    def compute(self):
        """Reprice using QuantLib's MidPointCdsEngine."""
        spread = self.credit_spread_source.value
        rate = self.rate_source.value

        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        # Risk-free yield curve
        rate_handle = ql.YieldTermStructureHandle(
            ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(rate)),
                           ql.Actual365Fixed())
        )

        # Flat hazard rate from credit spread
        # hazard rate ~ spread / (1 - recovery)
        hazard_rate = max(spread / (1.0 - self.recovery_rate), 1e-6)
        hazard_curve = ql.DefaultProbabilityTermStructureHandle(
            ql.FlatHazardRate(today, ql.QuoteHandle(ql.SimpleQuote(hazard_rate)),
                              ql.Actual365Fixed())
        )

        # CDS contract
        maturity_date = today + ql.Period(self.maturity, ql.Years)
        schedule = ql.Schedule(
            today, maturity_date,
            ql.Period(ql.Quarterly),
            ql.NullCalendar(),
            ql.Following, ql.Following,
            ql.DateGeneration.Forward, False,
        )

        # Running spread in basis points (market convention)
        running_spread = spread  # as a decimal

        cds = ql.CreditDefaultSwap(
            ql.Protection.Buyer,
            self.notional,
            running_spread,
            schedule,
            ql.Following,
            ql.Actual365Fixed(),
        )

        engine = ql.MidPointCdsEngine(hazard_curve, self.recovery_rate, rate_handle)
        cds.setPricingEngine(engine)

        self._value = round(cds.NPV(), 2)

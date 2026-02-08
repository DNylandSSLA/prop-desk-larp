"""
Power Law Option Pricing — Based on Chapter 27 of Taleb's
"Statistical Consequences of Fat Tails".

Under power law tails, option prices scale as a power of distance
from spot. Only needs one anchor option price + tail index alpha.
No vol surface, no dynamic hedging assumption.

Call: C(K2) = ((K2 - S0)/(K1 - S0))^{1-alpha} * C(K1)  for OTM calls (K > S0)
Put:  P(K2) = P(K1) * f(K2, alpha) / f(K1, alpha)       for OTM puts (K < S0)
"""

import math
from dataclasses import dataclass, field

import numpy as np

from bank_python.dagger import Instrument
from bank_python.risk_models import _norm_cdf, _norm_pdf, VolSurface


@dataclass
class PowerLawPricingResult:
    """Result of power law pricing across strikes."""
    spot: float
    alpha: float
    strikes: list
    pl_prices: list
    bsm_prices: list
    anchor_strike: float
    anchor_price: float


class PowerLawPricer:
    """
    Power law option pricer based on Taleb Ch 27.

    Given an anchor option (strike K1, price C1) and tail index alpha,
    prices options at arbitrary strikes using power law scaling.
    """

    def __init__(self, spot, anchor_strike, anchor_price, alpha, is_call=True):
        """
        Parameters
        ----------
        spot          : float — current spot price
        anchor_strike : float — strike of the anchor option
        anchor_price  : float — market price of the anchor option
        alpha         : float — tail index (Hill estimator)
        is_call       : bool — True for calls, False for puts
        """
        self.spot = spot
        self.anchor_strike = anchor_strike
        self.anchor_price = anchor_price
        self.alpha = alpha
        self.is_call = is_call

    def price_call(self, K):
        """
        Price an OTM call at strike K using power law scaling.

        C(K) = ((K - S0)/(K1 - S0))^{1-alpha} * C(K1)
        """
        S0 = self.spot

        if K <= S0:
            # ITM call: intrinsic + time value from anchor
            return (S0 - K) + self.anchor_price

        return self._price_otm_call(K)

    def _price_otm_call(self, K):
        """Price OTM call (K > S0)."""
        S0 = self.spot
        K1 = self.anchor_strike
        C1 = self.anchor_price
        alpha = self.alpha

        dist_K = K - S0
        dist_K1 = K1 - S0

        if dist_K1 <= 0 or dist_K <= 0:
            return C1

        ratio = dist_K / dist_K1
        return C1 * ratio ** (1.0 - alpha)

    def price_put(self, K):
        """
        Price an OTM put at strike K using power law scaling.

        Uses the put formula from Ch 27 with the survival function scaling.
        """
        S0 = self.spot

        if K >= S0:
            # ITM put: intrinsic + time value from anchor
            return (K - S0) + self.anchor_price

        return self._price_otm_put(K)

    def _price_otm_put(self, K):
        """Price OTM put (K < S0)."""
        S0 = self.spot
        K1 = self.anchor_strike
        P1 = self.anchor_price
        alpha = self.alpha

        dist_K = S0 - K
        dist_K1 = S0 - K1

        if dist_K1 <= 0 or dist_K <= 0:
            return P1

        ratio = dist_K / dist_K1
        return P1 * ratio ** (1.0 - alpha)

    def price_range(self, strikes, bsm_vol=0.20, T=1.0):
        """
        Price across a range of strikes and compare to BSM.

        Parameters
        ----------
        strikes : list[float]
        bsm_vol : float — BSM volatility for comparison
        T       : float — time to expiry for BSM comparison

        Returns
        -------
        PowerLawPricingResult
        """
        pl_prices = []
        bsm_prices = []
        S0 = self.spot

        for K in strikes:
            if self.is_call:
                pl_prices.append(self.price_call(K))
                bsm_prices.append(self._bsm_price(S0, K, bsm_vol, T, is_call=True))
            else:
                pl_prices.append(self.price_put(K))
                bsm_prices.append(self._bsm_price(S0, K, bsm_vol, T, is_call=False))

        return PowerLawPricingResult(
            spot=S0,
            alpha=self.alpha,
            strikes=list(strikes),
            pl_prices=pl_prices,
            bsm_prices=bsm_prices,
            anchor_strike=self.anchor_strike,
            anchor_price=self.anchor_price,
        )

    @classmethod
    def calibrate_alpha(cls, spot, strikes, prices, is_call=True):
        """
        Fit tail index alpha from observed option chain via OLS on log-log.

        For OTM calls: ln(C) = (1-alpha)*ln(K-S) + const
        Regresses to find alpha.

        Parameters
        ----------
        spot    : float
        strikes : array-like — option strikes
        prices  : array-like — observed option prices
        is_call : bool

        Returns
        -------
        float — estimated alpha
        """
        log_dists = []
        log_prices = []

        for K, P in zip(strikes, prices):
            if P <= 0:
                continue
            if is_call and K > spot:
                dist = K - spot
                if dist > 0:
                    log_dists.append(math.log(dist))
                    log_prices.append(math.log(P))
            elif not is_call and K < spot:
                dist = spot - K
                if dist > 0:
                    log_dists.append(math.log(dist))
                    log_prices.append(math.log(P))

        if len(log_dists) < 2:
            return 3.0  # default

        # OLS: log_prices = slope * log_dists + intercept
        # slope = 1 - alpha => alpha = 1 - slope
        x = np.array(log_dists)
        y = np.array(log_prices)
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1e-15)
        alpha = 1.0 - slope
        return max(1.5, min(alpha, 10.0))  # clamp to reasonable range

    @staticmethod
    def _bsm_price(S, K, sigma, T, is_call=True):
        """Black-Scholes price for comparison."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(0, S - K) if is_call else max(0, K - S)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)

        if is_call:
            return S * nd1 - K * nd2
        else:
            return K * (1.0 - nd2) - S * (1.0 - nd1)


class PowerLawOption(Instrument):
    """
    DependencyGraph-compatible power law option.

    Recomputes when spot changes using power law pricing instead of BSM.
    """

    def __init__(self, name, spot_source, strike=100.0, anchor_strike=None,
                 anchor_price=5.0, alpha=3.0, is_call=True):
        """
        Parameters
        ----------
        name          : str
        spot_source   : MarketData — spot price source
        strike        : float — strike to price
        anchor_strike : float or None — if None, uses ATM (spot + small offset)
        anchor_price  : float — price of the anchor option
        alpha         : float — tail index
        is_call       : bool
        """
        super().__init__(name)
        self.spot_source = spot_source
        self.strike = strike
        self.anchor_strike = anchor_strike
        self.anchor_price = anchor_price
        self.alpha = alpha
        self.is_call = is_call

    @property
    def underliers(self):
        return [self.spot_source]

    def compute(self):
        S0 = self.spot_source.value
        if S0 <= 0:
            self._value = 0.0
            return

        anchor_K = self.anchor_strike
        if anchor_K is None:
            # Default anchor: slightly OTM
            offset = S0 * 0.05
            anchor_K = S0 + offset if self.is_call else S0 - offset

        pricer = PowerLawPricer(
            spot=S0,
            anchor_strike=anchor_K,
            anchor_price=self.anchor_price,
            alpha=self.alpha,
            is_call=self.is_call,
        )

        if self.is_call:
            self._value = pricer.price_call(self.strike)
        else:
            self._value = pricer.price_put(self.strike)

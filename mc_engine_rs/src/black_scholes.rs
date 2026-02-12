use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::math_utils::norm_cdf;

/// Vectorized Black-Scholes option pricing.
///
/// Parameters
/// ----------
/// spot   : 1-D array of spot prices [n_paths]
/// strike : scalar strike
/// sigma  : scalar volatility
/// t      : time to expiry (years)
/// is_call: true for call, false for put
///
/// Returns
/// -------
/// prices : 1-D array [n_paths]
#[pyfunction]
pub fn bs_price_vec<'py>(
    py: Python<'py>,
    spot: PyReadonlyArray1<'py, f64>,
    strike: f64,
    sigma: f64,
    t: f64,
    is_call: bool,
) -> Bound<'py, PyArray1<f64>> {
    let spot = spot.as_array().to_owned();
    let n = spot.len();

    let prices = py.allow_threads(|| {
        let mut out = Array1::<f64>::zeros(n);

        if t <= 0.0 || sigma <= 0.0 {
            // Intrinsic value at expiry
            for i in 0..n {
                out[i] = if is_call {
                    (spot[i] - strike).max(0.0)
                } else {
                    (strike - spot[i]).max(0.0)
                };
            }
            return out;
        }

        let sqrt_t = t.sqrt();
        let sigma_sqrt_t = sigma * sqrt_t;
        let half_sigma2_t = 0.5 * sigma * sigma * t;

        for i in 0..n {
            let d1 = ((spot[i] / strike).ln() + half_sigma2_t) / sigma_sqrt_t;
            let d2 = d1 - sigma_sqrt_t;

            let nd1 = norm_cdf(d1);
            let nd2 = norm_cdf(d2);

            out[i] = if is_call {
                spot[i] * nd1 - strike * nd2
            } else {
                strike * (1.0 - nd2) - spot[i] * (1.0 - nd1)
            };
        }
        out
    });

    prices.into_pyarray_bound(py)
}

#[cfg(test)]
mod tests {
    use crate::math_utils::norm_cdf;

    #[test]
    fn test_put_call_parity() {
        // C - P = S - K  (for r=0)
        let s: f64 = 100.0;
        let k: f64 = 100.0;
        let sigma: f64 = 0.2;
        let t: f64 = 1.0;

        let sqrt_t = t.sqrt();
        let d1 = ((s / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);

        let call = s * nd1 - k * nd2;
        let put = k * (1.0 - nd2) - s * (1.0 - nd1);

        // C - P should equal S - K = 0 for ATM with r=0
        assert!((call - put - (s - k)).abs() < 1e-10);
    }

    #[test]
    fn test_call_lower_bound() {
        // Call >= max(S - K, 0)
        let s: f64 = 120.0;
        let k: f64 = 100.0;
        let sigma: f64 = 0.3;
        let t: f64 = 0.5;

        let sqrt_t = t.sqrt();
        let d1 = ((s / k).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        let call = s * norm_cdf(d1) - k * norm_cdf(d2);

        assert!(call >= (s - k).max(0.0));
    }
}

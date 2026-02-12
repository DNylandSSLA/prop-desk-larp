use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Vectorized bond present value with incremental discount factor accumulation.
///
/// Parameters
/// ----------
/// rates       : 1-D array of discount rates [n_paths]
/// face        : face value
/// coupon_rate : annual coupon rate (e.g. 0.05 for 5%)
/// maturity    : number of annual periods
///
/// Returns
/// -------
/// pv : 1-D array [n_paths]
#[pyfunction]
pub fn bond_pv_vec<'py>(
    py: Python<'py>,
    rates: PyReadonlyArray1<'py, f64>,
    face: f64,
    coupon_rate: f64,
    maturity: usize,
) -> Bound<'py, PyArray1<f64>> {
    let rates = rates.as_array().to_owned();
    let n = rates.len();
    let coupon = face * coupon_rate;

    let pv = py.allow_threads(|| {
        let mut out = Array1::<f64>::zeros(n);

        for i in 0..n {
            let r = rates[i];
            let inv_1pr = 1.0 / (1.0 + r);
            let mut df = 1.0; // incremental discount factor
            let mut coupon_pv = 0.0;

            for _ in 0..maturity {
                df *= inv_1pr;
                coupon_pv += df;
            }
            // df is now 1/(1+r)^maturity after the loop
            out[i] = coupon * coupon_pv + face * df;
        }
        out
    });

    pv.into_pyarray_bound(py)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_par_bond() {
        // When coupon_rate == discount rate, PV should equal face value
        let face: f64 = 1000.0;
        let rate: f64 = 0.05;
        let coupon_rate: f64 = 0.05;
        let maturity = 10;
        let coupon = face * coupon_rate;

        let inv = 1.0 / (1.0 + rate);
        let mut df = 1.0;
        let mut cpv = 0.0;
        for _ in 0..maturity {
            df *= inv;
            cpv += df;
        }
        let pv = coupon * cpv + face * df;
        assert!((pv - face).abs() < 1e-8);
    }

    #[test]
    fn test_zero_coupon() {
        // Zero-coupon bond: PV = face / (1+r)^T
        let face: f64 = 1000.0;
        let rate: f64 = 0.08;
        let maturity: usize = 5;

        let expected = face / (1.0 + rate).powi(maturity as i32);

        let inv = 1.0 / (1.0 + rate);
        let mut df = 1.0;
        for _ in 0..maturity {
            df *= inv;
        }
        let pv = face * df; // coupon = 0
        assert!((pv - expected).abs() < 1e-8);
    }
}

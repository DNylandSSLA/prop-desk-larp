use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::rng_utils::{fill_standard_normal, make_rng};

/// Simulate correlated GBM terminal prices.
///
/// S_T = S_0 * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt) * (L @ Z))
///
/// Parameters
/// ----------
/// s0          : 1-D array [n_assets] — initial spot prices
/// mu          : 1-D array [n_assets] — annualized drifts
/// sigma       : 1-D array [n_assets] — annualized volatilities
/// cholesky_l  : 2-D array [n_assets, n_assets] — lower-triangular Cholesky factor
/// n_paths     : number of Monte Carlo paths
/// horizon_days: trading days to simulate
/// seed        : RNG seed for reproducibility
///
/// Returns
/// -------
/// s_terminal : 2-D array [n_paths, n_assets]
#[pyfunction]
pub fn simulate_gbm<'py>(
    py: Python<'py>,
    s0: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    sigma: PyReadonlyArray1<'py, f64>,
    cholesky_l: PyReadonlyArray2<'py, f64>,
    n_paths: usize,
    horizon_days: usize,
    seed: u64,
) -> Bound<'py, PyArray2<f64>> {
    // Copy inputs to owned data for GIL release
    let s0 = s0.as_array().to_owned();
    let mu = mu.as_array().to_owned();
    let sigma = sigma.as_array().to_owned();
    let l = cholesky_l.as_array().to_owned();
    let n_assets = s0.len();

    let result = py.allow_threads(move || {
        let mut rng = make_rng(seed);
        let dt = horizon_days as f64 / 252.0;
        let sqrt_dt = dt.sqrt();

        // Precompute drift per asset
        let mut drift = vec![0.0f64; n_assets];
        for j in 0..n_assets {
            drift[j] = (mu[j] - 0.5 * sigma[j] * sigma[j]) * dt;
        }

        let mut out = Array2::<f64>::zeros((n_paths, n_assets));

        // Buffer for independent normals per path
        let mut z_indep = vec![0.0f64; n_assets];
        // Buffer for correlated normals per path
        let mut z_corr = vec![0.0f64; n_assets];

        for i in 0..n_paths {
            // Draw independent standard normals
            fill_standard_normal(&mut rng, &mut z_indep);

            // Multiply by lower-triangular L: z_corr = L @ z_indep
            // Exploit lower-triangular: z_corr[j] = sum_{k=0..=j} L[j,k] * z_indep[k]
            for j in 0..n_assets {
                let mut sum = 0.0;
                for k in 0..=j {
                    sum += l[[j, k]] * z_indep[k];
                }
                z_corr[j] = sum;
            }

            // GBM terminal price
            for j in 0..n_assets {
                let exponent = drift[j] + sigma[j] * sqrt_dt * z_corr[j];
                out[[i, j]] = s0[j] * exponent.exp();
            }
        }
        out
    });

    result.into_pyarray_bound(py)
}

#[cfg(test)]
mod tests {
    use crate::rng_utils::make_rng;
    use numpy::ndarray;

    #[test]
    fn test_single_asset_positivity() {
        let n_paths = 10_000;
        let s0: f64 = 100.0;
        let mu: f64 = 0.05;
        let sigma: f64 = 0.3;

        let mut rng = make_rng(42);
        let dt: f64 = 10.0 / 252.0;
        let sqrt_dt = dt.sqrt();
        let drift0 = (mu - 0.5 * sigma * sigma) * dt;

        for _ in 0..n_paths {
            let z = crate::rng_utils::sample_normal(&mut rng);
            let s_t = s0 * (drift0 + sigma * sqrt_dt * z).exp();
            assert!(s_t > 0.0);
        }
    }

    #[test]
    fn test_deterministic_seed() {
        let n_paths: usize = 100;
        let n_assets: usize = 2;
        let s0 = ndarray::Array1::<f64>::from_vec(vec![100.0, 50.0]);
        let mu = ndarray::Array1::<f64>::from_vec(vec![0.05, 0.08]);
        let sigma = ndarray::Array1::<f64>::from_vec(vec![0.2, 0.3]);
        let l = ndarray::Array2::<f64>::from_shape_vec(
            (2, 2),
            vec![1.0, 0.0, 0.5, 0.866025],
        )
        .unwrap();

        let run = |seed: u64| -> ndarray::Array2<f64> {
            let mut rng = make_rng(seed);
            let dt: f64 = 5.0 / 252.0;
            let sqrt_dt = dt.sqrt();
            let mut drift = vec![0.0f64; n_assets];
            for j in 0..n_assets {
                drift[j] = (mu[j] - 0.5 * sigma[j] * sigma[j]) * dt;
            }
            let mut out = ndarray::Array2::<f64>::zeros((n_paths, n_assets));
            let mut z_indep = vec![0.0f64; n_assets];
            let mut z_corr = vec![0.0f64; n_assets];
            for i in 0..n_paths {
                crate::rng_utils::fill_standard_normal(&mut rng, &mut z_indep);
                for j in 0..n_assets {
                    let mut sum = 0.0;
                    for k in 0..=j {
                        sum += l[[j, k]] * z_indep[k];
                    }
                    z_corr[j] = sum;
                }
                for j in 0..n_assets {
                    out[[i, j]] = s0[j] * (drift[j] + sigma[j] * sqrt_dt * z_corr[j]).exp();
                }
            }
            out
        };

        let a = run(42);
        let b = run(42);
        assert_eq!(a, b);
    }
}

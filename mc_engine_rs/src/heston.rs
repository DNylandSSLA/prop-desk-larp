use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::rng_utils::{fill_standard_normal, make_rng};

/// Simulate terminal prices under the Heston stochastic volatility model.
///
/// dS = mu*S*dt + sqrt(V)*S*dW_S
/// dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
/// corr(dW_S, dW_V) = rho
///
/// Uses Euler-Maruyama with full truncation (V floored at 0 in diffusion).
///
/// Parameters
/// ----------
/// s0      : initial spot price
/// mu      : annualized drift
/// n_paths : number of Monte Carlo paths
/// n_steps : number of time steps
/// dt      : time step size (years)
/// kappa   : mean-reversion speed
/// theta   : long-run variance
/// xi      : vol-of-vol
/// rho     : correlation between spot and variance Brownians
/// v0      : initial variance
/// seed    : RNG seed
///
/// Returns
/// -------
/// (s_terminal, v_terminal) : tuple of 1-D arrays [n_paths]
#[pyfunction]
pub fn simulate_heston<'py>(
    py: Python<'py>,
    s0: f64,
    mu: f64,
    n_paths: usize,
    n_steps: usize,
    dt: f64,
    kappa: f64,
    theta: f64,
    xi: f64,
    rho: f64,
    v0: f64,
    seed: u64,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let (s_out, v_out) = py.allow_threads(move || {
        let mut rng = make_rng(seed);
        let sqrt_dt = dt.sqrt();
        let rho_comp = (1.0 - rho * rho).sqrt();

        let mut s = vec![s0; n_paths];
        let mut v = vec![v0; n_paths];

        // Pre-allocate Z buffers reused across time steps
        let mut z1 = vec![0.0f64; n_paths];
        let mut z2 = vec![0.0f64; n_paths];

        for _ in 0..n_steps {
            fill_standard_normal(&mut rng, &mut z1);
            fill_standard_normal(&mut rng, &mut z2);

            for i in 0..n_paths {
                let dw_s = z1[i] * sqrt_dt;
                let dw_v = (rho * z1[i] + rho_comp * z2[i]) * sqrt_dt;

                // Full truncation: use max(V, 0) in diffusion
                let v_pos = v[i].max(0.0);
                let sqrt_v = v_pos.sqrt();

                // Log-Euler step for S (prevents negative prices)
                s[i] *= ((mu - 0.5 * v_pos) * dt + sqrt_v * dw_s).exp();

                // Euler-Maruyama step for V
                v[i] += kappa * (theta - v[i]) * dt + xi * sqrt_v * dw_v;
            }
        }

        // Floor final variance at 0
        for vi in v.iter_mut() {
            *vi = vi.max(0.0);
        }

        let s_arr = Array1::from_vec(s);
        let v_arr = Array1::from_vec(v);
        (s_arr, v_arr)
    });

    (s_out.into_pyarray_bound(py), v_out.into_pyarray_bound(py))
}

#[cfg(test)]
mod tests {
    use crate::rng_utils::make_rng;

    #[test]
    fn test_heston_positivity() {
        let n_paths: usize = 5_000;
        let n_steps: usize = 100;
        let dt: f64 = 1.0 / 252.0 / 50.0;
        let sqrt_dt = dt.sqrt();
        let s0: f64 = 100.0;
        let mu: f64 = 0.05;
        let kappa: f64 = 1.5;
        let theta: f64 = 0.04;
        let xi: f64 = 0.5;
        let rho: f64 = -0.7;
        let v0: f64 = 0.04;
        let rho_comp = (1.0_f64 - rho * rho).sqrt();

        let mut rng = make_rng(42);
        let mut s = vec![s0; n_paths];
        let mut v = vec![v0; n_paths];
        let mut z1 = vec![0.0f64; n_paths];
        let mut z2 = vec![0.0f64; n_paths];

        for _ in 0..n_steps {
            crate::rng_utils::fill_standard_normal(&mut rng, &mut z1);
            crate::rng_utils::fill_standard_normal(&mut rng, &mut z2);
            for i in 0..n_paths {
                let dw_s = z1[i] * sqrt_dt;
                let dw_v = (rho * z1[i] + rho_comp * z2[i]) * sqrt_dt;
                let v_pos = v[i].max(0.0);
                let sqrt_v = v_pos.sqrt();
                s[i] *= ((mu - 0.5 * v_pos) * dt + sqrt_v * dw_s).exp();
                v[i] += kappa * (theta - v[i]) * dt + xi * sqrt_v * dw_v;
            }
        }

        for &si in &s {
            assert!(si > 0.0, "Heston prices must be positive");
        }
        for vi in &v {
            // After flooring, should be >= 0
            assert!(vi.max(0.0) >= 0.0);
        }
    }
}

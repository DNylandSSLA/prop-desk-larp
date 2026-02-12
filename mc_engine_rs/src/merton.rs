use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::rng_utils::{make_rng, sample_normal, sample_poisson};

/// Simulate terminal prices under Merton jump-diffusion.
///
/// dS/S = (mu - lambda*m)*dt + sigma*dW + J*dN
/// N ~ Poisson(lambda*dt), J ~ LogNormal(mu_j, sigma_j)
/// m = E[e^J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
///
/// Per-path Poisson draws + compound jump accumulation.
///
/// Parameters
/// ----------
/// s0             : initial spot price
/// mu             : annualized drift
/// sigma          : annualized diffusion volatility
/// n_paths        : number of paths
/// dt             : total time horizon (years)
/// jump_intensity : expected jumps per year (lambda)
/// jump_mean      : mean of log-jump size (mu_j)
/// jump_vol       : vol of log-jump size (sigma_j)
/// seed           : RNG seed
///
/// Returns
/// -------
/// s_terminal : 1-D array [n_paths]
#[pyfunction]
pub fn simulate_merton<'py>(
    py: Python<'py>,
    s0: f64,
    mu: f64,
    sigma: f64,
    n_paths: usize,
    dt: f64,
    jump_intensity: f64,
    jump_mean: f64,
    jump_vol: f64,
    seed: u64,
) -> Bound<'py, PyArray1<f64>> {
    let result = py.allow_threads(move || {
        let mut rng = make_rng(seed);

        // Compensator: m = E[e^J - 1]
        let m = (jump_mean + 0.5 * jump_vol * jump_vol).exp() - 1.0;
        let drift = (mu - 0.5 * sigma * sigma - jump_intensity * m) * dt;
        let diffusion_scale = sigma * dt.sqrt();
        let lambda_dt = jump_intensity * dt;

        let mut out = Array1::<f64>::zeros(n_paths);

        for i in 0..n_paths {
            let z = sample_normal(&mut rng);
            let diffusion = diffusion_scale * z;

            // Poisson draw for jump count
            let n_jumps = sample_poisson(&mut rng, lambda_dt);

            // Compound jump: sum of n_jumps log-normal draws
            let mut jump_log_sum = 0.0;
            for _ in 0..n_jumps {
                let zj = sample_normal(&mut rng);
                jump_log_sum += jump_mean + jump_vol * zj;
            }

            out[i] = s0 * (drift + diffusion + jump_log_sum).exp();
        }
        out
    });

    result.into_pyarray_bound(py)
}

#[cfg(test)]
mod tests {
    use crate::rng_utils::{make_rng, sample_normal, sample_poisson};

    #[test]
    fn test_merton_positivity() {
        let n_paths: usize = 10_000;
        let s0: f64 = 100.0;
        let mu: f64 = 0.05;
        let sigma: f64 = 0.2;
        let dt: f64 = 10.0 / 252.0;
        let jump_intensity: f64 = 1.0;
        let jump_mean: f64 = -0.05;
        let jump_vol: f64 = 0.10;

        let m = (jump_mean + 0.5 * jump_vol * jump_vol).exp() - 1.0;
        let drift = (mu - 0.5 * sigma * sigma - jump_intensity * m) * dt;
        let diffusion_scale = sigma * dt.sqrt();
        let lambda_dt = jump_intensity * dt;

        let mut rng = make_rng(42);

        for _ in 0..n_paths {
            let z = sample_normal(&mut rng);
            let diffusion = diffusion_scale * z;
            let n_jumps = sample_poisson(&mut rng, lambda_dt);
            let mut jump_log_sum = 0.0;
            for _ in 0..n_jumps {
                let zj = sample_normal(&mut rng);
                jump_log_sum += jump_mean + jump_vol * zj;
            }
            let s_t = s0 * (drift + diffusion + jump_log_sum).exp();
            assert!(s_t > 0.0, "Merton prices must be positive");
        }
    }

    #[test]
    fn test_merton_deterministic() {
        let mut rng1 = make_rng(99);
        let mut rng2 = make_rng(99);

        let mut vals1 = Vec::new();
        let mut vals2 = Vec::new();
        for _ in 0..100 {
            let z1 = sample_normal(&mut rng1);
            let n1 = sample_poisson(&mut rng1, 2.0);
            vals1.push((z1, n1));

            let z2 = sample_normal(&mut rng2);
            let n2 = sample_poisson(&mut rng2, 2.0);
            vals2.push((z2, n2));
        }
        assert_eq!(vals1, vals2);
    }
}

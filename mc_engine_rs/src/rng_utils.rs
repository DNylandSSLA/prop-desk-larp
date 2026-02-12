use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Create a seeded ChaCha8 generator for reproducible Monte Carlo.
pub fn make_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// Fill a pre-allocated slice with standard normal draws (Box-Muller).
pub fn fill_standard_normal(rng: &mut ChaCha8Rng, buf: &mut [f64]) {
    let n = buf.len();
    let mut i = 0;

    // Box-Muller generates pairs
    while i + 1 < n {
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let r = (-2.0 * u1.ln()).sqrt();
        buf[i] = r * u2.cos();
        buf[i + 1] = r * u2.sin();
        i += 2;
    }

    // Handle odd length
    if i < n {
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        buf[i] = (-2.0 * u1.ln()).sqrt() * u2.cos();
    }
}

/// Sample a single standard normal.
#[inline]
pub fn sample_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

/// Sample from Poisson(lambda) using Knuth's algorithm.
/// Suitable for small to moderate lambda (< ~30).
pub fn sample_poisson(rng: &mut ChaCha8Rng, lambda: f64) -> u64 {
    let l = (-lambda).exp();
    let mut k: u64 = 0;
    let mut p: f64 = 1.0;

    loop {
        k += 1;
        let u: f64 = rng.gen();
        p *= u;
        if p <= l {
            return k - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducible() {
        let mut rng1 = make_rng(42);
        let mut rng2 = make_rng(42);
        let mut buf1 = vec![0.0; 100];
        let mut buf2 = vec![0.0; 100];
        fill_standard_normal(&mut rng1, &mut buf1);
        fill_standard_normal(&mut rng2, &mut buf2);
        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut rng1 = make_rng(1);
        let mut rng2 = make_rng(2);
        let mut buf1 = vec![0.0; 100];
        let mut buf2 = vec![0.0; 100];
        fill_standard_normal(&mut rng1, &mut buf1);
        fill_standard_normal(&mut rng2, &mut buf2);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_normal_statistics() {
        // Mean should be ~0, stddev ~1 with enough samples
        let mut rng = make_rng(123);
        let n = 100_000;
        let mut buf = vec![0.0; n];
        fill_standard_normal(&mut rng, &mut buf);

        let mean: f64 = buf.iter().sum::<f64>() / n as f64;
        let variance: f64 = buf.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!(mean.abs() < 0.02, "mean = {mean}");
        assert!((variance - 1.0).abs() < 0.05, "variance = {variance}");
    }

    #[test]
    fn test_poisson_mean() {
        let mut rng = make_rng(42);
        let lambda = 3.0;
        let n = 50_000;
        let sum: f64 = (0..n).map(|_| sample_poisson(&mut rng, lambda) as f64).sum();
        let mean = sum / n as f64;
        assert!((mean - lambda).abs() < 0.1, "poisson mean = {mean}");
    }

    #[test]
    fn test_odd_length_buffer() {
        let mut rng = make_rng(42);
        let mut buf = vec![0.0; 7]; // odd
        fill_standard_normal(&mut rng, &mut buf);
        // All values should be finite
        for &v in &buf {
            assert!(v.is_finite());
        }
    }
}

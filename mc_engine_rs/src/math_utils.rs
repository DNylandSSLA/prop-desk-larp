/// Abramowitz & Stegun approximation to the error function.
/// Maximum error ~1.5e-7, sufficient for Monte Carlo pricing.
pub fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // Constants from Abramowitz & Stegun 7.1.26
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let result = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();
    sign * result
}

/// Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

/// Standard normal PDF: phi(x) = exp(-x^2/2) / sqrt(2*pi)
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_values() {
        // erf(0) = 0 (A&S approximation has ~1e-9 residual at 0)
        assert!((erf(0.0)).abs() < 2e-7);
        // erf(1) ≈ 0.8427007929
        assert!((erf(1.0) - 0.8427007929).abs() < 2e-7);
        // erf(-1) = -erf(1)
        assert!((erf(-1.0) + erf(1.0)).abs() < 1e-10);
        // erf(3) ≈ 0.9999779
        assert!((erf(3.0) - 0.9999779).abs() < 2e-7);
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        // Phi(0) = 0.5 (inherits A&S approximation error)
        assert!((norm_cdf(0.0) - 0.5).abs() < 2e-7);
        // Phi(x) + Phi(-x) = 1
        for &x in &[0.5, 1.0, 2.0, 3.0] {
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() < 2e-7);
        }
    }

    #[test]
    fn test_norm_pdf_peak() {
        // phi(0) = 1/sqrt(2*pi) ≈ 0.39894228
        assert!((norm_pdf(0.0) - 0.39894228).abs() < 1e-7);
        // phi is symmetric
        assert!((norm_pdf(1.0) - norm_pdf(-1.0)).abs() < 1e-15);
    }
}

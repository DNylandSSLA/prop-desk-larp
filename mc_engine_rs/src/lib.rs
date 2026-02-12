use pyo3::prelude::*;

mod black_scholes;
mod bond;
mod gbm;
mod heston;
mod math_utils;
mod merton;
mod rng_utils;

#[pymodule]
fn mc_engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gbm::simulate_gbm, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes::bs_price_vec, m)?)?;
    m.add_function(wrap_pyfunction!(bond::bond_pv_vec, m)?)?;
    m.add_function(wrap_pyfunction!(heston::simulate_heston, m)?)?;
    m.add_function(wrap_pyfunction!(merton::simulate_merton, m)?)?;
    Ok(())
}

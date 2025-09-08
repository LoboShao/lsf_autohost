use pyo3::prelude::*;

pub mod job;
pub mod host;
pub mod event;
pub mod env;

#[cfg(test)]
// Run all tests: cargo test --package lsf_env_rust --lib
// Run tests matching pattern: cargo test --package lsf_env_rust --lib multihost
// Run with output: cargo test --package lsf_env_rust --lib multihost -- --nocapture
#[path = "testing/test_multihost_internal.rs"]
mod test_multihost_internal;

use env::ClusterSchedulerEnv;

#[pymodule]
fn lsf_env_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ClusterSchedulerEnv>()?;
    Ok(())
}
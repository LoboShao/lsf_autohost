use pyo3::prelude::*;

pub mod job;
pub mod host;
pub mod event;
pub mod env;  // Keep old env as backup
pub mod base_env;
pub mod host_sorting_env;

use env::ClusterSchedulerEnv;
use host_sorting_env::HostSortingEnv;

#[pymodule]
fn lsf_env_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    // Keep old env for compatibility
    m.add_class::<ClusterSchedulerEnv>()?;
    // Add new modular environments
    m.add_class::<HostSortingEnv>()?;
    Ok(())
}
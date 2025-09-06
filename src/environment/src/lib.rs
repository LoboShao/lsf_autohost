use pyo3::prelude::*;

pub mod job;
pub mod host;
pub mod event;
pub mod env;

use env::ClusterSchedulerEnv;

#[pymodule]
fn lsf_env_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<ClusterSchedulerEnv>()?;
    Ok(())
}
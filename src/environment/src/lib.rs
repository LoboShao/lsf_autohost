use pyo3::prelude::*;

pub mod job;
pub mod host;
pub mod event;
pub mod base_env;
pub mod host_sorting_env;
pub mod job_ordering_env;

use host_sorting_env::HostSortingEnv;
use job_ordering_env::JobOrderingEnv;

#[pymodule]
fn lsf_env_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<HostSortingEnv>()?;
    m.add_class::<JobOrderingEnv>()?;
    Ok(())
}
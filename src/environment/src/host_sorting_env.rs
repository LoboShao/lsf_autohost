use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::ops::{Deref, DerefMut};

use crate::base_env::{BaseClusterEnv, ClusterConfig};
use crate::job::Job;

/// Host sorting environment - agent provides priorities for hosts
#[pyclass]
pub struct HostSortingEnv {
    base: BaseClusterEnv,

    // Cached state vector for performance
    cached_state: Vec<f32>,
}

// Deref to BaseClusterEnv allows calling base methods directly
impl Deref for HostSortingEnv {
    type Target = BaseClusterEnv;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for HostSortingEnv {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

#[pymethods]
impl HostSortingEnv {
    // ==================== Constructor ====================

    #[new]
    #[pyo3(signature = (
        num_hosts = 1000,
        max_queue_length = None,
        host_cores_range = (32, 128),
        host_memory_range = (131072, 524288),
        job_cores_range = (1, 32),
        job_memory_range = (2048, 65536),
        job_duration_range = (1, 60),
        max_jobs_per_step = 50,
        max_time = 4096,
        use_skewed_arrivals = false,
        seed = None
    ))]
    fn new(
        num_hosts: usize,
        max_queue_length: Option<usize>,
        host_cores_range: (u32, u32),
        host_memory_range: (u32, u32),
        job_cores_range: (u32, u32),
        job_memory_range: (u32, u32),
        job_duration_range: (u32, u32),
        max_jobs_per_step: usize,
        max_time: usize,
        use_skewed_arrivals: bool,
        seed: Option<u64>,
    ) -> Self {
        let config = ClusterConfig::new(
            num_hosts,
            max_queue_length,
            host_cores_range,
            host_memory_range,
            job_cores_range,
            job_memory_range,
            job_duration_range,
            max_jobs_per_step,
            max_time,
            use_skewed_arrivals,
            seed,
        );
        Self::from_config(&config)
    }

    // ==================== Main RL Interface ====================

    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.reset_base();
        self.add_new_jobs_to_queue();
        self.get_state(py)
    }

    pub fn step(&mut self, py: Python, action: &PyAny) -> PyResult<(Py<PyArray1<f32>>, f32, bool, PyObject)> {
        self.maybe_start_batch();

        let host_priorities = self.parse_action(action)?;

        let finishing_batch_now = if let Some(bucket_idx) = self.select_next_bucket() {
            let (_jobs_scheduled, finishing) = self.process_selected_bucket(bucket_idx, &host_priorities);
            finishing
        } else {
            false
        };

        self.update_environment_state(finishing_batch_now);
        let reward = self.calculate_step_reward(finishing_batch_now);
        self.maybe_advance_time(finishing_batch_now);
        let done = self.check_episode_done();

        Ok((self.get_state(py)?, reward, done, self.base.get_step_info(py)?))
    }

    pub fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.skip_empty_buckets();
        self.cached_state.fill(0.0);

        // Fill host features (2 per host) using historical utilization from previous cycle
        // This simulates real LSF behavior with cycle-level snapshots
        for i in 0..self.num_hosts {
            let idx = i * 2;
            // Use historical utilization (from previous cycle) instead of real-time
            let core_util = self.hosts[i].get_core_utilization();
            let mem_util = self.hosts[i].get_memory_utilization();
            // Convert utilization to availability ratio (0-1)
            self.cached_state[idx] = 1.0 - core_util;
            self.cached_state[idx + 1] = 1.0 - mem_util;
        }

        // Fill job features (8 features) - clone to avoid borrow conflict
        if let Some(job) = self.base.get_current_job().cloned() {
            self.fill_job_features(&job, self.num_hosts * 2);
        }

        Ok(PyArray1::from_slice(py, &self.cached_state).to_owned())
    }

    // ==================== Info & Metrics ====================

    pub fn needs_decision(&self) -> bool {
        self.get_current_job().is_some()
    }

    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_step_info(py)
    }

    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_metrics(py)
    }

    pub fn get_host_configs(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_host_configs(py)
    }

    pub fn get_job_schedule(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_job_schedule(py)
    }

    pub fn get_cluster_info(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_cluster_info(py)
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.base.set_random_seed(seed);
    }
}

// ==================== Private Implementation ====================
impl HostSortingEnv {
    /// Create HostSortingEnv from config
    pub fn from_config(config: &ClusterConfig) -> Self {
        let base = BaseClusterEnv::from_config(config);
        let state_size = config.num_hosts * 2 + 8;
        let cached_state = vec![0.0; state_size];

        HostSortingEnv {
            base,
            cached_state,
        }
    }

    /// Parse and validate action array from Python
    fn parse_action(&self, action: &PyAny) -> PyResult<Vec<f64>> {
        let action_f64: Vec<f64> = if let Ok(arr32) = action.extract::<PyReadonlyArray1<f32>>() {
            arr32.as_slice()?.iter().map(|&x| x as f64).collect()
        } else if let Ok(arr64) = action.extract::<PyReadonlyArray1<f64>>() {
            arr64.as_slice()?.to_vec()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "`action` must be a 1-D numpy array of float32 or float64",
            ));
        };

        if action_f64.len() != self.num_hosts {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Action must have {} elements (one per host), got {}", self.num_hosts, action_f64.len())
            ));
        }

        Ok(action_f64)
    }

    /// Select next bucket to process (FCFS for HostSortingEnv)
    fn select_next_bucket(&mut self) -> Option<usize> {
        self.skip_empty_buckets();
        if self.current_bucket_index < self.job_buckets.len() {
            Some(self.current_bucket_index)
        } else {
            None
        }
    }

    /// Fill job and batch features into cached state (8 features)
    fn fill_job_features(&mut self, job: &Job, start_idx: usize) {
        // Basic job properties (normalized)
        self.cached_state[start_idx] = job.cores_required as f32 / self.job_cores_range.1 as f32;
        self.cached_state[start_idx + 1] = job.memory_required as f32 / self.job_memory_range.1 as f32;
        self.cached_state[start_idx + 2] = job.duration as f32 / self.job_duration_range.1 as f32;

        // Is deferred (binary: 1 if waiting > 1s)
        let waiting_time = self.current_time as f64 - job.submission_time;
        self.cached_state[start_idx + 3] = if waiting_time > 1.0 { 1.0 } else { 0.0 };

        // Batch progress (tanh normalized)
        let batch_scale = self.num_hosts as f32;
        self.cached_state[start_idx + 4] = (self.jobs_processed_in_batch as f32 / batch_scale).tanh();

        // Queue pressure (tanh normalized)
        let jobs_in_cycle = if self.total_jobs_in_current_batch > 0 {
            self.total_jobs_in_current_batch
        } else {
            self.job_queue.len() + self.job_buckets.iter().map(|b| b.jobs.len()).sum::<usize>()
        };
        self.cached_state[start_idx + 5] = (jobs_in_cycle as f32 / batch_scale).tanh();

        // Resource pressure (cores and memory)
        let (cores_needed, memory_needed) = if self.total_jobs_in_current_batch > 0 {
            (self.total_cores_in_current_batch, self.total_memory_in_current_batch)
        } else {
            let mut cores = 0u32;
            let mut memory = 0u32;
            for q_job in &self.job_queue {
                cores += q_job.cores_required;
                memory += q_job.memory_required;
            }
            for bucket in &self.job_buckets {
                for q_job in &bucket.jobs {
                    cores += q_job.cores_required;
                    memory += q_job.memory_required;
                }
            }
            (cores, memory)
        };
        self.cached_state[start_idx + 6] = cores_needed as f32 / self.total_cluster_cores as f32;
        self.cached_state[start_idx + 7] = memory_needed as f32 / self.total_cluster_memory as f32;
    }

    /// Calculate reward for this step
    fn calculate_step_reward(&self, batch_complete: bool) -> f32 {
        if batch_complete {
            self.calculate_pure_resource_utilization_reward()
        } else {
            -0.01
        }
    }

    /// Calculate resource utilization reward
    fn calculate_pure_resource_utilization_reward(&self) -> f32 {
        let mut rewards = 0.0;
        for i in 0..self.num_hosts {
            let core_util = self.host_core_utils[i] as f64;
            let memory_util = self.host_memory_utils[i] as f64;
            rewards += (core_util + memory_util) / 2.0;
        }
        (rewards / self.num_hosts as f64) as f32
    }
}
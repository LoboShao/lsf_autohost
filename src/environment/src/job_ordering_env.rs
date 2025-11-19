use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use std::ops::{Deref, DerefMut};

use crate::base_env::{BaseClusterEnv, ClusterConfig};

/// Job ordering environment - agent provides priorities for job buckets
/// Uses heuristic (first-available) for host selection
#[pyclass]
pub struct JobOrderingEnv {
    base: BaseClusterEnv,

    // Maximum number of buckets for fixed-size state/action space
    max_buckets: usize,

    // Cached state vector for performance
    cached_state: Vec<f32>,
}

// Deref to BaseClusterEnv allows calling base methods directly
impl Deref for JobOrderingEnv {
    type Target = BaseClusterEnv;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for JobOrderingEnv {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

#[pymethods]
impl JobOrderingEnv {
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
        max_buckets = 100,
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
        max_buckets: usize,
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
        Self::from_config(&config, max_buckets)
    }

    // ==================== Main RL Interface ====================

    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.reset_base();
        self.add_new_jobs_to_queue();
        self.get_state(py)
    }

    pub fn step(&mut self, py: Python, action: &PyAny) -> PyResult<(Py<PyArray1<f32>>, f32, bool, PyObject)> {
        self.maybe_start_batch();

        let bucket_priorities = self.parse_action(action)?;

        let finishing_batch_now = if let Some(bucket_idx) = self.select_next_bucket(&bucket_priorities) {
            // Use first-available heuristic for host selection
            let host_priorities = self.get_first_available_host_priorities();
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

        // Fill bucket features (2 features per bucket, up to max_buckets)
        // Format: [bucket0_cores, bucket0_count, bucket1_cores, bucket1_count, ...]
        let num_buckets = self.job_buckets.len().min(self.max_buckets);
        for i in 0..num_buckets {
            let start_idx = i * 2;
            self.fill_bucket_features(i, start_idx);
        }

        // Fill global cluster state (last 2 features)
        // Uses cycle-level snapshot (cached at batch start) to simulate real LSF
        let global_idx = self.max_buckets * 2;
        self.fill_global_state(global_idx);

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

    pub fn get_max_buckets(&self) -> usize {
        self.max_buckets
    }
}

// ==================== Private Implementation ====================
impl JobOrderingEnv {
    /// Create JobOrderingEnv from config
    pub fn from_config(config: &ClusterConfig, max_buckets: usize) -> Self {
        let base = BaseClusterEnv::from_config(config);
        // State size: max_buckets * 2 (cores + count per bucket) + 2 global features (available cores/mem ratio)
        let state_size = max_buckets * 2 + 2;
        let cached_state = vec![0.0; state_size];

        JobOrderingEnv {
            base,
            max_buckets,
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

        if action_f64.len() != self.max_buckets {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Action must have {} elements (max_buckets), got {}", self.max_buckets, action_f64.len())
            ));
        }

        Ok(action_f64)
    }

    /// Select next bucket to process based on agent priorities
    /// Returns the index of the highest-priority non-empty bucket that can be scheduled
    /// Swaps the selected bucket to current_bucket_index position for processing
    fn select_next_bucket(&mut self, bucket_priorities: &[f64]) -> Option<usize> {
        if self.current_bucket_index >= self.job_buckets.len() {
            return None;
        }

        // Find the highest priority non-empty bucket from remaining buckets
        // Only consider priorities for actual buckets (first k), not padding
        let mut best_local_idx = None;
        let mut best_priority = f64::NEG_INFINITY;

        for local_i in 0..(self.job_buckets.len() - self.current_bucket_index) {
            let global_i = self.current_bucket_index + local_i;

            if !self.job_buckets[global_i].jobs.is_empty() {
                // Only use priorities for actual buckets (ignore padding beyond actual count)
                // This prevents agent from learning meaningless priorities for non-existent buckets
                let priority = bucket_priorities[global_i]; // Use priority at position matching actual bucket index

                // Check if first job in this bucket can potentially be scheduled
                if let Some(first_job) = self.job_buckets[global_i].jobs.front() {
                    // Check if ANY host can accommodate this job's requirements
                    let can_schedule = self.hosts.iter().any(|host| {
                        host.total_cores >= first_job.cores_required
                        && host.total_memory >= first_job.memory_required
                    });

                    if can_schedule && priority > best_priority {
                        best_priority = priority;
                        best_local_idx = Some(local_i);
                    }
                }
            }
        }

        if let Some(local_idx) = best_local_idx {
            // Swap the selected bucket to current position
            let current_idx = self.current_bucket_index;
            if local_idx != 0 {
                let swap_idx = current_idx + local_idx;
                self.job_buckets.swap(current_idx, swap_idx);
            }
            Some(current_idx)
        } else {
            None
        }
    }

    /// Get host priorities using first-available heuristic
    /// Returns vector of priorities (higher = more available resources)
    fn get_first_available_host_priorities(&self) -> Vec<f64> {
        self.hosts.iter()
            .map(|host| {
                // Priority based on available resources (normalized)
                let core_avail = host.available_cores as f64 / host.total_cores.max(1) as f64;
                let mem_avail = host.available_memory as f64 / host.total_memory.max(1) as f64;
                (core_avail + mem_avail) / 2.0
            })
            .collect()
    }

    /// Fill global cluster state features (2 features)
    /// Uses cycle-level snapshot to simulate real LSF behavior
    fn fill_global_state(&mut self, start_idx: usize) {
        // Feature 1: Available cores ratio (from cycle snapshot)
        self.cached_state[start_idx] = self.cycle_available_cores as f32 / self.total_cluster_cores.max(1) as f32;

        // Feature 2: Available memory ratio (from cycle snapshot)
        self.cached_state[start_idx + 1] = self.cycle_available_memory as f32 / self.total_cluster_memory.max(1) as f32;
    }

    /// Fill bucket features (2 features per bucket)
    /// Format: [cores_required_normalized, job_count_in_bucket]
    fn fill_bucket_features(&mut self, bucket_idx: usize, start_idx: usize) {
        if bucket_idx >= self.job_buckets.len() {
            return; // Already zero-filled
        }

        // Extract values we need (to avoid borrow conflicts)
        let (cores_required, job_count) = {
            let bucket = &self.job_buckets[bucket_idx];
            if bucket.jobs.is_empty() {
                return; // Already zero-filled
            }
            let first_job = bucket.jobs.front().unwrap();
            (first_job.cores_required, bucket.jobs.len())
        }; // Borrow of self.job_buckets ends here

        // Feature 1: Job cores requirement (normalized by max job cores)
        self.cached_state[start_idx] = cores_required as f32 / self.job_cores_range.1.max(1) as f32;

        // Feature 2: Number of jobs in bucket (raw count, not normalized)
        self.cached_state[start_idx + 1] = job_count as f32;
    }

    /// Calculate reward for this step
    fn calculate_step_reward(&self, batch_complete: bool) -> f32 {
        if batch_complete {
            self.calculate_pure_resource_utilization_reward()
        } else {
            0.0  // No step penalty - agent just reorders buckets, can't speed up batch
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
        (rewards / self.num_hosts.max(1) as f64) as f32
    }
}

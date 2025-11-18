use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};

use crate::base_env::BaseClusterEnv;
use crate::job::Job;

/// Host sorting environment - agent provides priorities for hosts
#[pyclass]
pub struct HostSortingEnv {
    base: BaseClusterEnv,
    current_batch_index: usize,
    scheduling_attempts_this_batch: usize,

    // Cached state vector for performance
    cached_state: Vec<f32>,
}

#[pymethods]
impl HostSortingEnv {
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
        let base = BaseClusterEnv::new(
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

        // Pre-allocate cached state with correct size
        let state_size = num_hosts * 2 + 8;
        let cached_state = vec![0.0; state_size];

        HostSortingEnv {
            base,
            current_batch_index: 0,
            scheduling_attempts_this_batch: 0,
            cached_state,
        }
    }

    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.base.reset_base();
        self.current_batch_index = 0;
        self.scheduling_attempts_this_batch = 0;

        // Add initial jobs for time=0 (same as env_backup)
        self.base.add_new_jobs_to_queue();

        self.get_state(py)
    }

    pub fn step(&mut self, py: Python, action: &PyAny) -> PyResult<(Py<PyArray1<f32>>, f32, bool, PyObject)> {
        // Parse action (host priorities)
        let action_f64: Vec<f64> = if let Ok(arr32) = action.extract::<PyReadonlyArray1<f32>>() {
            let slice = arr32.as_slice()?;
            slice.iter().map(|&x| x as f64).collect()
        } else if let Ok(arr64) = action.extract::<PyReadonlyArray1<f64>>() {
            arr64.as_slice()?.to_vec()
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "`action` must be a 1-D numpy array of float32 or float64",
            ));
        };

        if action_f64.len() != self.base.num_hosts {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Action must have {} elements (one per host), got {}",
                self.base.num_hosts, action_f64.len())
            ));
        }

        // Schedule jobs using host priorities (bucket-aware like env_backup)
        let _jobs_scheduled = if self.current_batch_index < self.base.batch_processing_queue.len() {
            let job = self.base.batch_processing_queue[self.current_batch_index].clone();
            let job_bucket_key = job.bucket_key.clone();

            // Collect all consecutive jobs with same bucket_key
            let mut jobs_in_bucket = vec![job.clone()];
            let mut idx = self.current_batch_index + 1;
            while idx < self.base.batch_processing_queue.len() {
                if self.base.batch_processing_queue[idx].bucket_key == job_bucket_key {
                    jobs_in_bucket.push(self.base.batch_processing_queue[idx].clone());
                    idx += 1;
                } else {
                    break;
                }
            }
            let bucket_jobs_count = jobs_in_bucket.len();

            // Try to schedule each job in the bucket using the same host priorities
            let mut bucket_scheduled = 0;
            for bucket_job in jobs_in_bucket {
                let scheduled = self.schedule_job_with_host_priorities(bucket_job, &action_f64);
                if scheduled > 0 {
                    bucket_scheduled += 1;
                } else {
                    // If one job can't be scheduled, none of the remaining can (same requirements)
                    break;
                }
            }

            // Increment attempts counter (we made one decision for the entire bucket)
            self.scheduling_attempts_this_batch += 1;

            // Skip all jobs in this bucket (we've processed them all)
            self.current_batch_index += bucket_jobs_count;

            // Check if batch is complete
            if self.current_batch_index >= self.base.batch_processing_queue.len() {
                self.finish_batch_processing();
            }

            bucket_scheduled
        } else {
            0
        };

        // Reset completion counter
        self.base.jobs_completed_this_step = 0;

        // Update utilization
        self.base.update_host_utilization();

        // Process completions
        self.base.process_completions();

        // Calculate reward (same as env_backup)
        let reward = if self.current_batch_index >= self.base.batch_processing_queue.len() {
            self.calculate_pure_resource_utilization_reward()
        } else {
            -0.01
        };

        // Check if batch is complete
        let will_advance_time = self.current_batch_index >= self.base.batch_processing_queue.len();

        // Check if all jobs have been generated
        let all_jobs_generated = self.base.jobs_moved_to_queue >= self.base.total_jobs_in_pool;

        // Time advancement logic (same as env_backup)
        if will_advance_time && !all_jobs_generated {
            self.base.current_time += 1;
            // Generate new jobs for this time unit
            self.base.add_new_jobs_to_queue();
        } else if will_advance_time && all_jobs_generated {
            // If all jobs are generated but still processing, advance time without adding jobs
            self.base.current_time += 1;
        }

        // Move jobs from main queue to submission queue (simulate_job_submissions in env_backup)
        self.simulate_job_submissions();

        self.base.current_step += 1;

        // Episode ends when max_time is reached (same as env_backup)
        let done = self.base.current_time >= self.base.max_time as u64;

        // Set makespan when episode actually ends (same as env_backup)
        if done && self.base.makespan.is_none() {
            self.base.makespan = Some(self.base.current_time);
        }

        let state = self.get_state(py)?;
        let info = self.get_step_info(py)?;

        Ok((state, reward, done, info))
    }

    pub fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // Clear the cached state (already pre-allocated with correct size)
        self.cached_state.fill(0.0);

        // Fill state in format: [host1_avail_cores_norm, host1_avail_mem_norm, host2_avail_cores_norm, ...]
        // Using available resources normalized by environment max values (same as env_backup)
        let max_cores = self.base.host_cores_range.1 as f32;
        let max_memory = self.base.host_memory_range.1 as f32;

        for i in 0..self.base.num_hosts {
            let base_idx = i * 2;
            // Normalized available cores (0 = no cores available, 1 = max possible cores available)
            self.cached_state[base_idx] = self.base.hosts[i].available_cores as f32 / max_cores;
            // Normalized available memory (0 = no memory available, 1 = max possible memory available)
            self.cached_state[base_idx + 1] = self.base.hosts[i].available_memory as f32 / max_memory;
        }

        // Enhanced job info (8 features) - at end of state vector (matching env_backup exactly)
        let job_batch_idx = self.base.num_hosts * 2;

        if self.current_batch_index < self.base.batch_processing_queue.len() {
            let job = &self.base.batch_processing_queue[self.current_batch_index];

            // 1. Job cores normalized
            self.cached_state[job_batch_idx] = job.cores_required as f32 / self.base.job_cores_range.1 as f32;

            // 2. Job memory normalized
            self.cached_state[job_batch_idx + 1] = job.memory_required as f32 / self.base.job_memory_range.1 as f32;

            // 3. Job duration normalized (was missing!)
            self.cached_state[job_batch_idx + 2] = job.duration as f32 / self.base.job_duration_range.1 as f32;

            // 4. Binary is_deferred flag (1 if job has been waiting > 1 second, 0 otherwise)
            let current_waiting_time = self.base.current_time as f64 - job.submission_time;
            let is_deferred = if current_waiting_time > 1.0 { 1.0 } else { 0.0 };
            self.cached_state[job_batch_idx + 3] = is_deferred;

            // 5. Batch progress normalized with tanh - same scale as queue pressure
            // Using num_hosts as natural scale for consistency with queue pressure
            let batch_scale = self.base.num_hosts as f32;
            self.cached_state[job_batch_idx + 4] = (self.current_batch_index as f32 / batch_scale).tanh();

            // 6. Queue pressure - using tanh with num_hosts as natural scale
            let jobs_in_cycle = if !self.base.batch_processing_queue.is_empty() {
                self.base.batch_processing_queue.len()
            } else {
                // Count jobs in queue plus jobs in buckets (includes deferred)
                self.base.job_queue.len() + self.base.job_buckets.iter().map(|b| b.jobs.len()).sum::<usize>()
            };
            let queue_scale = self.base.num_hosts as f32;
            self.cached_state[job_batch_idx + 5] = (jobs_in_cycle as f32 / queue_scale).tanh();

            // Calculate total resource pressure for all jobs
            let mut total_cores_needed = 0u32;
            let mut total_memory_needed = 0u32;

            // Sum resources from batch queue (consistent view) or fallback to individual queues
            if !self.base.batch_processing_queue.is_empty() {
                for queue_job in &self.base.batch_processing_queue {
                    total_cores_needed += queue_job.cores_required;
                    total_memory_needed += queue_job.memory_required;
                }
            } else {
                // Fallback for when not in batch mode
                for queue_job in &self.base.job_queue {
                    total_cores_needed += queue_job.cores_required;
                    total_memory_needed += queue_job.memory_required;
                }
                // Include jobs in buckets (deferred and waiting)
                for bucket in &self.base.job_buckets {
                    for queue_job in &bucket.jobs {
                        total_cores_needed += queue_job.cores_required;
                        total_memory_needed += queue_job.memory_required;
                    }
                }
            }

            // 7. Core pressure ratio
            self.cached_state[job_batch_idx + 6] = (total_cores_needed as f32) / (self.base.total_cluster_cores as f32);

            // 8. Memory pressure ratio
            self.cached_state[job_batch_idx + 7] = (total_memory_needed as f32) / (self.base.total_cluster_memory as f32);

        }

        Ok(PyArray1::from_slice(py, &self.cached_state).to_owned())
    }

    pub fn needs_decision(&self) -> bool {
        self.current_batch_index < self.base.batch_processing_queue.len()
    }

    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        let info = PyDict::new(py);
        info.set_item("queue_length", self.base.job_queue.len())?;
        info.set_item("batch_queue_length", self.base.batch_processing_queue.len())?;
        info.set_item("active_jobs", self.base.active_jobs.len())?;
        info.set_item("needs_decision", self.needs_decision())?;
        info.set_item("total_jobs_generated", self.base.total_jobs_generated)?;
        info.set_item("total_jobs_completed", self.base.total_jobs_completed)?;
        info.set_item("current_time", self.base.current_time)?;
        info.set_item("current_step", self.base.current_step)?;
        info.set_item("total_jobs_deferred", self.base.total_jobs_deferred)?;

        if let Some(makespan_time) = self.base.makespan {
            info.set_item("makespan", makespan_time)?;
        }

        Ok(info.into())
    }

    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        self.base.get_metrics(py)
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.base.original_seed = seed;
        if let Some(s) = seed {
            self.base.rng = rand::SeedableRng::seed_from_u64(s);
        } else {
            self.base.rng = rand::SeedableRng::from_entropy();
        }
    }
}

// Private methods
impl HostSortingEnv {
    fn simulate_job_submissions(&mut self) {
        // Batch scheduling mode: Start batch processing if no current batch
        if self.base.batch_processing_queue.is_empty() && self.current_batch_index == 0 {
            self.start_batch_processing();
        }
    }

    fn start_batch_processing(&mut self) {
        // Add new jobs from job_queue to buckets
        while let Some(job) = self.base.job_queue.pop_front() {
            self.base.add_job_to_bucket(job);
        }

        // Build batch processing queue from buckets (in bucket creation order)
        self.base.batch_processing_queue.clear();
        for bucket in &self.base.job_buckets {
            for job in &bucket.jobs {
                self.base.batch_processing_queue.push_back(job.clone());
            }
        }

        self.current_batch_index = 0;
        self.scheduling_attempts_this_batch = 0;
    }

    fn finish_batch_processing(&mut self) {
        // Clear batch and reset index
        self.base.batch_processing_queue.clear();
        self.current_batch_index = 0;
        self.scheduling_attempts_this_batch = 0;

        // Clear host priorities for all buckets (will be recalculated in next batch)
        for bucket in &mut self.base.job_buckets {
            bucket.host_priorities = None;
        }

        // Remove empty buckets (all jobs in that bucket successfully scheduled)
        self.base.job_buckets.retain(|bucket| !bucket.jobs.is_empty());
    }

    /// Schedule a job using agent-provided host priorities
    /// This overrides the base's default first-available host selection
    fn schedule_job_with_host_priorities(&mut self, job: Job, action: &[f64]) -> usize {
        // Check if job can be scheduled on any host
        let can_be_scheduled = self.base.hosts.iter().any(|host| {
            host.total_cores >= job.cores_required && host.total_memory >= job.memory_required
        });

        if !can_be_scheduled {
            println!("WARNING: Job {} cannot be scheduled on any host", job.id);
            return 0;
        }

        // Sort hosts by agent-provided priorities (descending)
        let mut host_priorities: Vec<(usize, f64)> = action.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        host_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Try single-host scheduling with prioritized hosts
        for &(host_idx, _) in &host_priorities {
            if self.base.try_single_host_scheduling(&job, host_idx) {
                // Remove the job from its bucket since it's scheduled
                let job_id = job.id;
                let job_bucket_key = job.bucket_key.clone();
                for bucket in &mut self.base.job_buckets {
                    if bucket.bucket_key == job_bucket_key {
                        bucket.jobs.retain(|j| j.id != job_id);
                        break;
                    }
                }
                return 1;
            }
        }

        // Job couldn't be scheduled - it stays in its bucket
        // Update deferred count for job in bucket
        let job_id = job.id;
        let job_bucket_key = job.bucket_key.clone();
        for bucket in &mut self.base.job_buckets {
            if bucket.bucket_key == job_bucket_key {
                for bucket_job in &mut bucket.jobs {
                    if bucket_job.id == job_id {
                        bucket_job.deferred_count += 1;
                        self.base.total_jobs_deferred += 1;
                        break;
                    }
                }
                break;
            }
        }

        0
    }

    fn calculate_pure_resource_utilization_reward(&self) -> f32 {
        // Exact same calculation as env_backup
        let mut rewards = 0.0;

        for i in 0..self.base.num_hosts {
            let core_util = self.base.host_core_utils[i] as f64;
            let memory_util = self.base.host_memory_utils[i] as f64;
            rewards += (core_util + memory_util) / 2.0;
        }

        (rewards / self.base.num_hosts as f64) as f32
    }
}
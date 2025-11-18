use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};

use crate::base_env::BaseClusterEnv;
use crate::job::Job;

/// Host sorting environment - agent provides priorities for hosts
#[pyclass]
pub struct HostSortingEnv {
    base: BaseClusterEnv,

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
            cached_state,
        }
    }

    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        self.base.reset_base();

        // Add initial jobs for time=0 (same as env_backup)
        self.base.add_new_jobs_to_queue();

        self.get_state(py)
    }

    pub fn step(&mut self, py: Python, action: &PyAny) -> PyResult<(Py<PyArray1<f32>>, f32, bool, PyObject)> {
        // Start batch processing if needed (BEFORE checking if we should advance time)
        // This ensures we don't advance time when there are jobs waiting to be scheduled
        if self.base.total_jobs_in_current_batch == 0 && !self.base.job_queue.is_empty() {
            self.base.start_batch_processing();
        }
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

        // Schedule jobs using host priorities (directly from buckets)
        // Skip empty buckets to find the next bucket with jobs
        self.base.skip_empty_buckets();

        let _jobs_scheduled = if self.base.current_bucket_index < self.base.job_buckets.len() {
            // Get jobs from current bucket (guaranteed non-empty)
            let current_bucket_idx = self.base.current_bucket_index;
            let bucket = &self.base.job_buckets[current_bucket_idx];
            let jobs_to_schedule: Vec<Job> = bucket.jobs.iter().cloned().collect();

            // Track the original bucket size (before removing any jobs)
            let original_bucket_size = jobs_to_schedule.len();

            // Try to schedule each job in the bucket using the same host priorities
            let mut bucket_scheduled = 0;
            let mut scheduled_job_ids = Vec::new();
            for bucket_job in jobs_to_schedule {
                let job_id = bucket_job.id;
                let scheduled = self.schedule_job_with_host_priorities(bucket_job, &action_f64);
                if scheduled > 0 {
                    bucket_scheduled += 1;
                    scheduled_job_ids.push(job_id);
                    // Note: We'll remove the job from bucket below to match env.rs behavior
                } else {
                    // If one job can't be scheduled, stop trying for this bucket
                    break;
                }
            }

            // Remove scheduled jobs immediately from bucket (like env.rs does)
            // This matches env.rs line 1139: bucket.jobs.retain(|j| j.id != job_id)
            for job_id in scheduled_job_ids {
                self.base.job_buckets[current_bucket_idx].jobs.retain(|j| j.id != job_id);
            }

            // Increment attempts counter (we made one decision for the entire bucket)
            self.base.scheduling_attempts_this_batch += 1;

            // Count this as a processed bucket (for batch progress tracking)
            self.base.buckets_processed += 1;

            // Increment jobs processed counter by the ORIGINAL number of jobs in this bucket
            // (matches env.rs incrementing current_batch_index by bucket_jobs_count)
            self.base.jobs_processed_in_batch += original_bucket_size;

            // Move to next bucket
            self.base.current_bucket_index += 1;

            bucket_scheduled
        } else {
            0
        };

        // Check if we've finished all non-empty buckets
        // Track if we're finishing the batch THIS step (before resetting counters)
        let finishing_batch_now = self.base.total_jobs_in_current_batch > 0 &&
                                  self.base.get_current_job().is_none();

        if finishing_batch_now {
            self.base.finish_batch_processing();
        }

        // Reset completion counter
        self.base.jobs_completed_this_step = 0;

        // Update utilization
        self.base.update_host_utilization();

        // Process completions
        self.base.process_completions();

        // Calculate reward
        // Batch is complete if we just finished it this step
        let batch_complete = finishing_batch_now;
        let reward = if batch_complete {
            self.calculate_pure_resource_utilization_reward()
        } else {
            -0.01
        };

        // Check if batch is complete
        let will_advance_time = batch_complete;

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
        self.base.simulate_job_submissions();

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
        // Skip empty buckets to ensure we're looking at a valid bucket
        self.base.skip_empty_buckets();

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

        // Enhanced job info (8 features) - at end of state vector
        let job_batch_idx = self.base.num_hosts * 2;

        // Get current job from current bucket (empty buckets already skipped above)
        if let Some(job) = self.base.get_current_job() {

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

            // 5. Batch progress normalized with tanh - tracks number of jobs processed (matches env.rs current_batch_index)
            // Using num_hosts as natural scale for consistency with queue pressure
            let batch_scale = self.base.num_hosts as f32;
            self.cached_state[job_batch_idx + 4] = (self.base.jobs_processed_in_batch as f32 / batch_scale).tanh();

            // 6. Queue pressure - matches env.rs behavior with batch_processing_queue
            // During batch: use total_jobs_in_current_batch (equivalent to batch_processing_queue.len())
            // Between batches: count job_queue + buckets
            let jobs_in_cycle: usize = if self.base.total_jobs_in_current_batch > 0 {
                // During batch: use the total count from batch start (like env.rs batch_processing_queue)
                self.base.total_jobs_in_current_batch
            } else {
                // Between batches: count job_queue + buckets (for deferred jobs)
                self.base.job_queue.len() + self.base.job_buckets.iter().map(|b| b.jobs.len()).sum::<usize>()
            };
            let queue_scale = self.base.num_hosts as f32;
            self.cached_state[job_batch_idx + 5] = (jobs_in_cycle as f32 / queue_scale).tanh();

            // 7. Core pressure ratio - matches env.rs behavior
            // During batch: use total resources from batch start (like env.rs batch_processing_queue)
            // Between batches: count job_queue + buckets
            let total_cores_needed: u32;
            let total_memory_needed: u32;

            if self.base.total_jobs_in_current_batch > 0 {
                // During batch: use the total resources from batch start
                // This matches env.rs using batch_processing_queue which is immutable
                total_cores_needed = self.base.total_cores_in_current_batch;
                total_memory_needed = self.base.total_memory_in_current_batch;
            } else {
                // Between batches: count job_queue + buckets (for deferred jobs)
                let mut cores = 0u32;
                let mut memory = 0u32;
                for queue_job in &self.base.job_queue {
                    cores += queue_job.cores_required;
                    memory += queue_job.memory_required;
                }
                for bucket in &self.base.job_buckets {
                    for queue_job in &bucket.jobs {
                        cores += queue_job.cores_required;
                        memory += queue_job.memory_required;
                    }
                }
                total_cores_needed = cores;
                total_memory_needed = memory;
            }

            self.cached_state[job_batch_idx + 6] = (total_cores_needed as f32) / (self.base.total_cluster_cores as f32);
            self.cached_state[job_batch_idx + 7] = (total_memory_needed as f32) / (self.base.total_cluster_memory as f32);

        }

        Ok(PyArray1::from_slice(py, &self.cached_state).to_owned())
    }

    pub fn needs_decision(&self) -> bool {
        // Check if there are any non-empty buckets remaining
        self.base.get_current_job().is_some()
    }

    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        let info = PyDict::new(py);
        info.set_item("queue_length", self.base.job_queue.len())?;

        // Count total jobs in buckets
        let bucket_jobs_count: usize = self.base.job_buckets.iter().map(|b| b.jobs.len()).sum();
        info.set_item("bucket_jobs_count", bucket_jobs_count)?;
        info.set_item("num_buckets", self.base.job_buckets.len())?;

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
                // Job scheduled successfully - increment will be done in step()
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
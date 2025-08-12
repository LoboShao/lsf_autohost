use std::collections::{BinaryHeap, HashMap, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};
use rand::prelude::*;
use rand::distributions::{Uniform, Distribution};

use crate::job::{Job, JobStatus};
use crate::host::Host;
use crate::event::CompletionEvent;


#[allow(dead_code)]
#[pyclass]
pub struct ClusterSchedulerEnv {
    pub num_hosts: usize,
    max_queue_length: usize,
    host_cores_range: (u32, u32),
    host_memory_range: (u32, u32),
    job_cores_range: (u32, u32),
    job_memory_range: (u32, u32),
    job_duration_range: (u32, u32),
    max_jobs_per_step: usize,
    episode_length: usize,  // Maximum time for episode (time-based termination)
    
    // State
    hosts: Vec<Host>,
    host_core_utils: Vec<f32>,           // Real-time core utilization (for performance checking only)
    host_core_utils_15s: Vec<f32>,       // 15-second average core utilization (LSF accessible)
    host_memory_utils: Vec<f32>,         // Real-time memory utilization (LSF accessible)
    
    // Time-based utilization statistics (updated every second)
    core_util_sum: f64,                 // Sum of core utilization values
    memory_util_sum: f64,               // Sum of memory utilization values  
    host_imbalance_sum: f64,            // Sum of host imbalance values
    host_imbalance_sum_squares: f64,    // Sum of squares for imbalance std dev
    effective_util_sum: f64,            // Sum of effective utilization values
    effective_util_sum_squares: f64,    // Sum of squares for effective util std dev
    utilization_sample_count: u64,      // Number of time samples collected
    last_stats_update_time: f64,        // Last time statistics were updated
    job_queue: VecDeque<Job>,
    submission_queue: VecDeque<Job>,  // Jobs waiting for scheduling decisions
    deferred_jobs: VecDeque<Job>,     // Jobs that couldn't be scheduled, retry later
    active_jobs: HashMap<u32, Job>,
    completion_heap: BinaryHeap<CompletionEvent>,
    current_time: f64,
    current_step: usize,  // Track agent steps
    next_job_id: u32,
    
    
    // Deterministic job generation
    
    // Pre-generated deterministic job schedule
    job_arrival_schedule: Vec<usize>,  // Number of jobs arriving at each timestep
    job_cores_schedule: Vec<u32>,      // Cores required for each job
    job_memory_schedule: Vec<u32>,     // Memory required for each job
    job_duration_schedule: Vec<u32>,   // Duration for each job
    total_jobs_in_pool: usize,         // Total jobs that will be generated
    jobs_moved_to_queue: usize,        // Jobs moved from pool to queue so far
    
    // Metrics
    total_jobs_generated: u32,
    total_jobs_completed: u32,
    total_jobs_failed: u32,
    jobs_completed_this_step: u32,
    
    // Batch-wise reward tracking
    last_reward_time: f64,  // Last time we gave a batch reward
    
    // RNG
    rng: StdRng,
    original_seed: Option<u64>,  // Store original seed for deterministic timestep-based generation
    
    
    // Cached state vector
    cached_state: Vec<f32>,
}

#[pymethods]
impl ClusterSchedulerEnv {
    #[new]
    #[pyo3(signature = (
        num_hosts = 1000,
        max_queue_length = None,
        host_cores_range = (32, 128),
        host_memory_range = (128, 512),
        job_cores_range = (1, 32),
        job_memory_range = (2, 64),
        job_duration_range = (1, 60),
        max_jobs_per_step = 50,
        episode_length = 4096,
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
        episode_length: usize,
        seed: Option<u64>,
    ) -> Self {
        // Calculate max_queue_length if not provided
        let actual_max_queue_length = max_queue_length.unwrap_or(episode_length * max_jobs_per_step);
        
        let original_seed = seed;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        
        // Pre-generate deterministic job schedule
        let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) = 
            Self::generate_deterministic_job_schedule(
                episode_length,
                max_jobs_per_step,
                job_cores_range,
                job_memory_range,
                job_duration_range,
                &mut rng
            );

        // Create hosts
        let mut hosts = Vec::with_capacity(num_hosts);
        for i in 0..num_hosts {
            let cores = rng.gen_range(host_cores_range.0..=host_cores_range.1);
            let memory = rng.gen_range(host_memory_range.0..=host_memory_range.1);
            hosts.push(Host::new(i, cores, memory));
        }
        
        let host_core_utils = vec![0.0; num_hosts];
        let host_core_utils_15s = vec![0.0; num_hosts];
        let host_memory_utils = vec![0.0; num_hosts];
        
        ClusterSchedulerEnv {
            num_hosts,
            max_queue_length: actual_max_queue_length,
            host_cores_range,
            host_memory_range,
            job_cores_range,
            job_memory_range,
            job_duration_range,
            max_jobs_per_step,
            episode_length,
            hosts,
            host_core_utils,
            host_core_utils_15s,
            host_memory_utils,
            core_util_sum: 0.0,
            memory_util_sum: 0.0,
            host_imbalance_sum: 0.0,
            host_imbalance_sum_squares: 0.0,
            effective_util_sum: 0.0,
            effective_util_sum_squares: 0.0,
            utilization_sample_count: 0,
            last_stats_update_time: 0.0,
            job_queue: VecDeque::new(),
            submission_queue: VecDeque::new(),
            deferred_jobs: VecDeque::new(),
            active_jobs: HashMap::new(),
            completion_heap: BinaryHeap::new(),
            current_time: 0.0,
            current_step: 0,
            next_job_id: 0,
            job_arrival_schedule,
            job_cores_schedule,
            job_memory_schedule,
            job_duration_schedule,
            total_jobs_in_pool,
            jobs_moved_to_queue: 0,
            total_jobs_generated: 0,
            total_jobs_completed: 0,
            total_jobs_failed: 0,
            jobs_completed_this_step: 0,
            last_reward_time: -1.0,
            rng,
            original_seed,
            cached_state: Vec::new(),
        }
    }
    
    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // Reset hosts
        for host in &mut self.hosts {
            host.available_cores = host.total_cores;
            host.available_memory = host.total_memory;
            host.running_job_ids.clear();
        }
        
        // Clear collections
        self.job_queue.clear();
        self.submission_queue.clear();
        self.deferred_jobs.clear();
        self.active_jobs.clear();
        self.completion_heap.clear();
        
        // Reset state
        self.current_time = 0.0;
        self.current_step = 0;
        self.next_job_id = 0;
        
        // Reset metrics
        self.total_jobs_generated = 0;
        self.total_jobs_completed = 0;
        self.total_jobs_failed = 0;
        self.jobs_completed_this_step = 0;
        
        // Reset batch tracking
        self.last_reward_time = -1.0;
        
        // Reset job tracking
        self.jobs_moved_to_queue = 0;
        
        // Reset utilization arrays
        self.host_core_utils.fill(0.0);
        self.host_core_utils_15s.fill(0.0);
        self.host_memory_utils.fill(0.0);
        
        // Reset time-based statistics
        self.core_util_sum = 0.0;
        self.memory_util_sum = 0.0;
        self.host_imbalance_sum = 0.0;
        self.host_imbalance_sum_squares = 0.0;
        self.effective_util_sum = 0.0;
        self.effective_util_sum_squares = 0.0;
        self.utilization_sample_count = 0;
        self.last_stats_update_time = 0.0;
        
        
        // Re-generate deterministic job schedule if we have original seed
        if let Some(seed) = self.original_seed {
            let mut schedule_rng = StdRng::seed_from_u64(seed);
            let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) = 
                Self::generate_deterministic_job_schedule(
                    self.episode_length,
                    self.max_jobs_per_step,
                    self.job_cores_range,
                    self.job_memory_range,
                    self.job_duration_range,
                    &mut schedule_rng
                );
            
            self.job_arrival_schedule = job_arrival_schedule;
            self.job_cores_schedule = job_cores_schedule;
            self.job_memory_schedule = job_memory_schedule;
            self.job_duration_schedule = job_duration_schedule;
            self.total_jobs_in_pool = total_jobs_in_pool;
        }
        
        self.add_jobs_to_queue();
        
        self.get_state(py)
    }
    
    pub fn step(&mut self, py: Python, action: &PyAny) -> PyResult<(Py<PyArray1<f32>>, f32, bool, PyObject)> {
        // Parse action
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
        
        if action_f64.len() != self.num_hosts {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Action must have {} elements (one per host), got {}", 
                self.num_hosts, action_f64.len())
            ));
        }
        
        // Apply scheduling decision
        let jobs_scheduled = if !self.submission_queue.is_empty() {
            let job = self.submission_queue.pop_front().unwrap();
            self.schedule_single_job_from_submission(job, &action_f64)
        } else {
            0
        };
        
        // Reset completion counter for this step
        self.jobs_completed_this_step = 0;
        
        // Update host utilization BEFORE processing completions to capture actual resource usage
        self.update_host_utilization();
        
        self.process_completions();
        
        // Check if we should advance time based on arrival batching
        if self.should_advance_time_after_current_job() {
            // Generate jobs for this new time unit
            self.add_jobs_to_queue();
            
            // When time advances, move deferred jobs back to main queue for retry
            while let Some(deferred_job) = self.deferred_jobs.pop_front() {
                self.job_queue.push_back(deferred_job);
            }
        }
        
        // Move jobs from main queue to submission queue
        self.simulate_job_submissions();
        
        self.current_step += 1;
        
        // Calculate resource balance reward - addresses core/memory imbalance
        let total_reward = self.calculate_resource_balance_reward(jobs_scheduled) as f32;
        let done = self.current_time >= self.episode_length as f64;
        
        let info = PyDict::new(py);
        info.set_item("jobs_scheduled", jobs_scheduled)?;
        info.set_item("queue_length", self.job_queue.len())?;
        info.set_item("submission_queue_length", self.submission_queue.len())?;
        info.set_item("active_jobs", self.active_jobs.len())?;
        info.set_item("needs_decision", !self.submission_queue.is_empty())?;
        
        let state = self.get_state(py)?;
        
        Ok((state, total_reward as f32, done, info.into()))
    }
    
    fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {

        // Calculate state size: 15s core utils + memory utils + single job
        let state_size = self.num_hosts * 2 + 2;
        
        // Resize cached vector if needed
        if self.cached_state.len() != state_size {
            self.cached_state.resize(state_size, 0.0);
        }
        
        // Clear the cached state
        self.cached_state.fill(0.0);
        
        // Fill state in format: [host1_cores15s, host1_mem, ..., hostN_cores15s, hostN_mem]
        for i in 0..self.num_hosts {
            let base_idx = i * 2;
            self.cached_state[base_idx] = self.host_core_utils_15s[i];     // 15-second core average (LSF accessible)
            self.cached_state[base_idx + 1] = self.host_memory_utils[i];    // Available memory (LSF accessible)
        }
        
        // Single job info (only 1 job observable) - at end of state vector
        let job_batch_idx = self.num_hosts * 2;
        
        if !self.submission_queue.is_empty() {
            let job = &self.submission_queue[0];
            self.cached_state[job_batch_idx] = job.cores_required as f32 / self.job_cores_range.1 as f32;
            self.cached_state[job_batch_idx + 1] = job.memory_required as f32 / self.job_memory_range.1 as f32;
        }
  
        Ok(PyArray1::from_vec(py, self.cached_state.clone()).to_owned())
    }
    
    pub fn needs_decision(&self) -> bool {
        !self.submission_queue.is_empty()
    }
    
    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.original_seed = seed;
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        } else {
            self.rng = StdRng::from_entropy();
        }
    }
    
    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        let total_jobs_in_system = self.jobs_moved_to_queue.max(1) as f64;
        
        let avg_core_util = if self.utilization_sample_count > 0 {
            (self.core_util_sum / self.utilization_sample_count as f64) as f32
        } else {
            0.0
        };
        
        let avg_memory_util = if self.utilization_sample_count > 0 {
            (self.memory_util_sum / self.utilization_sample_count as f64) as f32
        } else {
            0.0
        };
        
        let imbalance_std = if self.utilization_sample_count > 1 {
            let mean = self.host_imbalance_sum / self.utilization_sample_count as f64;
            let variance = (self.host_imbalance_sum_squares / self.utilization_sample_count as f64) - (mean * mean);
            variance.max(0.0).sqrt()
        } else {
            0.0
        };
        
        let effective_util_std = if self.utilization_sample_count > 1 {
            let mean = self.effective_util_sum / self.utilization_sample_count as f64;
            let variance = (self.effective_util_sum_squares / self.utilization_sample_count as f64) - (mean * mean);
            variance.max(0.0).sqrt()
        } else {
            0.0
        };
        
        let metrics = PyDict::new(py);
        
        // Job completion metrics
        metrics.set_item("total_jobs_completed", self.total_jobs_completed)?;
        metrics.set_item("completion_rate", self.total_jobs_completed as f64 / self.total_jobs_generated.max(1) as f64)?;
        metrics.set_item("jobs_in_progress", self.active_jobs.len())?;
        
        // Time-based utilization metrics (averaged over seconds, not timesteps)
        metrics.set_item("avg_host_core_utilization", avg_core_util)?;
        metrics.set_item("avg_host_memory_utilization", avg_memory_util)?;
        metrics.set_item("utilization_samples_collected", self.utilization_sample_count)?;
        
        // Statistical balance metrics (no thresholds needed)
        metrics.set_item("host_imbalance_std", imbalance_std)?;  // Lower = more consistent balance
        metrics.set_item("effective_util_std", effective_util_std)?;  // Lower = more uniform utilization
        
        Ok(metrics.into())
    }
    
}

// Private implementation methods
impl ClusterSchedulerEnv {
    fn generate_deterministic_job_schedule(
        episode_length: usize,
        max_jobs_per_step: usize,
        job_cores_range: (u32, u32),
        job_memory_range: (u32, u32),
        job_duration_range: (u32, u32),
        rng: &mut StdRng,
    ) -> (Vec<usize>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        let mut job_arrival_schedule = Vec::with_capacity(episode_length);
        let mut job_cores_schedule = Vec::new();
        let mut job_memory_schedule = Vec::new();
        let mut job_duration_schedule = Vec::new();
        
        let cores_dist = Uniform::from(job_cores_range.0..=job_cores_range.1);
        let memory_dist = Uniform::from(job_memory_range.0..=job_memory_range.1);
        let duration_dist = Uniform::from(job_duration_range.0..=job_duration_range.1);
        
        // Generate job arrival schedule for each timestep
        for _timestep in 0..episode_length {
            let num_jobs = rng.gen_range(1..=max_jobs_per_step);
            job_arrival_schedule.push(num_jobs);
            
            // Generate properties for jobs arriving at this timestep
            for _job in 0..num_jobs {
                job_cores_schedule.push(cores_dist.sample(rng));
                job_memory_schedule.push(memory_dist.sample(rng));
                job_duration_schedule.push(duration_dist.sample(rng));
            }
        }
        let total_jobs_in_pool = job_cores_schedule.len();
        
        (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool)
    }
    
    fn simulate_job_submissions(&mut self) {
        // Move jobs from main queue to submission queue
        // This creates the one-at-a-time decision pattern
        while !self.job_queue.is_empty() && self.submission_queue.is_empty() {
            let job = self.job_queue.pop_front().unwrap();
            self.submission_queue.push_back(job);
        }
    }
    
    fn add_jobs_to_queue(&mut self) {
        let timestep = self.current_time as usize;
        
        // Check if we're at or beyond episode end - this is normal termination condition
        if timestep >= self.episode_length {
            return; // Episode should end, no more jobs to add
        }
        
        // Error checking - this should never happen in a properly designed system
        if timestep >= self.job_arrival_schedule.len() {
            panic!("Timestep {} exceeds job arrival schedule length {}. This indicates a design error.", 
                   timestep, self.job_arrival_schedule.len());
        }
        
        if self.jobs_moved_to_queue >= self.total_jobs_in_pool {
            // This could happen near episode end, just return instead of panicking
            return;
        }
        
        let num_jobs_to_add = self.job_arrival_schedule[timestep];
        let jobs_to_add = num_jobs_to_add.min(self.max_queue_length - self.job_queue.len());
        
        for _ in 0..jobs_to_add {
            // Final safety check
            if self.jobs_moved_to_queue >= self.total_jobs_in_pool {
                panic!("Job pool exhausted during job addition. Pool size: {}, Jobs moved: {}", 
                       self.total_jobs_in_pool, self.jobs_moved_to_queue);
            }
            
            // Get job properties from pre-generated schedule
            let cores = self.job_cores_schedule[self.jobs_moved_to_queue];
            let memory = self.job_memory_schedule[self.jobs_moved_to_queue];
            let duration = self.job_duration_schedule[self.jobs_moved_to_queue];
            
            let job = Job::new(
                self.next_job_id,
                cores,
                memory,
                duration,
                self.current_time,
            );
            
            self.next_job_id += 1;
            self.total_jobs_generated += 1;
            self.jobs_moved_to_queue += 1;
            self.job_queue.push_back(job);
        }
    }
    
    fn process_completions(&mut self) {
        let mut completed_jobs = Vec::new();
        
        while let Some(event) = self.completion_heap.peek() {
            if event.completion_time > self.current_time.floor() {
                break;
            }
            
            let event = self.completion_heap.pop().unwrap();
            
            if let Some(mut job) = self.active_jobs.remove(&event.job_id) {
                job.end_time = Some(event.completion_time);
                job.status = JobStatus::Completed;
                
                // Release from host
                if let Some(host_id) = job.assigned_host {
                    self.hosts[host_id].release_job(&job);
                }
                
                self.total_jobs_completed += 1;
                self.jobs_completed_this_step += 1;
                
                completed_jobs.push(job);
            }
        }
    }
    
    fn update_host_utilization(&mut self) {
        for (i, host) in self.hosts.iter_mut().enumerate() {
            // Update core history first with current time
            host.update_core_history(self.current_time.floor() as f32);
            
            // Core utilization: real-time (LSF can see this instantly)
            self.host_core_utils[i] = host.get_core_utilization();
            
            // Core utilization: 15-second average (LSF records recent core usage)
            self.host_core_utils_15s[i] = host.get_core_utilization_15s_avg();
            
            
            // Memory utilization: exact (LSF can see this instantly)
            self.host_memory_utils[i] = host.get_memory_utilization();
        }
        
        // Update time-based statistics every second
        self.update_utilization_statistics();
    }
    
    fn update_utilization_statistics(&mut self) {
        // Only update once per second
        let current_time_floor = self.current_time.floor();
        if current_time_floor > self.last_stats_update_time {
            self.last_stats_update_time = current_time_floor;
            
            // Calculate current average utilization across all hosts
            let avg_core_util = self.host_core_utils.iter().sum::<f32>() / self.num_hosts as f32;
            let avg_memory_util = self.host_memory_utils.iter().sum::<f32>() / self.num_hosts as f32;
            
            // Calculate host-level balance statistics for this time sample
            let mut host_imbalances = Vec::new();
            let mut effective_utils = Vec::new();
            
            for i in 0..self.num_hosts {
                let core_util = self.host_core_utils[i] as f64;
                let memory_util = self.host_memory_utils[i] as f64;
                
                // Only consider active hosts
                if core_util > 0.01 || memory_util > 0.01 {
                    host_imbalances.push((core_util - memory_util).abs());
                    effective_utils.push(core_util.min(memory_util));
                }
            }
            
            // Calculate averages for this time sample
            let avg_imbalance = if !host_imbalances.is_empty() {
                host_imbalances.iter().sum::<f64>() / host_imbalances.len() as f64
            } else {
                0.0
            };
            
            let avg_effective_util = if !effective_utils.is_empty() {
                effective_utils.iter().sum::<f64>() / effective_utils.len() as f64
            } else {
                0.0
            };
            
            // Update running statistics
            self.core_util_sum += avg_core_util as f64;
            self.memory_util_sum += avg_memory_util as f64;
            self.host_imbalance_sum += avg_imbalance;
            self.host_imbalance_sum_squares += avg_imbalance * avg_imbalance;
            self.effective_util_sum += avg_effective_util;
            self.effective_util_sum_squares += avg_effective_util * avg_effective_util;
            self.utilization_sample_count += 1;
        }
    }
    
    // Removed create_job_buckets - no longer needed for single job scheduling
    
    fn schedule_single_job_from_submission(&mut self, job: Job, action: &[f64]) -> usize {
        // Sort hosts by priority (descending)
        let mut host_priorities: Vec<(usize, f64)> = action.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        host_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Try hosts in priority order
        for &(host_idx, _) in &host_priorities {
            let host = &mut self.hosts[host_idx];
            
            if host.can_accommodate(&job) {
                // Clone only when we can actually schedule the job
                let mut job_to_schedule = job.clone();
                
                // Successfully schedule
                if host.allocate_job(&mut job_to_schedule) {
                    job_to_schedule.start_time = Some(self.current_time);
                    
                    // Schedule completion
                    let completion_time = self.current_time.floor() + job_to_schedule.duration as f64;
                    let job_id = job_to_schedule.id;
                    self.completion_heap.push(CompletionEvent {
                        completion_time,
                        job_id,
                    });
                    
                    self.active_jobs.insert(job_id, job_to_schedule);
                    return 1; // Successfully scheduled one job
                }
            }
        }
        
        // Put unscheduled job to deferred queue - will be retried after time advances
        self.deferred_jobs.push_back(job);
        
        0 // No job scheduled
    }

    // Baseline scheduler: first available host
    fn schedule_single_job_baseline_first_available(&mut self, job: Job) -> usize {
        // Try hosts in order (0, 1, 2, ...)
        for host in &mut self.hosts {
            if host.can_accommodate(&job) {
                // Clone only when we can actually schedule the job
                let mut job_to_schedule = job.clone();
                
                // Successfully schedule
                if host.allocate_job(&mut job_to_schedule) {
                    job_to_schedule.start_time = Some(self.current_time);
                    
                    // Schedule completion
                    let completion_time = self.current_time.floor() + job_to_schedule.duration as f64;
                    let job_id = job_to_schedule.id;
                    self.completion_heap.push(CompletionEvent {
                        completion_time,
                        job_id,
                    });
                    
                    self.active_jobs.insert(job_id, job_to_schedule);
                    return 1; // Successfully scheduled one job
                }
            }
        }
        
        // Put unscheduled job to deferred queue - will be retried after time advances
        self.deferred_jobs.push_back(job);
        
        0 // No job scheduled
    }
    
    
    fn should_advance_time_after_current_job(&mut self) -> bool {
        // Advance time when we've processed all jobs that need immediate decisions
        // This means all jobs from the current time unit have been scheduled or deferred
        if self.submission_queue.is_empty() {
            let old_timestep = self.current_time.floor() as usize;
            
            // Advance time proportionally - if N jobs arrived this second, 
            // each job represents 1/N of a second
            let current_timestep = self.current_time.floor() as usize;
            if current_timestep < self.job_arrival_schedule.len() {
                let jobs_this_second = self.job_arrival_schedule[current_timestep];
                if jobs_this_second > 0 {
                    self.current_time += 1.0 / jobs_this_second as f64;
                } else {
                    self.current_time += 1.0; // If no jobs, advance full second
                }
            } else {
                self.current_time += 1.0;
            }
            
            // Only return true (trigger job generation) when we cross into a new second
            let new_timestep = self.current_time.floor() as usize;
            new_timestep > old_timestep
        } else {
            false
        }
    }
    

    fn calculate_resource_balance_reward(&mut self, scheduled_jobs: usize) -> f64 {
        // Resource efficiency component (0.7 weight)
        let mut total_effective_util = 0.0;
        
        for i in 0..self.num_hosts {
            let core_util = self.host_core_utils[i] as f64;
            let memory_util = self.host_memory_utils[i] as f64;
            
            total_effective_util += core_util.min(memory_util);
        }
        
        let avg_effective_util = total_effective_util / self.num_hosts as f64;
        let resource_reward = avg_effective_util * 0.7;

        // Throughput component (0.3 weight) - normalized by host capacity to prevent domination
        let completion_rate = self.jobs_completed_this_step as f64 / self.num_hosts as f64;
        let throughput_reward = completion_rate.min(1.0) * 0.3; // Cap at 0.3 to prevent reward explosion
        
        resource_reward + throughput_reward
    }

}
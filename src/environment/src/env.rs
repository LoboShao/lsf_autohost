use std::collections::{BinaryHeap, HashMap, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1};
use rand::prelude::*;
use rand::distributions::{Uniform, Distribution};

use crate::job::{Job, JobStatus};
use crate::host::Host;
use crate::event::CompletionEvent;

// Bucket for grouping jobs with same resource requirements
#[derive(Debug, Clone)]
struct JobBucket {
    bucket_key: String,
    jobs: VecDeque<Job>,
    // Shared host priorities for all jobs in this bucket
    host_priorities: Option<Vec<f32>>,
}


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
    max_time: usize,  // Maximum time for job generation (after this, wait for completion)
    use_skewed_arrivals: bool,  // Whether to use skewed job arrival distribution
    
    // State
    hosts: Vec<Host>,
    host_core_utils: Vec<f32>,           // Host core utilization in last second (LSF accessible)
    host_memory_utils: Vec<f32>,         // Host memory utilization in last second (LSF accessible)
    
    // Time-based utilization statistics (updated every second)
    core_util_sum: f64,                 // Sum of core utilization values
    memory_util_sum: f64,               // Sum of memory utilization values  
    last_stats_update_time: u64,        // Last time statistics were updated
    job_queue: VecDeque<Job>,
    
    // Batch scheduling mode
    batch_processing_queue: VecDeque<Job>,  // All jobs being scheduled this cycle
    current_batch_index: usize,             // Which job we're deciding on
    scheduling_attempts_this_batch: usize,  // Number of scheduling attempts in current batch
    
    // Bucket scheduling (LSF-style)
    job_buckets: Vec<JobBucket>,           // Buckets persist across cycles
    
    active_jobs: HashMap<u32, Job>,
    completion_heap: BinaryHeap<CompletionEvent>,
    current_time: u64,  // Integer seconds only
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
    total_jobs_deferred: u32,  // Track total deferral events
    jobs_completed_this_step: u32,
    
    // Waiting time tracking for all jobs in episode
    total_waiting_time_all_jobs: f64,  // Sum of waiting times for all jobs (completed + in progress)
    
    // Makespan tracking
    makespan: Option<u64>,  // Time when all jobs finished (set when episode ends)
    
    // RNG
    rng: StdRng,
    original_seed: Option<u64>,  // Store original seed for deterministic timestep-based generation
    
    // Cluster resource totals (cached for performance)
    total_cluster_cores: u32,
    total_cluster_memory: u32,
    
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
        host_memory_range = (131072, 524288),  // 128GB-512GB in MB
        job_cores_range = (1, 32),
        job_memory_range = (2048, 65536),      // 2GB-64GB in MB
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
        // Calculate max_queue_length if provided externally; otherwise default to max_time * max_jobs_per_step
        let actual_max_queue_length = max_queue_length.unwrap_or(max_time * max_jobs_per_step);
        let original_seed = seed;
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        
        // Pre-generate deterministic job schedule
        let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) = 
            Self::generate_deterministic_job_schedule(
                max_time,
                max_jobs_per_step,
                job_cores_range,
                job_memory_range,
                job_duration_range,
                use_skewed_arrivals,
                &mut rng
            );

        // Create hosts with realistic configurations and calculate min/max for normalization
        let mut hosts = Vec::with_capacity(num_hosts);
        let mut host_configs = Vec::with_capacity(num_hosts);
        
        // First pass: generate all host configurations
        for _i in 0..num_hosts {
            let (cores, memory) = Self::generate_realistic_host_config(host_cores_range, host_memory_range, &mut rng);
            host_configs.push((cores, memory));
        }
        
        // Create hosts from configs
        for (i, (cores, memory)) in host_configs.into_iter().enumerate() {
            hosts.push(Host::new(i, cores, memory));
        }
        
        // Calculate total cluster resources
        let total_cluster_cores: u32 = hosts.iter().map(|h| h.total_cores).sum();
        let total_cluster_memory: u32 = hosts.iter().map(|h| h.total_memory).sum();
        
        let host_core_utils = vec![0.0; num_hosts];
        let host_memory_utils = vec![0.0; num_hosts];
        
        // Pre-allocate cached state with correct size (2 features per host + 8 job features)
        let state_size = num_hosts * 2 + 8;
        let cached_state = vec![0.0; state_size];
        
        ClusterSchedulerEnv {
            num_hosts,
            max_queue_length: actual_max_queue_length,
            host_cores_range,
            host_memory_range,
            job_cores_range,
            job_memory_range,
            job_duration_range,
            max_jobs_per_step,
            max_time,
            use_skewed_arrivals,
            hosts,
            host_core_utils,
            host_memory_utils,
            core_util_sum: 0.0,
            memory_util_sum: 0.0,
            last_stats_update_time: 0,
            job_queue: VecDeque::new(),
            batch_processing_queue: VecDeque::new(),
            current_batch_index: 0,
            scheduling_attempts_this_batch: 0,
            job_buckets: Vec::new(),
            active_jobs: HashMap::new(),
            completion_heap: BinaryHeap::new(),
            current_time: 0,
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
            total_jobs_deferred: 0,
            jobs_completed_this_step: 0,
            total_waiting_time_all_jobs: 0.0,
            makespan: None,
            rng,
            original_seed,
            total_cluster_cores,
            total_cluster_memory,
            cached_state,
        }
    }
    
    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // Re-generate cluster configuration and recalculate min/max for normalization
        let mut host_configs = Vec::with_capacity(self.num_hosts);
        
        if let Some(seed) = self.original_seed {
            // Use seed for deterministic host generation during testing
            let mut cluster_rng = StdRng::seed_from_u64(seed);
            for _i in 0..self.num_hosts {
                let (cores, memory) = Self::generate_realistic_host_config(
                    self.host_cores_range, 
                    self.host_memory_range, 
                    &mut cluster_rng
                );
                host_configs.push((cores, memory));
            }
        } else {
            // Use random host generation for training diversity
            for _i in 0..self.num_hosts {
                let (cores, memory) = Self::generate_realistic_host_config(
                    self.host_cores_range, 
                    self.host_memory_range, 
                    &mut self.rng
                );
                host_configs.push((cores, memory));
            }
        }
        
        
        // Apply configurations to hosts with proper normalization
        for (i, (cores, memory)) in host_configs.into_iter().enumerate() {
            let host = &mut self.hosts[i];
            host.total_cores = cores;
            host.total_memory = memory;
            host.available_cores = cores;
            host.available_memory = memory;
            host.running_job_ids.clear();
        }
        
        // Recalculate total cluster resources
        self.total_cluster_cores = self.hosts.iter().map(|h| h.total_cores).sum();
        self.total_cluster_memory = self.hosts.iter().map(|h| h.total_memory).sum();
        
        // Clear collections
        self.job_queue.clear();
        self.batch_processing_queue.clear();
        self.current_batch_index = 0;
        self.scheduling_attempts_this_batch = 0;
        self.job_buckets.clear();
        self.active_jobs.clear();
        self.completion_heap.clear();
        
        // Reset state
        self.current_time = 0;
        self.current_step = 0;
        self.next_job_id = 0;
        
        // Reset metrics
        self.total_jobs_generated = 0;
        self.total_jobs_completed = 0;
        self.total_jobs_deferred = 0;
        self.jobs_completed_this_step = 0;
        self.total_waiting_time_all_jobs = 0.0;
        
        // Reset makespan tracking
        self.makespan = None;
        
        // Reset job tracking
        self.jobs_moved_to_queue = 0;
        
        // Reset utilization arrays
        self.host_core_utils.fill(0.0);
        self.host_memory_utils.fill(0.0);
        
        // Reset time-based statistics
        self.core_util_sum = 0.0;
        self.memory_util_sum = 0.0;
        self.last_stats_update_time = 0;
        
        
        // Re-generate deterministic job schedule if we have original seed
        if let Some(seed) = self.original_seed {
            let mut schedule_rng = StdRng::seed_from_u64(seed);
            let (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool) = 
                Self::generate_deterministic_job_schedule(
                    self.max_time,
                    self.max_jobs_per_step,
                    self.job_cores_range,
                    self.job_memory_range,
                    self.job_duration_range,
                    self.use_skewed_arrivals,
                    &mut schedule_rng
                );
            
            self.job_arrival_schedule = job_arrival_schedule;
            self.job_cores_schedule = job_cores_schedule;
            self.job_memory_schedule = job_memory_schedule;
            self.job_duration_schedule = job_duration_schedule;
            self.total_jobs_in_pool = total_jobs_in_pool;
        }
        
        
        // Add initial jobs for time=0
        self.add_new_jobs_to_queue();
        
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
        
        // Apply scheduling decision in batch mode
        let _jobs_scheduled = if self.current_batch_index < self.batch_processing_queue.len() {
            let job = self.batch_processing_queue[self.current_batch_index].clone();
            let job_bucket_key = job.bucket_key.clone();
            
            // When we see a job, we're deciding priorities for its entire bucket
            // Schedule all jobs in this bucket that can be scheduled
            let mut bucket_scheduled = 0;
            
            // Collect all jobs in this bucket from the current position onwards
            let mut jobs_in_bucket = vec![job.clone()];
            let mut idx = self.current_batch_index + 1;
            while idx < self.batch_processing_queue.len() {
                if self.batch_processing_queue[idx].bucket_key == job_bucket_key {
                    jobs_in_bucket.push(self.batch_processing_queue[idx].clone());
                    idx += 1;
                } else {
                    break;
                }
            }
            let bucket_jobs_count = jobs_in_bucket.len();
            
            // Try to schedule each job in the bucket using the same host priorities
            for bucket_job in jobs_in_bucket {
                let scheduled = self.schedule_single_job_from_batch(bucket_job, &action_f64);
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
            
            // Check if batch is complete and force completion if needed
            if self.current_batch_index >= self.batch_processing_queue.len() {
                self.finish_batch_processing();
            }
            
            bucket_scheduled
        } else {
            0
        };
        
        // Reset completion counter for this step
        self.jobs_completed_this_step = 0;
        
        // Update host utilization BEFORE processing completions to capture actual resource usage
        self.update_host_utilization();
        
        self.process_completions();

        let will_advance_time = self.current_batch_index >= self.batch_processing_queue.len();
        
        let total_reward = if self.current_batch_index >= self.batch_processing_queue.len() {
            self.calculate_pure_resource_utilization_reward() as f32
        } else {
            -0.01
        };
        // let total_reward = self.calculate_pure_resource_utilization_reward() as f32;
        
        // Check if all jobs have been generated (no more jobs in the pool)
        let all_jobs_generated = self.jobs_moved_to_queue >= self.total_jobs_in_pool;
        
        // NOW advance time if needed (after reward calculation)
        // Only generate new jobs if we haven't generated all jobs yet
        if will_advance_time && !all_jobs_generated {
            self.current_time += 1;
            // Generate NEW jobs for this new time unit (deferred jobs already moved above)
            self.add_new_jobs_to_queue();
        } else if will_advance_time && all_jobs_generated {
            // If all jobs are generated but we're still processing, advance time without adding jobs
            self.current_time += 1;
        }
        
        // Move jobs from main queue to submission queue
        self.simulate_job_submissions();
        
        self.current_step += 1;
        
        // Episode ends when max_time is reached (original behavior)
        let done = self.current_time >= self.max_time as u64;
        
        // Still track if all jobs finished for metrics
        // let all_jobs_finished = self.active_jobs.is_empty() && 
        //                        self.job_queue.is_empty() && 
        //                        self.batch_processing_queue.is_empty() &&
        //                        self.job_buckets.is_empty();
        
        // Set makespan when episode actually ends
        if done && self.makespan.is_none() {
            self.makespan = Some(self.current_time);
        }
        
        let info = PyDict::new(py);
        // Return minimal info - use get_step_info() for detailed information
        
        let state = self.get_state(py)?;
        
        Ok((state, total_reward as f32, done, info.into()))
    }
    
    fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // Clear the cached state (already pre-allocated with correct size)
        self.cached_state.fill(0.0);

        // Fill state in format: [host1_avail_cores_norm, host1_avail_mem_norm, host2_avail_cores_norm, ...]
        // Using available resources from PREVIOUS scheduling cycle (LSF-style periodic updates)
        let max_cores = self.host_cores_range.1 as f32;
        let max_memory = self.host_memory_range.1 as f32;

        for i in 0..self.num_hosts {
            let base_idx = i * 2;

            // Get utilization from last scheduling cycle (from history)
            let core_util = self.hosts[i].get_core_utilization();
            let memory_util = self.hosts[i].get_memory_utilization();

            // Convert utilization to available resources
            // utilization = 1.0 - (available / total)
            // available = total * (1.0 - utilization)
            let available_cores = self.hosts[i].total_cores as f32 * (1.0 - core_util);
            let available_memory = self.hosts[i].total_memory as f32 * (1.0 - memory_util);

            // Normalized available cores (0 = no cores available, 1 = max possible cores available)
            self.cached_state[base_idx] = available_cores / max_cores;
            // Normalized available memory (0 = no memory available, 1 = max possible memory available)
            self.cached_state[base_idx + 1] = available_memory / max_memory;
        }
        
        // Enhanced job info (7 features) - at end of state vector
        let job_batch_idx = self.num_hosts * 2;
        
        if self.current_batch_index < self.batch_processing_queue.len() {
            let job = &self.batch_processing_queue[self.current_batch_index];
            
            // 1. Job cores normalized
            self.cached_state[job_batch_idx] = job.cores_required as f32 / self.job_cores_range.1 as f32;
            
            // 2. Job memory normalized
            self.cached_state[job_batch_idx + 1] = job.memory_required as f32 / self.job_memory_range.1 as f32;
            
            // 3. Job duration normalized
            self.cached_state[job_batch_idx + 2] = job.duration as f32 / self.job_duration_range.1 as f32;
            
            // 4. Binary is_deferred flag (1 if job has been waiting > 1 second, 0 otherwise)
            let current_waiting_time = self.current_time as f64 - job.submission_time;
            let is_deferred = if current_waiting_time > 1.0 { 1.0 } else { 0.0 };
            self.cached_state[job_batch_idx + 3] = is_deferred;
            
            // 5. Batch progress normalized with tanh - same scale as queue pressure
            // Using num_hosts as natural scale for consistency with queue pressure
            // This is more robust for real LSF where exact batch index might not be available
            let batch_scale = self.num_hosts as f32;
            self.cached_state[job_batch_idx + 4] = (self.current_batch_index as f32 / batch_scale).tanh();
            
            // Queue features start at index 5
            // 6. Queue pressure - using tanh with num_hosts as natural scale
            // num_hosts represents one round of scheduling capacity
            // tanh gives smooth saturation: queue=num_hosts → 0.76, queue=2*num_hosts → 0.96
            let jobs_in_cycle = if !self.batch_processing_queue.is_empty() {
                self.batch_processing_queue.len()
            } else {
                // Count jobs in queue plus jobs in buckets (includes deferred)
                self.job_queue.len() + self.job_buckets.iter().map(|b| b.jobs.len()).sum::<usize>()
            };
            let queue_scale = self.num_hosts as f32;
            self.cached_state[job_batch_idx + 5] = (jobs_in_cycle as f32 / queue_scale).tanh();
            
            // Calculate total resource pressure for all jobs
            let mut total_cores_needed = 0u32;
            let mut total_memory_needed = 0u32;
            
            // Sum resources from batch queue (consistent view) or fallback to individual queues
            if !self.batch_processing_queue.is_empty() {
                for queue_job in &self.batch_processing_queue {
                    total_cores_needed += queue_job.cores_required;
                    total_memory_needed += queue_job.memory_required;
                }
            } else {
                // Fallback for when not in batch mode
                for queue_job in &self.job_queue {
                    total_cores_needed += queue_job.cores_required;
                    total_memory_needed += queue_job.memory_required;
                }
                // Include jobs in buckets (deferred and waiting)
                for bucket in &self.job_buckets {
                    for queue_job in &bucket.jobs {
                        total_cores_needed += queue_job.cores_required;
                        total_memory_needed += queue_job.memory_required;
                    }
                }
            }
            
            // 7. Core pressure ratio 
            self.cached_state[job_batch_idx + 6] = (total_cores_needed as f32) / (self.total_cluster_cores as f32);
            
            // 8. Memory pressure ratio
            self.cached_state[job_batch_idx + 7] = (total_memory_needed as f32) / (self.total_cluster_memory as f32);
            
        }
  
        Ok(PyArray1::from_slice(py, &self.cached_state).to_owned())
    }
    
    pub fn needs_decision(&self) -> bool {
        self.current_batch_index < self.batch_processing_queue.len()
    }
    
    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        // Return detailed step information when explicitly requested
        let info = PyDict::new(py);
        info.set_item("queue_length", self.job_queue.len())?;
        info.set_item("batch_queue_length", self.batch_processing_queue.len())?;
        info.set_item("active_jobs", self.active_jobs.len())?;
        info.set_item("needs_decision", self.current_batch_index < self.batch_processing_queue.len())?;
        info.set_item("total_jobs_generated", self.total_jobs_generated)?;
        info.set_item("total_jobs_completed", self.total_jobs_completed)?;
        info.set_item("current_time", self.current_time)?;
        info.set_item("current_step", self.current_step)?;
        info.set_item("deferred_jobs", self.count_deferred_jobs())?;
        info.set_item("total_jobs_deferred", self.total_jobs_deferred)?;
        
        // Add makespan if available
        if let Some(makespan_time) = self.makespan {
            info.set_item("makespan", makespan_time)?;
        }
        
        Ok(info.into())
    }
    
    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.original_seed = seed;
        if let Some(s) = seed {
            self.rng = StdRng::seed_from_u64(s);
        } else {
            self.rng = StdRng::from_entropy();
        }
    }
    
    pub fn get_host_configs(&self, py: Python) -> PyResult<PyObject> {
        let hosts_list = pyo3::types::PyList::empty(py);
        
        for (i, host) in self.hosts.iter().enumerate() {
            let host_dict = pyo3::types::PyDict::new(py);
            host_dict.set_item("host_id", i)?;
            host_dict.set_item("total_cores", host.total_cores)?;
            host_dict.set_item("total_memory", host.total_memory)?;
            hosts_list.append(host_dict)?;
        }
        
        Ok(hosts_list.to_object(py))
    }
    
    pub fn get_job_schedule(&self, py: Python) -> PyResult<PyObject> {
        let schedule_dict = pyo3::types::PyDict::new(py);
        
        schedule_dict.set_item("job_arrival_schedule", self.job_arrival_schedule.clone())?;
        schedule_dict.set_item("job_cores_schedule", self.job_cores_schedule.clone())?;
        schedule_dict.set_item("job_memory_schedule", self.job_memory_schedule.clone())?;
        schedule_dict.set_item("job_duration_schedule", self.job_duration_schedule.clone())?;
        schedule_dict.set_item("total_jobs_in_pool", self.total_jobs_in_pool)?;
        schedule_dict.set_item("max_time", self.max_time)?;
        schedule_dict.set_item("max_jobs_per_step", self.max_jobs_per_step)?;
        schedule_dict.set_item("num_hosts", self.num_hosts)?;
        schedule_dict.set_item("host_cores_range", self.host_cores_range)?;
        schedule_dict.set_item("host_memory_range", self.host_memory_range)?;
        schedule_dict.set_item("job_cores_range", self.job_cores_range)?;
        schedule_dict.set_item("job_memory_range", self.job_memory_range)?;
        schedule_dict.set_item("job_duration_range", self.job_duration_range)?;
        
        Ok(schedule_dict.to_object(py))
    }
    
    pub fn get_cluster_info(&self, py: Python) -> PyResult<PyObject> {
        let info_dict = pyo3::types::PyDict::new(py);
        info_dict.set_item("total_cluster_cores", self.total_cluster_cores)?;
        info_dict.set_item("total_cluster_memory", self.total_cluster_memory)?;
        info_dict.set_item("num_hosts", self.num_hosts)?;
        info_dict.set_item("host_cores_range", self.host_cores_range)?;
        info_dict.set_item("host_memory_range", self.host_memory_range)?;
        Ok(info_dict.to_object(py))
    }
    
    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        
        // Use actual episode duration (current_time) since episodes now run until completion
        let episode_duration = self.current_time.max(1) as f64; // Actual seconds elapsed
        let avg_core_util = (self.core_util_sum / episode_duration) as f32;
        
        let avg_memory_util = (self.memory_util_sum / episode_duration) as f32;
        
        // Calculate meaningful policy behavior metrics
        
        // Calculate deferral rate using total deferral events counter
        let defer_rate = if self.total_jobs_generated > 0 {
            self.total_jobs_deferred as f64 / self.total_jobs_generated as f64
        } else {
            -0.02
        };
        
        
        
        let metrics = PyDict::new(py);
        
        // Calculate average waiting time for ALL jobs in episode (completed + currently waiting)
        let mut total_current_waiting_time = 0.0;
        
        // Add waiting time for jobs currently in batch queue
        for job in &self.batch_processing_queue {
            let waiting_time = self.current_time as f64 - job.submission_time;
            total_current_waiting_time += waiting_time;
        }
        
        for job in &self.job_queue {
            let waiting_time = self.current_time as f64 - job.submission_time;
            total_current_waiting_time += waiting_time;
        }
        
        // Add waiting time for jobs in buckets (includes deferred)
        for bucket in &self.job_buckets {
            for job in &bucket.jobs {
                let waiting_time = self.current_time as f64 - job.submission_time;
                total_current_waiting_time += waiting_time;
            }
        }
        
        // Total waiting time = completed jobs + currently waiting jobs
        let total_episode_waiting_time = self.total_waiting_time_all_jobs + total_current_waiting_time;
        
        // Average across all jobs generated in this episode
        let avg_waiting_time = if self.total_jobs_generated > 0 {
            total_episode_waiting_time / self.total_jobs_generated as f64
        } else {
            0.0
        };

        // Job completion metrics
        metrics.set_item("total_jobs_completed", self.total_jobs_completed)?;
        metrics.set_item("avg_waiting_time", avg_waiting_time)?;
        metrics.set_item("defer_rate", defer_rate)?;
        
        // Makespan metric (time when all jobs finished)
        if let Some(makespan_time) = self.makespan {
            metrics.set_item("makespan", makespan_time as f64)?;
        } else {
            metrics.set_item("makespan", py.None())?;
        }
        
        // Time-based utilization metrics (averaged over seconds, not timesteps)
        metrics.set_item("avg_host_core_utilization", avg_core_util)?;
        metrics.set_item("avg_host_memory_utilization", avg_memory_util)?;
        
        Ok(metrics.into())
    }
    
}

// Private implementation methods
impl ClusterSchedulerEnv {
    fn generate_deterministic_job_schedule(
        max_time: usize,
        max_jobs_per_step: usize,
        job_cores_range: (u32, u32),
        job_memory_range: (u32, u32),
        job_duration_range: (u32, u32),
        use_skewed_arrivals: bool,
        rng: &mut StdRng,
    ) -> (Vec<usize>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        let mut job_arrival_schedule = Vec::with_capacity(max_time);
        let mut job_cores_schedule = Vec::new();
        let mut job_memory_schedule = Vec::new();
        let mut job_duration_schedule = Vec::new();
        
        let cores_dist = Uniform::from(job_cores_range.0..=job_cores_range.1);
        let duration_dist = Uniform::from(job_duration_range.0..=job_duration_range.1);
        
        // Generate job arrival schedule for each timestep
        for _timestep in 0..max_time {
            let num_jobs = Self::generate_job_arrivals(max_jobs_per_step, use_skewed_arrivals, rng);
            job_arrival_schedule.push(num_jobs);
            
            // Generate properties for jobs arriving at this timestep
            for _job in 0..num_jobs {
                let cores = cores_dist.sample(rng);
                job_cores_schedule.push(cores);
                
                // Generate memory with correlation to cores (EDA job patterns)
                let memory = Self::generate_correlated_job_memory(cores, job_cores_range, job_memory_range, rng);
                job_memory_schedule.push(memory);
                
                job_duration_schedule.push(duration_dist.sample(rng));
            }
        }
        let total_jobs_in_pool = job_cores_schedule.len();
        
        (job_arrival_schedule, job_cores_schedule, job_memory_schedule, job_duration_schedule, total_jobs_in_pool)
    }
    
    fn generate_realistic_host_config(
        cores_range: (u32, u32), 
        memory_range: (u32, u32), 
        rng: &mut StdRng
    ) -> (u32, u32) {
        // Common core configurations for EDA clusters
        const COMMON_CORES: &[u32] = &[8, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128];
        
        // Common memory configurations (in MB - matching env units)
        const COMMON_MEMORY_MB: &[u32] = &[
            32 * 1024,   // 32GB
            48 * 1024,   // 48GB  
            64 * 1024,   // 64GB
            96 * 1024,   // 96GB
            128 * 1024,  // 128GB
            192 * 1024,  // 192GB
            256 * 1024,  // 256GB
            384 * 1024,  // 384GB
            512 * 1024,  // 512GB
            768 * 1024,  // 768GB
            1024 * 1024, // 1024GB
        ];
        
        // Filter cores within range
        let valid_cores: Vec<u32> = COMMON_CORES
            .iter()
            .filter(|&&c| c >= cores_range.0 && c <= cores_range.1)
            .cloned()
            .collect();
        
        // Filter memory within range (memory_range already in MB)
        let valid_memory_mb: Vec<u32> = COMMON_MEMORY_MB
            .iter()
            .filter(|&&m| m >= memory_range.0 && m <= memory_range.1)
            .cloned()
            .collect();
        
        // Pick random valid configuration
        let cores = if valid_cores.is_empty() {
            // Fallback if no common cores in range
            cores_range.0 + (cores_range.1 - cores_range.0) / 2
        } else {
            valid_cores[rng.gen_range(0..valid_cores.len())]
        };
        
        let memory_mb = if valid_memory_mb.is_empty() {
            // Fallback if no common memory in range  
            memory_range.0 + (memory_range.1 - memory_range.0) / 2
        } else {
            valid_memory_mb[rng.gen_range(0..valid_memory_mb.len())]
        };
        
        (cores, memory_mb)
    }
    
    fn generate_correlated_job_memory(
        cores: u32,
        cores_range: (u32, u32),
        memory_range: (u32, u32),
        rng: &mut StdRng
    ) -> u32 {
        // EDA job patterns show correlation between cores and memory
        // Memory is always in multiples of 1024 MB (1 GB)
        
        // Normalize cores to 0-1 range
        let cores_norm = (cores - cores_range.0) as f32 / (cores_range.1 - cores_range.0).max(1) as f32;
        
        // Determine job type probabilistically
        let job_type_roll = rng.gen::<f32>();
        
        // Convert memory range to GB units for easier calculation
        let min_gb = (memory_range.0 / 1024).max(1);
        let max_gb = memory_range.1 / 1024;
        
        let memory_gb = if cores_norm < 0.2 {
            // Low core jobs (bottom 20% of core range)
            if job_type_roll < 0.6 {
                // 60%: Small compile/lint jobs - low memory
                // Use 5-20% of memory range
                let mem_fraction = rng.gen_range(0.05..0.2);
                let gb = min_gb + ((max_gb - min_gb) as f32 * mem_fraction) as u32;
                gb.max(min_gb).min(max_gb)
            } else if job_type_roll < 0.85 {
                // 25%: Memory-intensive timing/analysis - high memory, few cores  
                // Use 50-90% of memory range
                let mem_fraction = rng.gen_range(0.5..0.9);
                let gb = min_gb + ((max_gb - min_gb) as f32 * mem_fraction) as u32;
                gb.max(min_gb).min(max_gb)
            } else {
                // 15%: Medium synthesis - moderate memory
                // Use 20-40% of memory range
                let mem_fraction = rng.gen_range(0.2..0.4);
                let gb = min_gb + ((max_gb - min_gb) as f32 * mem_fraction) as u32;
                gb.max(min_gb).min(max_gb)
            }
        } else if cores_norm < 0.6 {
            // Medium core jobs (20-60% of core range): typically P&R or simulation
            // Use 30-60% of memory range
            let mem_fraction = rng.gen_range(0.3..0.6);
            let gb = min_gb + ((max_gb - min_gb) as f32 * mem_fraction) as u32;
            gb.max(min_gb).min(max_gb)
        } else {
            // High core jobs (top 40% of core range): large P&R or parallel verification
            // Use 40-80% of memory range (scales with cores but not linearly)
            let base_fraction = 0.4 + cores_norm * 0.3; // 0.4 to 0.7 based on cores
            let noise = rng.gen_range(-0.1..0.1);
            let mem_fraction = (base_fraction + noise).max(0.4).min(0.8);
            let gb = min_gb + ((max_gb - min_gb) as f32 * mem_fraction) as u32;
            gb.max(min_gb).min(max_gb)
        };
        
        // Convert back to MB (always multiple of 1024)
        memory_gb * 1024
    }
    

    fn generate_job_arrivals(max_jobs_per_step: usize, use_skewed: bool, rng: &mut StdRng) -> usize {
        if use_skewed {
            // Create a skewed distribution favoring higher values
            // Target: For max=50, average should be around 35 (70% of max)
            // Use Beta distribution transformed to desired range
            
            // Beta(2, 1) gives us right-skewed distribution (more values near 1.0)
            // For stronger skew toward high values, use Beta(3, 1) or even Beta(4, 1)
            let alpha = 2.0;
            let beta = 1.0;
            
            // Generate two uniform samples to create Beta distribution using acceptance-rejection
            let sample = loop {
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                
                // Simple Beta(alpha, beta) using ratio of uniforms method
                let x = u1.powf(1.0 / alpha);
                let y = u2.powf(1.0 / beta);
                let sum = x + y;
                
                if sum <= 1.0 {
                    break x / sum;
                }
            };
            
            // Transform to range [1, max_jobs_per_step]
            let scaled = 1.0 + sample * (max_jobs_per_step - 1) as f64;
            
            // Clamp and round to ensure valid range
            scaled.round().max(1.0).min(max_jobs_per_step as f64) as usize
        } else {
            // Uniform distribution over [1, max_jobs_per_step]
            rng.gen_range(1..=max_jobs_per_step)
        }
    }
    
    fn simulate_job_submissions(&mut self) {
        // Batch scheduling mode: Start batch processing if no current batch
        if self.batch_processing_queue.is_empty() && self.current_batch_index == 0 {
            self.start_batch_processing();
        }
    }
    
    fn start_batch_processing(&mut self) {
        // Add new jobs from job_queue to buckets
        while let Some(job) = self.job_queue.pop_front() {
            self.add_job_to_bucket(job);
        }
        
        // Build batch processing queue from buckets (in bucket creation order)
        // This includes both new and deferred jobs, maintaining FCFS within each bucket
        self.batch_processing_queue.clear();
        for bucket in &self.job_buckets {
            for job in &bucket.jobs {
                self.batch_processing_queue.push_back(job.clone());
            }
        }
        
        self.current_batch_index = 0;
        self.scheduling_attempts_this_batch = 0;  // Reset attempts counter for new batch
    }
    
    fn generate_bucket_key(job_id: u32, _cores: u32, _memory: u32) -> String {
        // Flexible function for generating bucket keys
        // Can be easily modified in the future to change grouping strategy
        // Current strategy: unique key per job (effectively no bucketing)
        // Alternative strategies:
        // - Group by resources: format!("c_{}_m_{}", cores, memory)
        // - Group by cores only: format!("c_{}", cores)
        // - Group by memory ranges: format!("m_{}", memory / 1024)
        format!("job_{}", job_id)
    }
    
    fn add_job_to_bucket(&mut self, job: Job) {
        // Generate key for this job  
        let key = Self::generate_bucket_key(job.id, job.cores_required, job.memory_required);
        
        // Find existing bucket or create new one
        let bucket_idx = self.job_buckets.iter().position(|b| b.bucket_key == key);
        
        match bucket_idx {
            Some(idx) => {
                // Add to existing bucket (maintains FCFS within bucket)
                self.job_buckets[idx].jobs.push_back(job);
            }
            None => {
                // Create new bucket at the end (maintains creation order)
                let mut new_bucket = JobBucket {
                    bucket_key: key,
                    jobs: VecDeque::new(),
                    host_priorities: None,  // Will be calculated when first job is scheduled
                };
                new_bucket.jobs.push_back(job);
                self.job_buckets.push(new_bucket);
            }
        }
    }
    
    fn add_new_jobs_to_queue(&mut self) {
        let timestep = self.current_time as usize;
        
        // Check if we're at or beyond max_time - stop generating new jobs
        if timestep >= self.max_time {
            return; // No more jobs to generate
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
        
        // First: Add new jobs for this second
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
                self.current_time as f64,  // This is the arrival time for this job
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
            if event.completion_time > self.current_time as f64 {
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
        // Only update once per second
        if self.current_time > self.last_stats_update_time {
            // Update all hosts and get utilization values directly
            for (i, host) in self.hosts.iter_mut().enumerate() {
                let (core_util, memory_util) = host.update_utilization_history();
                self.host_core_utils[i] = core_util;
                self.host_memory_utils[i] = memory_util;
            }
            
            
            // Update time-based statistics
            self.update_utilization_statistics_optimized(self.current_time);
        }
    }
    
    fn update_utilization_statistics_optimized(&mut self, current_time: u64) {
        // Time check already done in caller, just update statistics
        self.last_stats_update_time = current_time;
        
        // Calculate current average utilization across all hosts
        let avg_core_util = self.host_core_utils.iter().sum::<f32>() / self.num_hosts as f32;
        let avg_memory_util = self.host_memory_utils.iter().sum::<f32>() / self.num_hosts as f32;
        
        // Update running statistics
        self.core_util_sum += avg_core_util as f64;
        self.memory_util_sum += avg_memory_util as f64;
        
    }
    
    fn finish_batch_processing(&mut self) {
        // Clear batch and reset index
        self.batch_processing_queue.clear();
        self.current_batch_index = 0;
        
        // Clear host priorities for all buckets (will be recalculated in next batch)
        for bucket in &mut self.job_buckets {
            bucket.host_priorities = None;
        }
        
        // Remove empty buckets (all jobs in that bucket successfully scheduled)
        self.job_buckets.retain(|bucket| !bucket.jobs.is_empty());
    }
    
    fn count_deferred_jobs(&self) -> usize {
        // Count all jobs in buckets that have been deferred at least once
        self.job_buckets.iter()
            .flat_map(|bucket| &bucket.jobs)
            .filter(|job| job.deferred_count > 0)
            .count()
    }
    
    fn schedule_single_job_from_batch(&mut self, job: Job, action: &[f64]) -> usize {
        // Track that we're attempting to allocate resources for this job
        
        // First check if this job can be scheduled on ANY host
        let can_be_scheduled = self.hosts.iter().any(|host| {
            host.total_cores >= job.cores_required && host.total_memory >= job.memory_required
        });
        
        if !can_be_scheduled {
            // Job requires more resources than any host can provide
            println!("WARNING: Job {} cannot be scheduled on any host (requires {} cores, {} MB memory)", 
                     job.id, job.cores_required, job.memory_required);
            return 0; // Don't defer, just skip
        }
        
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
                    job_to_schedule.start_time = Some(self.current_time as f64);
                    
                    // Record waiting time for this job
                    let waiting_time = self.current_time as f64 - job_to_schedule.submission_time;
                    self.total_waiting_time_all_jobs += waiting_time;
                    
                    // Schedule completion
                    let completion_time = (self.current_time + job_to_schedule.duration as u64) as f64;
                    let job_id = job_to_schedule.id;
                    self.completion_heap.push(CompletionEvent {
                        completion_time,
                        job_id,
                    });
                    
                    // Remove the job from its bucket since it's scheduled
                    for bucket in &mut self.job_buckets {
                        if bucket.bucket_key == job.bucket_key {
                            bucket.jobs.retain(|j| j.id != job_id);
                            break;
                        }
                    }
                    
                    self.active_jobs.insert(job_id, job_to_schedule);
                    return 1; // Successfully scheduled one job
                }
            }
        }
        
        // Job couldn't be scheduled - update deferred count in the bucket
        // Find the job in its bucket and update its deferred count
        for bucket in &mut self.job_buckets {
            if bucket.bucket_key == job.bucket_key {
                // Find the job in this bucket
                for bucket_job in &mut bucket.jobs {
                    if bucket_job.id == job.id {
                        bucket_job.deferred_count += 1;
                        self.total_jobs_deferred += 1;
                        break;
                    }
                }
                break;
            }
        }
        
        0 // No job scheduled
    }

    

    
    fn calculate_pure_resource_utilization_reward(&self) -> f64 {
        let mut rewards = 0.0;
        
        for i in 0..self.num_hosts {
            let core_util = self.host_core_utils[i] as f64;
            let memory_util = self.host_memory_utils[i] as f64;

            rewards += (core_util + memory_util)/2.0;

        }
        rewards = rewards / self.num_hosts as f64;
        
        rewards
    }
}

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
    max_time: usize,  // Maximum time for job generation (after this, wait for completion)
    
    // State
    hosts: Vec<Host>,
    host_core_utils: Vec<f32>,           // Host core utilization in last second (LSF accessible)
    host_memory_utils: Vec<f32>,         // Host memory utilization in last second (LSF accessible)
    
    // Time-based utilization statistics (updated every second)
    core_util_sum: f64,                 // Sum of core utilization values
    memory_util_sum: f64,               // Sum of memory utilization values  
    utilization_sample_count: u64,      // Number of time samples collected
    last_stats_update_time: u64,        // Last time statistics were updated
    job_queue: VecDeque<Job>,
    submission_queue: VecDeque<Job>,  // Jobs waiting for scheduling decisions
    deferred_jobs: VecDeque<Job>,     // Jobs that couldn't be scheduled, retry later
    active_jobs: HashMap<u32, Job>,
    completion_heap: BinaryHeap<CompletionEvent>,
    current_time: u64,  // Integer seconds only
    jobs_attempted_this_second: usize,  // Total jobs attempted in current second (deferred + new)
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
    
    // Waiting time tracking for all jobs in episode
    total_waiting_time_all_jobs: f64,  // Sum of waiting times for all jobs (completed + in progress)
    
    // Batch-wise reward tracking
    last_reward_time: u64,  // Last time we gave a batch reward
    
    // Makespan tracking
    makespan: Option<u64>,  // Time when all jobs finished (set when episode ends)
    
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
        host_memory_range = (131072, 524288),  // 128GB-512GB in MB
        job_cores_range = (1, 32),
        job_memory_range = (2048, 65536),      // 2GB-64GB in MB
        job_duration_range = (1, 60),
        max_jobs_per_step = 50,
        max_time = 4096,
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
                &mut rng
            );

        // Create hosts with realistic configurations
        let mut hosts = Vec::with_capacity(num_hosts);
        for i in 0..num_hosts {
            let (cores, memory) = Self::generate_realistic_host_config(host_cores_range, host_memory_range, &mut rng);
            hosts.push(Host::new(i, cores, memory, host_cores_range.1, host_memory_range.1));
        }
        
        let host_core_utils = vec![0.0; num_hosts];
        let host_memory_utils = vec![0.0; num_hosts];
        
        // Pre-allocate cached state with correct size
        let state_size = num_hosts * 4 + 2;
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
            hosts,
            host_core_utils,
            host_memory_utils,
            core_util_sum: 0.0,
            memory_util_sum: 0.0,
            utilization_sample_count: 0,
            last_stats_update_time: 0,
            job_queue: VecDeque::new(),
            submission_queue: VecDeque::new(),
            deferred_jobs: VecDeque::new(),
            active_jobs: HashMap::new(),
            completion_heap: BinaryHeap::new(),
            current_time: 0,
            jobs_attempted_this_second: 0,
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
            total_waiting_time_all_jobs: 0.0,
            last_reward_time: 0,
            makespan: None,
            rng,
            original_seed,
            cached_state,
        }
    }
    
    pub fn reset(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // Re-generate cluster configuration if we have original seed
        if let Some(seed) = self.original_seed {
            // Use seed for deterministic host generation during testing
            let mut cluster_rng = StdRng::seed_from_u64(seed);
            for host in &mut self.hosts {
                let (cores, memory) = Self::generate_realistic_host_config(
                    self.host_cores_range, 
                    self.host_memory_range, 
                    &mut cluster_rng
                );
                
                host.total_cores = cores;
                host.total_memory = memory;
                host.available_cores = cores;
                host.available_memory = memory;
                host.running_job_ids.clear();
                host.update_normalized_values(self.host_cores_range.1, self.host_memory_range.1);
            }
        } else {
            // Use random host generation for training diversity
            for host in &mut self.hosts {
                let (cores, memory) = Self::generate_realistic_host_config(
                    self.host_cores_range, 
                    self.host_memory_range, 
                    &mut self.rng
                );
                
                host.total_cores = cores;
                host.total_memory = memory;
                host.available_cores = cores;
                host.available_memory = memory;
                host.running_job_ids.clear();
                host.update_normalized_values(self.host_cores_range.1, self.host_memory_range.1);
            }
        }

        
        // Clear collections
        self.job_queue.clear();
        self.submission_queue.clear();
        self.deferred_jobs.clear();
        self.active_jobs.clear();
        self.completion_heap.clear();
        
        // Reset state
        self.current_time = 0;
        self.jobs_attempted_this_second = 0;
        self.current_step = 0;
        self.next_job_id = 0;
        
        // Reset metrics
        self.total_jobs_generated = 0;
        self.total_jobs_completed = 0;
        self.total_jobs_failed = 0;
        self.jobs_completed_this_step = 0;
        self.total_waiting_time_all_jobs = 0.0;
        
        // Reset batch tracking
        self.last_reward_time = 0;
        
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
        self.utilization_sample_count = 0;
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
        
        // CRITICAL FIX: Check time advancement condition BEFORE actually advancing time
        // This ensures reward calculation sees the same state that triggers time advancement
        let will_advance_time = self.submission_queue.is_empty() && self.job_queue.is_empty();
        
        // Calculate reward based on current state (before any changes)
        let total_reward = self.calculate_adaptive_efficiency_reward(will_advance_time, jobs_scheduled) as f32;
        
        // NOW advance time if needed (after reward calculation)
        if will_advance_time {
            self.current_time += 1;
            self.jobs_attempted_this_second = 0;
            // Generate jobs for this new time unit (includes deferred jobs)
            self.add_jobs_to_queue();
        }
        
        // Move jobs from main queue to submission queue
        self.simulate_job_submissions();
        
        self.current_step += 1;
        
        // Episode ends when all deterministic jobs are completed and no jobs remain
        let all_jobs_generated = self.current_time >= self.max_time as u64;
        let all_jobs_finished = self.job_queue.is_empty() && 
                               self.submission_queue.is_empty() && 
                               self.deferred_jobs.is_empty() && 
                               self.active_jobs.is_empty();
        
        // Safety check: If we're past max_time and stuck with no progress for too long
        let stuck_too_long = all_jobs_generated && 
                            self.current_time > (self.max_time as u64 + 1000) && 
                            !self.deferred_jobs.is_empty();
        
        let done = (all_jobs_generated && all_jobs_finished) || stuck_too_long;
        
        // Set makespan when episode ends (all jobs finished)
        if done && self.makespan.is_none() {
            self.makespan = Some(self.current_time);
        }
        
        let info = PyDict::new(py);
        info.set_item("jobs_scheduled", jobs_scheduled)?;
        info.set_item("queue_length", self.job_queue.len())?;
        info.set_item("submission_queue_length", self.submission_queue.len())?;
        info.set_item("active_jobs", self.active_jobs.len())?;
        info.set_item("needs_decision", !self.submission_queue.is_empty())?;
        info.set_item("total_jobs_generated", self.total_jobs_generated)?;
        info.set_item("total_jobs_completed", self.total_jobs_completed)?;
        info.set_item("total_jobs_failed", self.total_jobs_failed)?;
        info.set_item("current_time", self.current_time)?;
        
        let state = self.get_state(py)?;
        
        Ok((state, total_reward as f32, done, info.into()))
    }
    
    fn get_state(&mut self, py: Python) -> PyResult<Py<PyArray1<f32>>> {
        // Clear the cached state (already pre-allocated with correct size)
        self.cached_state.fill(0.0);
        
        // Fill state in format: [host1_core_util, host1_mem_util, host1_cores_norm, host1_mem_norm, ...]
        for i in 0..self.num_hosts {
            let base_idx = i * 4;
            self.cached_state[base_idx] = self.host_core_utils[i];     // Host core utilization in last second (LSF accessible)
            self.cached_state[base_idx + 1] = self.host_memory_utils[i];    // Host memory utilization in last second (LSF accessible)
            self.cached_state[base_idx + 2] = self.hosts[i].normalized_cores;  // Pre-calculated normalized cores
            self.cached_state[base_idx + 3] = self.hosts[i].normalized_memory; // Pre-calculated normalized memory
        }
        
        // Single job info (only 1 job observable) - at end of state vector
        let job_batch_idx = self.num_hosts * 4;
        
        if !self.submission_queue.is_empty() {
            let job = &self.submission_queue[0];
            self.cached_state[job_batch_idx] = job.cores_required as f32 / self.job_cores_range.1 as f32;
            self.cached_state[job_batch_idx + 1] = job.memory_required as f32 / self.job_memory_range.1 as f32;
        }
  
        Ok(PyArray1::from_slice(py, &self.cached_state).to_owned())
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
    
    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        
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
        
        
        let metrics = PyDict::new(py);
        
        // Calculate average waiting time for ALL jobs in episode (completed + currently waiting)
        let mut total_current_waiting_time = 0.0;
        
        // Add waiting time for jobs currently in queues
        for job in &self.submission_queue {
            let waiting_time = self.current_time as f64 - job.submission_time;
            total_current_waiting_time += waiting_time;
        }
        
        for job in &self.job_queue {
            let waiting_time = self.current_time as f64 - job.submission_time;
            total_current_waiting_time += waiting_time;
        }
        
        for job in &self.deferred_jobs {
            let waiting_time = self.current_time as f64 - job.submission_time;
            total_current_waiting_time += waiting_time;
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
    
    pub fn get_queue_states(&self, py: Python) -> PyResult<PyObject> {
        // Expose internal queue states for visualization
        let queue_states = PyDict::new(py);
        
        // Job queue info
        queue_states.set_item("job_queue_length", self.job_queue.len())?;
        queue_states.set_item("submission_queue_length", self.submission_queue.len())?;
        queue_states.set_item("deferred_jobs_length", self.deferred_jobs.len())?;
        queue_states.set_item("active_jobs_count", self.active_jobs.len())?;
        
        // Current time info
        queue_states.set_item("current_time", self.current_time)?;
        queue_states.set_item("max_time", self.max_time)?;
        queue_states.set_item("current_step", self.current_step)?;
        
        // Job statistics
        queue_states.set_item("total_jobs_generated", self.total_jobs_generated)?;
        queue_states.set_item("total_jobs_completed", self.total_jobs_completed)?;
        queue_states.set_item("total_jobs_failed", self.total_jobs_failed)?;
        queue_states.set_item("jobs_completed_this_step", self.jobs_completed_this_step)?;
        
        // Completion heap size
        queue_states.set_item("pending_completions", self.completion_heap.len())?;
        
        Ok(queue_states.to_object(py))
    }
    
    pub fn get_visualization_data(&self, py: Python) -> PyResult<PyObject> {
        // Get comprehensive data for web dashboard visualization
        let viz_dict = pyo3::types::PyDict::new(py);
        
        // Time information
        viz_dict.set_item("current_time", self.current_time)?;
        viz_dict.set_item("max_time", self.max_time)?;
        
        // Job counts
        viz_dict.set_item("total_jobs_generated", self.total_jobs_generated)?;
        viz_dict.set_item("total_jobs_completed", self.total_jobs_completed)?;
        viz_dict.set_item("total_jobs_failed", self.total_jobs_failed)?;
        viz_dict.set_item("active_jobs_count", self.active_jobs.len())?;
        
        // Queue information
        viz_dict.set_item("job_queue_length", self.job_queue.len())?;
        viz_dict.set_item("submission_queue_length", self.submission_queue.len())?;
        viz_dict.set_item("deferred_jobs_length", self.deferred_jobs.len())?;
        
        // Decision state
        viz_dict.set_item("needs_decision", !self.submission_queue.is_empty())?;
        
        // Episode state
        let all_jobs_generated = self.current_time >= self.max_time as u64;
        let no_pending_work = self.job_queue.is_empty() && 
                             self.submission_queue.is_empty() && 
                             self.deferred_jobs.is_empty() && 
                             self.active_jobs.is_empty();
        let episode_done = all_jobs_generated && no_pending_work;
        viz_dict.set_item("episode_done", episode_done)?;
        viz_dict.set_item("all_jobs_generated", all_jobs_generated)?;
        
        // Host utilization data
        let hosts_list = pyo3::types::PyList::empty(py);
        for (i, host) in self.hosts.iter().enumerate() {
            let host_dict = pyo3::types::PyDict::new(py);
            host_dict.set_item("id", i)?;
            host_dict.set_item("cpu_util", self.host_core_utils[i] * 100.0)?; // Convert to percentage
            host_dict.set_item("memory_util", self.host_memory_utils[i] * 100.0)?; // Convert to percentage
            host_dict.set_item("total_cores", host.total_cores)?;
            host_dict.set_item("total_memory", host.total_memory)?;
            hosts_list.append(host_dict)?;
        }
        viz_dict.set_item("hosts", hosts_list)?;
        
        Ok(viz_dict.to_object(py))
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
            let num_jobs = rng.gen_range(1..=max_jobs_per_step);
            job_arrival_schedule.push(num_jobs);
            
            // Generate properties for jobs arriving at this timestep
            for _job in 0..num_jobs {
                job_cores_schedule.push(cores_dist.sample(rng));
                job_memory_schedule.push(Self::generate_realistic_job_memory(job_memory_range, rng));
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
    
    fn generate_realistic_job_memory(memory_range: (u32, u32), rng: &mut StdRng) -> u32 {
        // Common job memory configurations for EDA workloads (in MB)
        const COMMON_JOB_MEMORY_MB: &[u32] = &[
            512,      // 512MB - small jobs (lint, quick synthesis)
            768,      // 768MB - small-medium jobs
            1024,     // 1GB - medium jobs (block-level P&R)
            1536,     // 1.5GB - medium-large jobs
            2048,     // 2GB - large jobs (medium synthesis)
            3072,     // 3GB - large jobs (full-chip P&R)
            4096,     // 4GB - large simulations
            6144,     // 6GB - very large jobs
            8192,     // 8GB - huge jobs
            12288,    // 12GB - massive jobs
            16384,    // 16GB - maximum typical EDA job
        ];
        
        // Filter memory within range
        let valid_memory: Vec<u32> = COMMON_JOB_MEMORY_MB
            .iter()
            .filter(|&&m| m >= memory_range.0 && m <= memory_range.1)
            .cloned()
            .collect();
        
        if valid_memory.is_empty() {
            // Fallback if no common memory in range
            memory_range.0 + (memory_range.1 - memory_range.0) / 2
        } else {
            valid_memory[rng.gen_range(0..valid_memory.len())]
        }
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
        
        // Always move deferred jobs back to queue first (even after max_time)
        while let Some(deferred_job) = self.deferred_jobs.pop_front() {
            self.job_queue.push_back(deferred_job);
        }
        
        // Check if we're at or beyond max_time - stop generating new jobs
        if timestep >= self.max_time {
            // Still need to track attempted jobs for time advancement
            self.jobs_attempted_this_second = self.job_queue.len();
            return; // No more jobs to generate, but deferred jobs were added
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
        
        // Track total jobs to attempt this second
        self.jobs_attempted_this_second = self.job_queue.len();
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
        self.utilization_sample_count += 1;
    }

    // // Old func for backup
    // fn update_utilization_statistics(&mut self) {
    //     let current_time = self.current_time;
    //     if current_time > self.last_stats_update_time {
    //         self.update_utilization_statistics_optimized(current_time);
    //     }
    // }
    
    // Removed create_job_buckets - no longer needed for single job scheduling
    
    fn schedule_single_job_from_submission(&mut self, job: Job, action: &[f64]) -> usize {
        // First check if this job can be scheduled on ANY host
        let can_be_scheduled = self.hosts.iter().any(|host| {
            host.total_cores >= job.cores_required && host.total_memory >= job.memory_required
        });
        
        if !can_be_scheduled {
            // Job requires more resources than any host can provide
            // Mark as failed to prevent infinite loop
            self.total_jobs_failed += 1;
            println!("WARNING: Job {} cannot be scheduled on any host (requires {} cores, {} MB memory)", 
                     job.id, job.cores_required, job.memory_required);
            return 0; // Don't defer, just fail
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
        // Advance time when we've attempted all jobs for this second
        // Condition 1: submission_queue is empty (no job currently needs a decision)
        // Condition 2: job_queue is empty (no more jobs to move to submission queue)
        // Deferred jobs are safely stored and will be added back next second
        if self.submission_queue.is_empty() && self.job_queue.is_empty() {
            // Advance by full second
            self.current_time += 1;
            self.jobs_attempted_this_second = 0;
            true  // Signal to add jobs for new second
        } else {
            false
        }
    }
    

    fn calculate_resource_scheduling_reward(&mut self, _scheduled_jobs: usize) -> f64 {
        // Pure cluster health reward - action is host priorities, job will find a host
        // Focus on cluster resource balance and efficiency from priority rankings
        
        let mut cluster_balance_score = 0.0;
        let mut total_efficiency_score = 0.0;
        let mut active_hosts = 0;
        
        for i in 0..self.num_hosts {
            let core_util = self.host_core_utils[i] as f64;
            let memory_util = self.host_memory_utils[i] as f64;
            
            if core_util > 0.01 || memory_util > 0.01 {
                active_hosts += 1;
                
                // Reward balanced resource usage within each host
                let balance = 1.0 - (core_util - memory_util).abs();
                cluster_balance_score += balance;
                
                // Reward effective utilization 
                let efficiency = core_util.min(memory_util);
                total_efficiency_score += efficiency;
            }
        }
        
        let avg_balance = if active_hosts > 0 {
            cluster_balance_score / active_hosts as f64
        } else { 0.0 };
        
        let avg_efficiency = if active_hosts > 0 {
            total_efficiency_score / active_hosts as f64
        } else { 0.0 };
        
        // Pure cluster quality reward
        avg_balance * 0.5 + avg_efficiency * 0.5
    }

    fn calculate_adaptive_efficiency_reward(&self, will_advance_time: bool, scheduled_jobs: usize) -> f64 {
        let mut reward = 0.0;
        
        // Primary reward: Time advancement penalty (makespan optimization)
        if will_advance_time {
            reward -= 1.0;
        }
        
        // Secondary reward: Scheduling quality (only when actively scheduling)
        if scheduled_jobs > 0 {
            reward += 0.02;  // Base reward for making progress
            
            // Threshold-free utilization reward: adaptive to actual cluster state
            let cluster_utilization = self.get_current_cluster_utilization();
            
            // Linear reward for utilization (higher utilization = better resource usage)
            reward += cluster_utilization * 0.05;
            
            // Quadratic bonus for high efficiency (increasingly rewards optimal usage)
            let efficiency_bonus = cluster_utilization * cluster_utilization * 0.02;
            reward += efficiency_bonus;
        }
        
        reward
    }
    
    fn get_current_cluster_utilization(&self) -> f64 {
        let total_available: u32 = self.hosts.iter().map(|h| h.available_cores).sum();
        let total_capacity: u32 = self.hosts.iter().map(|h| h.total_cores).sum();
        1.0 - (total_available as f64 / total_capacity as f64)
    }

}

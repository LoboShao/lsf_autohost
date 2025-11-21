use std::collections::{BinaryHeap, HashMap, VecDeque};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;
use rand::distributions::{Uniform, Distribution};

use crate::job::{Job, JobStatus};
use crate::host::Host;
use crate::event::CompletionEvent;

/// Configuration for cluster environment
/// All child environments (HostSortingEnv, JobOrderingEnv) use this config
#[derive(Clone)]
pub struct ClusterConfig {
    pub num_hosts: usize,
    pub max_queue_length: Option<usize>,
    pub host_cores_range: (u32, u32),
    pub host_memory_range: (u32, u32),
    pub job_cores_range: (u32, u32),
    pub job_memory_range: (u32, u32),
    pub job_duration_range: (u32, u32),
    pub max_jobs_per_step: usize,
    pub max_time: usize,
    pub use_skewed_arrivals: bool,
    pub seed: Option<u64>,
}

impl ClusterConfig {
    /// Create new config with all parameters
    pub fn new(
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
        ClusterConfig {
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
        }
    }
}

/// Bucket for grouping jobs with same resource requirements
#[derive(Debug, Clone)]
pub struct JobBucket {
    pub bucket_key: String,
    pub jobs: VecDeque<Job>,
    // Shared host priorities for all jobs in this bucket
    pub host_priorities: Option<Vec<f32>>,
    // Number of jobs dispatched from this bucket in current cycle
    pub dispatched_count: usize,
}

/// Base environment with shared functionality for all scheduling environments
pub struct BaseClusterEnv {
    // Cluster configuration
    pub num_hosts: usize,
    pub max_queue_length: usize,
    pub host_cores_range: (u32, u32),
    pub host_memory_range: (u32, u32),
    pub job_cores_range: (u32, u32),
    pub job_memory_range: (u32, u32),
    pub job_duration_range: (u32, u32),
    pub max_jobs_per_step: usize,
    pub max_time: usize,
    pub use_skewed_arrivals: bool,

    // Cluster state
    pub hosts: Vec<Host>,
    pub host_core_utils: Vec<f32>,
    pub host_memory_utils: Vec<f32>,

    // Time-based utilization statistics
    pub core_util_sum: f64,
    pub memory_util_sum: f64,
    pub last_stats_update_time: u64,

    // Job management
    pub job_queue: VecDeque<Job>,
    pub current_bucket_index: usize,  // Which bucket we're currently scheduling (includes empties)
    pub buckets_processed: usize,  // Number of non-empty buckets processed (for batch progress)
    pub jobs_processed_in_batch: usize,  // Total jobs processed so far in this batch (for batch progress)
    pub total_jobs_in_current_batch: usize,  // Total jobs when batch started (for queue pressure)
    pub total_cores_in_current_batch: u32,  // Total cores needed by all jobs in batch (for resource pressure)
    pub total_memory_in_current_batch: u32,  // Total memory needed by all jobs in batch (for resource pressure)
    pub scheduling_attempts_this_batch: usize,
    pub job_buckets: Vec<JobBucket>,  // Bucket scheduling (LSF-style) - immutable during batch!
    pub active_jobs: HashMap<u32, Job>,
    pub completion_heap: BinaryHeap<CompletionEvent>,

    // Time and counters
    pub current_time: u64,
    pub current_step: usize,
    pub next_job_id: u32,

    // Deterministic job generation
    pub job_arrival_schedule: Vec<usize>,
    pub job_cores_schedule: Vec<u32>,
    pub job_memory_schedule: Vec<u32>,
    pub job_duration_schedule: Vec<u32>,
    pub total_jobs_in_pool: usize,
    pub jobs_moved_to_queue: usize,

    // Metrics
    pub total_jobs_generated: u32,
    pub total_jobs_completed: u32,
    pub total_jobs_deferred: u32,
    pub jobs_completed_this_step: u32,
    pub total_waiting_time_all_jobs: f64,
    pub makespan: Option<u64>,

    // RNG
    pub rng: StdRng,
    pub original_seed: Option<u64>,

    // Cluster resource totals (cached)
    pub total_cluster_cores: u32,
    pub total_cluster_memory: u32,

    // Cycle-level resource snapshot (cached at start of each scheduling cycle)
    pub cycle_available_cores: u32,
    pub cycle_available_memory: u32,
}

impl BaseClusterEnv {
    // ==================== Configuration & Construction ====================

    /// Create new BaseClusterEnv from config
    pub fn from_config(config: &ClusterConfig) -> Self {
        Self::new(
            config.num_hosts,
            config.max_queue_length,
            config.host_cores_range,
            config.host_memory_range,
            config.job_cores_range,
            config.job_memory_range,
            config.job_duration_range,
            config.max_jobs_per_step,
            config.max_time,
            config.use_skewed_arrivals,
            config.seed,
        )
    }

    /// Create new BaseClusterEnv (internal constructor)
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

        // Create hosts
        let mut hosts = Vec::with_capacity(num_hosts);
        for i in 0..num_hosts {
            let (cores, memory) = Self::generate_realistic_host_config(host_cores_range, host_memory_range, &mut rng);
            hosts.push(Host::new(i, cores, memory));
        }

        // Calculate total cluster resources
        let total_cluster_cores: u32 = hosts.iter().map(|h| h.total_cores).sum();
        let total_cluster_memory: u32 = hosts.iter().map(|h| h.total_memory).sum();

        let host_core_utils = vec![0.0; num_hosts];
        let host_memory_utils = vec![0.0; num_hosts];

        BaseClusterEnv {
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
            current_bucket_index: 0,
            buckets_processed: 0,
            jobs_processed_in_batch: 0,
            total_jobs_in_current_batch: 0,
            total_cores_in_current_batch: 0,
            total_memory_in_current_batch: 0,
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
            cycle_available_cores: total_cluster_cores,
            cycle_available_memory: total_cluster_memory,
        }
    }

    /// Reset the environment to initial state
    pub fn reset_base(&mut self) {
        // Re-generate cluster configuration (matching env_backup.rs behavior)
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

        // Apply configurations to hosts
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

        // Reset cycle-level snapshots to match new totals
        self.cycle_available_cores = self.total_cluster_cores;
        self.cycle_available_memory = self.total_cluster_memory;

        // Clear queues and jobs
        self.job_queue.clear();
        self.current_bucket_index = 0;
        self.buckets_processed = 0;
        self.jobs_processed_in_batch = 0;
        self.total_jobs_in_current_batch = 0;
        self.total_cores_in_current_batch = 0;
        self.total_memory_in_current_batch = 0;
        self.scheduling_attempts_this_batch = 0;
        self.job_buckets.clear();
        self.active_jobs.clear();
        self.completion_heap.clear();

        // Reset time and counters
        self.current_time = 0;
        self.current_step = 0;
        self.next_job_id = 0;
        self.jobs_moved_to_queue = 0;

        // Reset metrics
        self.total_jobs_generated = 0;
        self.total_jobs_completed = 0;
        self.total_jobs_deferred = 0;
        self.jobs_completed_this_step = 0;
        self.total_waiting_time_all_jobs = 0.0;
        self.makespan = None;

        // Reset utilization tracking
        self.host_core_utils.fill(0.0);
        self.host_memory_utils.fill(0.0);
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
    }

    // ==================== Job Generation & Host Configuration ====================

    /// Generate deterministic job schedule
    pub fn generate_deterministic_job_schedule(
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

    /// Generate realistic host configuration
    pub fn generate_realistic_host_config(
        cores_range: (u32, u32),
        memory_range: (u32, u32),
        rng: &mut StdRng,
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

    pub fn add_new_jobs_to_queue(&mut self) {
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

    // ==================== Batch & Bucket Management ====================

    /// Start batch processing (collect all jobs into buckets for scheduling)
    pub fn start_batch_processing(&mut self) {
        // Add new jobs from job_queue to buckets
        while let Some(job) = self.job_queue.pop_front() {
            self.add_job_to_bucket(job);
        }

        // Snapshot cluster available resources from historical utilization (for cycle-level state caching)
        // This simulates real LSF behavior where scheduling decisions see a snapshot from the previous cycle
        // Use historical utilization (from previous time step) instead of current real-time values
        self.cycle_available_cores = self.hosts.iter()
            .map(|h| {
                let util = h.get_core_utilization(); // Historical utilization
                let available_ratio = 1.0 - util;
                (available_ratio * h.total_cores as f32) as u32
            })
            .sum();

        self.cycle_available_memory = self.hosts.iter()
            .map(|h| {
                let util = h.get_memory_utilization(); // Historical utilization
                let available_ratio = 1.0 - util;
                (available_ratio * h.total_memory as f32) as u32
            })
            .sum();

        // Count total jobs and resources in all buckets (equivalent to batch_processing_queue in env.rs)
        self.total_jobs_in_current_batch = 0;
        self.total_cores_in_current_batch = 0;
        self.total_memory_in_current_batch = 0;

        for bucket in &self.job_buckets {
            for job in &bucket.jobs {
                self.total_jobs_in_current_batch += 1;
                self.total_cores_in_current_batch += job.cores_required;
                self.total_memory_in_current_batch += job.memory_required;
            }
        }

        // Reset dispatch counters from previous batch
        for bucket in &mut self.job_buckets {
            bucket.dispatched_count = 0;
        }

        // Reset indices to start from first bucket
        self.current_bucket_index = 0;
        self.buckets_processed = 0;
        self.jobs_processed_in_batch = 0;
        self.scheduling_attempts_this_batch = 0;
    }

    /// Finish batch processing (cleanup after batch completes)
    pub fn finish_batch_processing(&mut self) {
        // Jobs are already removed from buckets when scheduled (matching env.rs behavior)
        // Just reset priorities and clean up
        for bucket in &mut self.job_buckets {
            bucket.dispatched_count = 0;  // Reset counter for next batch
            bucket.host_priorities = None;  // Clear priorities for next batch
        }

        // Remove empty buckets
        self.job_buckets.retain(|bucket| !bucket.jobs.is_empty());

        // Reset indices
        self.current_bucket_index = 0;
        self.buckets_processed = 0;
        self.jobs_processed_in_batch = 0;
        self.total_jobs_in_current_batch = 0;
        self.total_cores_in_current_batch = 0;
        self.total_memory_in_current_batch = 0;
        self.scheduling_attempts_this_batch = 0;
    }

    /// Simulate job submissions - start batch processing if needed
    pub fn simulate_job_submissions(&mut self) {
        // Batch scheduling mode: Start batch processing if no current batch
        // Check if batch finished (total_jobs_in_current_batch is 0 means no active batch)
        if self.total_jobs_in_current_batch == 0 && self.current_bucket_index == 0 {
            self.start_batch_processing();
        }
    }

    /// Advance current_bucket_index to the next non-empty bucket
    pub fn skip_empty_buckets(&mut self) {
        while self.current_bucket_index < self.job_buckets.len()
              && self.job_buckets[self.current_bucket_index].jobs.is_empty() {
            self.current_bucket_index += 1;
        }
    }

    /// Get the current job being scheduled from current bucket
    /// Returns the first job from the current bucket (assumes empty buckets already skipped)
    pub fn get_current_job(&self) -> Option<&Job> {
        if self.current_bucket_index < self.job_buckets.len() {
            self.job_buckets[self.current_bucket_index].jobs.front()
        } else {
            None
        }
    }

    /// Get reference to current bucket being scheduled
    pub fn get_current_bucket(&self) -> Option<&JobBucket> {
        if self.current_bucket_index < self.job_buckets.len() {
            Some(&self.job_buckets[self.current_bucket_index])
        } else {
            None
        }
    }

    /// Generate bucket key for a job
    /// Current strategy: unique key per job (for fine-grained control)
    /// Can be modified to group by resources: format!("c_{}_m_{}", cores, memory)
    pub fn generate_bucket_key(job_id: u32, _cores: u32, _memory: u32) -> String {
        // Flexible function for generating bucket keys
        // Current: Each job gets its own bucket for maximum agent control
        format!("job_{}", job_id)
        // Alternative strategies (commented out):
        // format!("c_{}_m_{}", _cores, _memory)  // Group by exact resources
        // format!("size_{}", if cores <= 4 { "small" } else if cores <= 16 { "medium" } else { "large" })
    }

    /// Add a job to the appropriate bucket
    pub fn add_job_to_bucket(&mut self, job: Job) {
        // Generate key for this job
        let key = Self::generate_bucket_key(job.id, job.cores_required, job.memory_required);

        // Find existing bucket or create new one
        let bucket_index = self.job_buckets.iter().position(|b| b.bucket_key == key);

        match bucket_index {
            Some(idx) => {
                // Add to existing bucket
                self.job_buckets[idx].jobs.push_back(job);
            }
            None => {
                // Create new bucket at the end (maintains creation order)
                let mut new_bucket = JobBucket {
                    bucket_key: key,
                    jobs: VecDeque::new(),
                    host_priorities: None,
                    dispatched_count: 0,
                };
                new_bucket.jobs.push_back(job);
                self.job_buckets.push(new_bucket);
            }
        }
    }

    // ==================== Job Scheduling & Execution ====================

    /// Select host for a job - Default: First available host
    /// Returns host index if found, None otherwise
    /// Child classes can override this for different host selection policies
    pub fn select_host_for_job(&self, job: &Job) -> Option<usize> {
        // First check if any host can accommodate this job
        for (idx, host) in self.hosts.iter().enumerate() {
            if host.can_accommodate(job) {
                return Some(idx);
            }
        }
        None
    }

    /// Select hosts for a job (with priority order) - Default: First available
    /// Returns vector of (host_idx, priority) sorted by priority
    /// Child classes can override this for agent-based host selection
    pub fn get_host_priorities_for_job(&self, _job: &Job) -> Vec<(usize, f64)> {
        // Default: return all hosts with equal priority (first-available order)
        self.hosts.iter()
            .enumerate()
            .map(|(idx, _)| (idx, 1.0))
            .collect()
    }

    /// Schedule a job using host priorities (agent-provided or heuristic-based)
    /// Returns number of jobs scheduled (0 or 1)
    pub fn schedule_job_with_host_priorities(&mut self, job: Job, host_priorities: &[f64]) -> usize {
        // Check if job can be scheduled on any host
        let can_be_scheduled = self.hosts.iter().any(|host| {
            host.total_cores >= job.cores_required && host.total_memory >= job.memory_required
        });

        if !can_be_scheduled {
            println!("WARNING: Job {} cannot be scheduled on any host", job.id);
            return 0;
        }

        // Sort hosts by priorities (descending)
        let mut sorted_host_priorities: Vec<(usize, f64)> = host_priorities.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        sorted_host_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Try single-host scheduling with prioritized hosts
        for &(host_idx, _) in &sorted_host_priorities {
            if self.try_single_host_scheduling(&job, host_idx) {
                // Job scheduled successfully
                return 1;
            }
        }

        // Job couldn't be scheduled - it stays in its bucket
        // Update deferred count for job in bucket
        let job_id = job.id;
        let job_bucket_key = job.bucket_key.clone();
        for bucket in &mut self.job_buckets {
            if bucket.bucket_key == job_bucket_key {
                for bucket_job in &mut bucket.jobs {
                    if bucket_job.id == job_id {
                        bucket_job.deferred_count += 1;
                        self.total_jobs_deferred += 1;
                        break;
                    }
                }
                break;
            }
        }

        0
    }

    /// Process a selected bucket by scheduling its jobs with given host priorities
    /// This is the core bucket processing logic shared across all environments
    /// Returns (jobs_scheduled, finishing_batch_now)
    pub fn process_selected_bucket(&mut self, bucket_idx: usize, host_priorities: &[f64]) -> (usize, bool) {
        // Get jobs from selected bucket
        let bucket = &self.job_buckets[bucket_idx];
        let jobs_to_schedule: Vec<Job> = bucket.jobs.iter().cloned().collect();

        // Track the original bucket size (before removing any jobs)
        let original_bucket_size = jobs_to_schedule.len();

        // Try to schedule each job in the bucket using the same host priorities
        let mut bucket_scheduled = 0;
        let mut scheduled_job_ids = Vec::new();
        for bucket_job in jobs_to_schedule {
            let job_id = bucket_job.id;
            let scheduled = self.schedule_job_with_host_priorities(bucket_job, host_priorities);
            if scheduled > 0 {
                bucket_scheduled += 1;
                scheduled_job_ids.push(job_id);
            } else {
                // If one job can't be scheduled, stop trying for this bucket
                break;
            }
        }

        // Remove scheduled jobs immediately from bucket
        for job_id in scheduled_job_ids {
            self.job_buckets[bucket_idx].jobs.retain(|j| j.id != job_id);
        }

        // Increment attempts counter (we made one decision for the entire bucket)
        self.scheduling_attempts_this_batch += 1;

        // Count this as a processed bucket (for batch progress tracking)
        self.buckets_processed += 1;

        // Increment jobs processed counter by the ORIGINAL number of jobs in this bucket
        self.jobs_processed_in_batch += original_bucket_size;

        // Move to next bucket
        self.current_bucket_index += 1;

        // Check if we've finished all non-empty buckets
        let finishing_batch_now = self.total_jobs_in_current_batch > 0 && self.get_current_job().is_none();

        (bucket_scheduled, finishing_batch_now)
    }

    /// Try single-host scheduling
    pub fn try_single_host_scheduling(&mut self, job: &Job, host_idx: usize) -> bool {
        let host = &mut self.hosts[host_idx];

        if host.can_accommodate(job) {
            let mut scheduled_job = job.clone();

            // Allocate resources using allocate_job method (sets assigned_host)
            if host.allocate_job(&mut scheduled_job) {
                scheduled_job.status = JobStatus::Running;
                scheduled_job.start_time = Some(self.current_time as f64);

                // Record waiting time for this job (matching env_backup behavior)
                let waiting_time = self.current_time as f64 - scheduled_job.submission_time;
                self.total_waiting_time_all_jobs += waiting_time;

                // Add completion event
                let completion_time = (self.current_time + scheduled_job.duration as u64) as f64;
                self.completion_heap.push(CompletionEvent {
                    completion_time,
                    job_id: scheduled_job.id,
                });

                // Track active job
                self.active_jobs.insert(scheduled_job.id, scheduled_job);

                return true;
            }
        }
        false
    }

    /// Process job completions
    pub fn process_completions(&mut self) {
        while let Some(event) = self.completion_heap.peek() {
            if event.completion_time > self.current_time as f64 {
                break;
            }

            let event = self.completion_heap.pop().unwrap();

            if let Some(mut completed_job) = self.active_jobs.remove(&event.job_id) {
                completed_job.status = JobStatus::Completed;
                completed_job.end_time = Some(self.current_time as f64);

                // Release resources from host
                if let Some(host_id) = completed_job.assigned_host {
                    self.hosts[host_id].release_job(&completed_job);
                }

                self.total_jobs_completed += 1;
                self.jobs_completed_this_step += 1;
            }
        }
    }

    // ==================== Resource Tracking ====================

    /// Update host utilization metrics
    pub fn update_host_utilization(&mut self) {
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

    /// Calculate basic resource utilization
    pub fn calculate_utilization(&self) -> f32 {
        let mut used_cores = 0;
        let mut used_memory = 0;

        for host in &self.hosts {
            used_cores += host.total_cores - host.available_cores;
            used_memory += host.total_memory - host.available_memory;
        }

        let core_util = used_cores as f32 / self.total_cluster_cores as f32;
        let mem_util = used_memory as f32 / self.total_cluster_memory as f32;

        (core_util + mem_util) / 2.0
    }

    // ==================== Environment Control ====================

    /// Start batch processing if needed (no active batch and jobs in queue)
    pub fn maybe_start_batch(&mut self) {
        if self.total_jobs_in_current_batch == 0 && !self.job_queue.is_empty() {
            self.start_batch_processing();
        }
    }

    /// Update environment state: finish batch if needed, reset counters, update utilization, process completions
    pub fn update_environment_state(&mut self, finishing_batch_now: bool) {
        if finishing_batch_now {
            self.finish_batch_processing();
        }

        // Reset completion counter
        self.jobs_completed_this_step = 0;

        // Update utilization
        self.update_host_utilization();

        // Process completions
        self.process_completions();
    }

    /// Advance time if conditions are met and add new jobs
    pub fn maybe_advance_time(&mut self, will_advance_time: bool) {
        // Check if all jobs have been generated
        let all_jobs_generated = self.jobs_moved_to_queue >= self.total_jobs_in_pool;

        // Time advancement logic
        if will_advance_time && !all_jobs_generated {
            self.current_time += 1;
            // Generate new jobs for this time unit
            self.add_new_jobs_to_queue();
        } else if will_advance_time && all_jobs_generated {
            // If all jobs are generated but still processing, advance time without adding jobs
            self.current_time += 1;
        }

        // Move jobs from main queue to submission queue
        self.simulate_job_submissions();

        self.current_step += 1;
    }

    /// Check if episode is done and set makespan if needed
    pub fn check_episode_done(&mut self) -> bool {
        let done = self.current_time >= self.max_time as u64;

        // Set makespan when episode actually ends
        if done && self.makespan.is_none() {
            self.makespan = Some(self.current_time);
        }

        done
    }

    /// Check if environment is done
    pub fn is_done(&self) -> bool {
        let no_more_jobs = self.current_time >= self.max_time as u64
            && self.jobs_moved_to_queue >= self.total_jobs_in_pool;

        let all_complete = no_more_jobs
            && self.job_queue.is_empty()
            && self.job_buckets.is_empty()
            && self.active_jobs.is_empty();

        all_complete
    }

    // ==================== Info & Metrics (Python API) ====================

    /// Get step info (generic for all environments)
    pub fn get_step_info(&self, py: Python) -> PyResult<PyObject> {
        let info = PyDict::new(py);
        info.set_item("queue_length", self.job_queue.len())?;

        // Count total jobs in buckets
        let bucket_jobs_count: usize = self.job_buckets.iter().map(|b| b.jobs.len()).sum();
        info.set_item("bucket_jobs_count", bucket_jobs_count)?;
        info.set_item("num_buckets", self.job_buckets.len())?;

        info.set_item("active_jobs", self.active_jobs.len())?;
        info.set_item("needs_decision", self.get_current_job().is_some())?;
        info.set_item("total_jobs_generated", self.total_jobs_generated)?;
        info.set_item("total_jobs_completed", self.total_jobs_completed)?;
        info.set_item("current_time", self.current_time)?;
        info.set_item("current_step", self.current_step)?;
        info.set_item("total_jobs_deferred", self.total_jobs_deferred)?;

        if let Some(makespan_time) = self.makespan {
            info.set_item("makespan", makespan_time)?;
        }

        Ok(info.into())
    }

    /// Get basic metrics
    pub fn get_metrics(&self, py: Python) -> PyResult<PyObject> {
        let episode_duration = self.current_time.max(1) as f64;
        let avg_core_util = (self.core_util_sum / episode_duration) as f32;
        let avg_memory_util = (self.memory_util_sum / episode_duration) as f32;

        let defer_rate = if self.total_jobs_generated > 0 {
            self.total_jobs_deferred as f64 / self.total_jobs_generated as f64
        } else {
            0.0
        };

        let metrics = PyDict::new(py);

        // Calculate average waiting time
        let mut total_current_waiting_time = 0.0;

        // Jobs in main queue (not yet bucketed)
        for job in &self.job_queue {
            let waiting_time = self.current_time as f64 - job.submission_time;
            total_current_waiting_time += waiting_time;
        }

        // Add waiting time for jobs in buckets (includes all pending/deferred jobs)
        for bucket in &self.job_buckets {
            for job in &bucket.jobs {
                let waiting_time = self.current_time as f64 - job.submission_time;
                total_current_waiting_time += waiting_time;
            }
        }

        let total_episode_waiting_time = self.total_waiting_time_all_jobs + total_current_waiting_time;
        let avg_waiting_time = if self.total_jobs_generated > 0 {
            total_episode_waiting_time / self.total_jobs_generated as f64
        } else {
            0.0
        };

        metrics.set_item("total_jobs_completed", self.total_jobs_completed)?;
        metrics.set_item("avg_waiting_time", avg_waiting_time)?;
        metrics.set_item("defer_rate", defer_rate)?;

        if let Some(makespan_time) = self.makespan {
            metrics.set_item("makespan", makespan_time as f64)?;
        } else {
            metrics.set_item("makespan", py.None())?;
        }

        metrics.set_item("avg_host_core_utilization", avg_core_util)?;
        metrics.set_item("avg_host_memory_utilization", avg_memory_util)?;

        Ok(metrics.into())
    }

    /// Get host configurations
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

    /// Get job schedule information
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

    /// Get cluster resource information
    pub fn get_cluster_info(&self, py: Python) -> PyResult<PyObject> {
        let info_dict = pyo3::types::PyDict::new(py);
        info_dict.set_item("total_cluster_cores", self.total_cluster_cores)?;
        info_dict.set_item("total_cluster_memory", self.total_cluster_memory)?;
        info_dict.set_item("num_hosts", self.num_hosts)?;
        info_dict.set_item("host_cores_range", self.host_cores_range)?;
        info_dict.set_item("host_memory_range", self.host_memory_range)?;
        Ok(info_dict.to_object(py))
    }

    /// Set random seed for the environment
    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.original_seed = seed;
        if let Some(s) = seed {
            self.rng = rand::SeedableRng::seed_from_u64(s);
        } else {
            self.rng = rand::SeedableRng::from_entropy();
        }
    }
}
use std::collections::{HashMap, VecDeque};
use crate::job::Job;

#[derive(Debug, Clone)]
pub struct Host {
    pub id: usize,
    pub total_cores: u32,
    pub total_memory: u32,
    pub available_cores: u32,
    pub available_memory: u32,
    pub running_job_ids: HashMap<u32, Job>,
    
    // Pre-calculated normalized values for efficiency
    pub normalized_cores: f32,   // total_cores / env_max_cores
    pub normalized_memory: f32,  // total_memory / env_max_mem
    
    // Utilization history for per-second tracking (store up to 60 seconds)
    core_history: VecDeque<f32>, // Store past per-second core utilization values
    memory_history: VecDeque<f32>, // Store past per-second memory utilization values
}

impl Host {
    pub fn new(id: usize, total_cores: u32, total_memory: u32, env_max_cores: u32, env_max_mem: u32) -> Self {
        Host {
            id,
            total_cores,
            total_memory,
            available_cores: total_cores,
            available_memory: total_memory,
            running_job_ids: HashMap::new(),
            normalized_cores: total_cores as f32 / env_max_cores as f32,
            normalized_memory: total_memory as f32 / env_max_mem as f32,
            core_history: VecDeque::new(),
            memory_history: VecDeque::new(),
        }
    }
    
    pub fn can_accommodate(&self, job: &Job) -> bool {
        self.available_cores >= job.cores_required && 
        self.available_memory >= job.memory_required
    }
    
    pub fn allocate_job(&mut self, job: &mut Job) -> bool {
        if self.can_accommodate(job) {
            self.available_cores -= job.cores_required;
            self.available_memory -= job.memory_required;
            job.assigned_host = Some(self.id);
            self.running_job_ids.insert(job.id, job.clone());
            true
        } else {
            false
        }
    }
    
    
    pub fn release_job(&mut self, job: &Job) {
        if let Some(running_job) = self.running_job_ids.get(&job.id) {
            self.available_cores += running_job.cores_required;
            self.available_memory += running_job.memory_required;
            self.running_job_ids.remove(&job.id);
        }
    }
    
    pub fn get_core_utilization(&self) -> f32 {
        // Return per-second average, fallback to current if no history
        if self.core_history.is_empty() {
            self.get_current_core_utilization()
        } else {
            // Return the most recent per-second value
            *self.core_history.back().unwrap_or(&0.0)
        }
    }
    
    pub fn get_current_core_utilization(&self) -> f32 {
        if self.total_cores == 0 {
            0.0
        } else {
            1.0 - (self.available_cores as f32 / self.total_cores as f32)
        }
    }
    
    pub fn get_memory_utilization(&self) -> f32 {
        // Return per-second average, fallback to current if no history
        if self.memory_history.is_empty() {
            self.get_current_memory_utilization()
        } else {
            // Return the most recent per-second value
            *self.memory_history.back().unwrap_or(&0.0)
        }
    }
    
    pub fn get_current_memory_utilization(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            1.0 - (self.available_memory as f32 / self.total_memory as f32)
        }
    }
    
    pub fn update_utilization_history(&mut self) -> (f32, f32) {
        // Calculate current utilization values (optimized - no redundant calls)
        let current_core_util = if self.total_cores == 0 {
            0.0
        } else {
            1.0 - (self.available_cores as f32 / self.total_cores as f32)
        };
        
        let current_memory_util = if self.total_memory == 0 {
            0.0
        } else {
            1.0 - (self.available_memory as f32 / self.total_memory as f32)
        };
        
        // Add current utilization to history
        self.core_history.push_back(current_core_util);
        self.memory_history.push_back(current_memory_util);
        
        // Keep only last 60 values (60 seconds of history)
        if self.core_history.len() > 60 {
            self.core_history.pop_front();
            self.memory_history.pop_front();
        }
        
        // Return current values directly (no need to store in intermediate arrays)
        (current_core_util, current_memory_util)
    }
    
    pub fn update_normalized_values(&mut self, env_max_cores: u32, env_max_mem: u32) {
        self.normalized_cores = self.total_cores as f32 / env_max_cores as f32;
        self.normalized_memory = self.total_memory as f32 / env_max_mem as f32;
    }
    
    // old func for backup
    // pub fn update_core_history(&mut self, _current_time: f32) {
    //     self.update_utilization_history();
    // }
    
}
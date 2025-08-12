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
    
    // Utilization history for per-second tracking (store up to 60 seconds)
    core_history: VecDeque<f32>, // Store past per-second core utilization values
    memory_history: VecDeque<f32>, // Store past per-second memory utilization values
    last_history_update_time: f32,
}

impl Host {
    pub fn new(id: usize, total_cores: u32, total_memory: u32) -> Self {
        Host {
            id,
            total_cores,
            total_memory,
            available_cores: total_cores,
            available_memory: total_memory,
            running_job_ids: HashMap::new(),
            core_history: VecDeque::new(),
            memory_history: VecDeque::new(),
            last_history_update_time: 0.0,
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
        self.available_cores += job.cores_required;
        self.available_memory += job.memory_required;
        self.running_job_ids.remove(&job.id);
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
    
    pub fn update_utilization_history(&mut self, current_time: f32) {
        // Only update once per second
        let current_time_floor = current_time.floor();
        if current_time_floor > self.last_history_update_time {
            // Get current utilization values (real-time)
            let current_core_util = self.get_current_core_utilization();
            let current_memory_util = self.get_current_memory_utilization();
            
            // Add current utilization to history
            self.core_history.push_back(current_core_util);
            self.memory_history.push_back(current_memory_util);
            
            // Keep only last 60 values (60 seconds of history)
            while self.core_history.len() > 60 {
                self.core_history.pop_front();
            }
            while self.memory_history.len() > 60 {
                self.memory_history.pop_front();
            }
            
            self.last_history_update_time = current_time_floor;
        }
    }
    
    // Keep the old method name for backward compatibility, but make it call the new one
    pub fn update_core_history(&mut self, current_time: f32) {
        self.update_utilization_history(current_time);
    }
    
}
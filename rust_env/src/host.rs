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
    
    // Core utilization history for 15-second average calculation (store up to 60 seconds)
    core_history: VecDeque<f32>, // Store 60 past core utilization values
    last_core_update_time: f32,
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
            last_core_update_time: 0.0,
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
        if self.total_cores == 0 {
            0.0
        } else {
            1.0 - (self.available_cores as f32 / self.total_cores as f32)
        }
    }
    
    pub fn get_memory_utilization(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            1.0 - (self.available_memory as f32 / self.total_memory as f32)
        }
    }
    
    pub fn update_core_history(&mut self, current_time: f32) {
        let current_util = self.get_core_utilization();
        
        // Add current utilization to history
        self.core_history.push_back(current_util);
        
        // Keep only last 60 values (60 seconds of history)
        while self.core_history.len() > 60 {
            self.core_history.pop_front();
        }
        
        self.last_core_update_time = current_time;
    }
    
    pub fn get_core_utilization_15s_avg(&self) -> f32 {
        if self.core_history.is_empty() {
            return self.get_core_utilization();
        }
        
        // Take last 15 values for 15-second average
        let recent_count = self.core_history.len().min(15);
        let total_util: f32 = self.core_history.iter().rev().take(recent_count).sum();
        total_util / recent_count as f32
    }
}
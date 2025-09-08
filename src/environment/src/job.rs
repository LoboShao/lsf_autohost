#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: u32,
    pub cores_required: u32,
    pub memory_required: u32,
    pub duration: u32,
    pub submission_time: f64,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
    pub assigned_hosts: Vec<(usize, u32, u32)>, // (host_id, cores_allocated, memory_allocated)
    pub status: JobStatus,
}

impl Job {
    pub fn new(
        id: u32,
        cores_required: u32,
        memory_required: u32,
        duration: u32,
        submission_time: f64,
    ) -> Self {
        Job {
            id,
            cores_required,
            memory_required,
            duration,
            submission_time,
            start_time: None,
            end_time: None,
            assigned_hosts: Vec::new(),
            status: JobStatus::Pending,
        }
    }
    
    pub fn is_multihost(&self) -> bool {
        self.assigned_hosts.len() > 1
    }
    
    pub fn total_allocated_cores(&self) -> u32 {
        self.assigned_hosts.iter().map(|(_, cores, _)| cores).sum()
    }
    
    pub fn total_allocated_memory(&self) -> u32 {
        self.assigned_hosts.iter().map(|(_, _, memory)| memory).sum()
    }
}
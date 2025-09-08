// Internal tests that don't require Python runtime
#[cfg(test)]
mod tests {
    use crate::host::Host;
    use crate::job::Job;
    
    #[test]
    fn test_host_partial_allocation() {
        println!("\n{}", "=".repeat(60));
        println!("Test: Host Partial Allocation");
        println!("{}", "=".repeat(60));
        
        // Create a host with 2 cores and 4GB memory
        let mut host = Host::new(0, 2, 4096, 4, 8192);
        
        println!("\nHost created: 2 cores, 4GB memory");
        
        // Test partial allocation
        let job_id = 1;
        let cores_to_allocate = 1;
        let memory_to_allocate = 2048;
        
        let success = host.allocate_partial(job_id, cores_to_allocate, memory_to_allocate);
        
        assert!(success, "Partial allocation should succeed");
        assert_eq!(host.available_cores, 1, "Should have 1 core remaining");
        assert_eq!(host.available_memory, 2048, "Should have 2GB remaining");
        
        println!("✓ Partial allocation successful");
        println!("  Allocated: 1 core, 2GB memory");
        println!("  Remaining: {} cores, {}MB memory", host.available_cores, host.available_memory);
        
        // Test releasing partial resources
        host.release_partial(job_id, cores_to_allocate, memory_to_allocate);
        
        assert_eq!(host.available_cores, 2, "Should have all cores back");
        assert_eq!(host.available_memory, 4096, "Should have all memory back");
        
        println!("✓ Partial release successful");
        println!("  Resources fully restored");
    }
    
    #[test]
    fn test_job_multihost_tracking() {
        println!("\n{}", "=".repeat(60));
        println!("Test: Job Multi-Host Tracking (Memory per host)");
        println!("{}", "=".repeat(60));
        
        // Create a job that requires 3 cores and 3GB memory
        let mut job = Job::new(1, 3, 3072, 10, 0.0);
        
        println!("\nJob created: 3 cores, 3GB memory required");
        
        // Simulate multi-host allocation with new behavior
        // Each host allocates the FULL memory requirement
        job.assigned_hosts.push((0, 1, 3072));  // Host 0: 1 core, 3GB memory
        job.assigned_hosts.push((1, 1, 3072));  // Host 1: 1 core, 3GB memory
        job.assigned_hosts.push((2, 1, 3072));  // Host 2: 1 core, 3GB memory
        
        println!("Job assigned to multiple hosts (memory per host):");
        for (host_id, cores, memory) in &job.assigned_hosts {
            println!("  Host {}: {} cores, {}MB memory", host_id, cores, memory);
        }
        
        // Test helper methods
        assert!(job.is_multihost(), "Job should be marked as multi-host");
        assert_eq!(job.total_allocated_cores(), 3, "Total cores should be 3");
        assert_eq!(job.total_allocated_memory(), 9216, "Total memory should be 9216MB (3072 * 3)");
        
        // Verify all hosts allocate memory
        let hosts_with_memory = job.assigned_hosts.iter()
            .filter(|(_, _, mem)| *mem > 0)
            .count();
        assert_eq!(hosts_with_memory, 3, "All participating hosts should allocate memory");
        
        println!("\n✓ Multi-host tracking verified:");
        println!("  is_multihost: {}", job.is_multihost());
        println!("  total_allocated_cores: {}", job.total_allocated_cores());
        println!("  total_allocated_memory: {}MB", job.total_allocated_memory());
        println!("  hosts_allocating_memory: {}", hosts_with_memory);
    }
    
    #[test]
    fn test_multihost_resource_aggregation() {
        println!("\n{}", "=".repeat(60));
        println!("Test: Multi-Host Resource Aggregation (Memory per host)");
        println!("{}", "=".repeat(60));
        
        // Create 3 hosts with enough memory for the job
        let mut hosts = vec![
            Host::new(0, 1, 4096, 4, 8192),  // 1 core, 4GB
            Host::new(1, 1, 4096, 4, 8192),  // 1 core, 4GB
            Host::new(2, 2, 4096, 4, 8192),  // 2 cores, 4GB
        ];
        
        println!("\nHosts created:");
        for host in &hosts {
            println!("  Host {}: {} cores, {}MB memory", 
                     host.id, host.total_cores, host.total_memory);
        }
        
        // Calculate total available resources
        let total_cores: u32 = hosts.iter().map(|h| h.available_cores).sum();
        let max_memory_single_host: u32 = hosts.iter().map(|h| h.available_memory).max().unwrap();
        
        println!("\nCluster resources:");
        println!("  Total cores: {}", total_cores);
        println!("  Max memory on single host: {}MB", max_memory_single_host);
        
        assert_eq!(total_cores, 4, "Total cores should be 4");
        assert_eq!(max_memory_single_host, 4096, "Max memory on single host should be 4096MB");
        
        // Simulate allocating a job that needs 3 cores and 3GB
        // (Memory must fit on ONE host, cores can be distributed)
        let job_cores_needed = 3;
        let job_memory_needed = 3072;
        
        println!("\nJob requires: {} cores, {}MB memory", job_cores_needed, job_memory_needed);
        
        // Check if job can fit on any single host
        let can_fit_single = hosts.iter().any(|h| 
            h.available_cores >= job_cores_needed && 
            h.available_memory >= job_memory_needed
        );
        
        assert!(!can_fit_single, "Job should NOT fit on any single host");
        println!("✓ Confirmed: Job cannot fit on any single host");
        
        // Check if we can distribute cores with each host having enough memory
        // NEW behavior: each participating host must have enough memory
        let hosts_with_enough_memory = hosts.iter()
            .filter(|h| h.available_memory >= job_memory_needed)
            .count();
        
        // All hosts should have enough memory in this test
        assert_eq!(hosts_with_enough_memory, 3, "All hosts should have enough memory");
        println!("✓ Confirmed: {} host(s) have enough memory", hosts_with_enough_memory);
        
        // Simulate the NEW allocation logic (memory per host)
        let mut allocations = Vec::new();
        let mut cores_remaining = job_cores_needed;
        
        // Allocate cores from hosts that have enough memory
        for host in hosts.iter_mut() {
            if cores_remaining == 0 {
                break;
            }
            
            // Only use hosts that have enough memory
            if host.available_memory >= job_memory_needed && host.available_cores > 0 {
                let cores_to_allocate = cores_remaining.min(host.available_cores);
                
                // Each participating host allocates the FULL memory requirement
                host.allocate_partial(1, cores_to_allocate, job_memory_needed);
                allocations.push((host.id, cores_to_allocate, job_memory_needed));
                cores_remaining -= cores_to_allocate;
            }
        }
        
        println!("\nAllocation result (memory per host):");
        let total_memory_allocated: u32 = allocations.iter().map(|(_, _, mem)| mem).sum();
        for (host_id, cores, memory) in &allocations {
            println!("  Host {}: {} cores, {}MB memory", host_id, cores, memory);
        }
        println!("  Total memory allocated: {}MB", total_memory_allocated);
        
        assert_eq!(cores_remaining, 0, "All cores should be allocated");
        
        // Verify memory is allocated from ALL participating hosts (new behavior)
        let hosts_with_memory: Vec<_> = allocations.iter()
            .filter(|(_, _, mem)| *mem > 0)
            .collect();
        assert_eq!(hosts_with_memory.len(), allocations.len(), "All participating hosts should allocate memory");
        
        println!("\n✓ Successfully distributed job across {} hosts", allocations.len());
        println!("  Each host allocates full memory requirement ({}MB)", job_memory_needed);
    }
    
    #[test]
    fn test_single_vs_multihost_preference() {
        println!("\n{}", "=".repeat(60));
        println!("Test: Single-Host Preference Logic");
        println!("{}", "=".repeat(60));
        
        // Create hosts with varying capacities
        let hosts = vec![
            Host::new(0, 4, 8192, 8, 16384),  // Large host
            Host::new(1, 1, 2048, 8, 16384),  // Small host
            Host::new(2, 1, 2048, 8, 16384),  // Small host
        ];
        
        println!("\nHosts:");
        for host in &hosts {
            println!("  Host {}: {} cores, {}MB memory", 
                     host.id, host.total_cores, host.total_memory);
        }
        
        // Small job that fits on single host
        let small_job = Job::new(1, 2, 4096, 10, 0.0);
        println!("\nSmall job: {} cores, {}MB memory", 
                 small_job.cores_required, small_job.memory_required);
        
        // Check which hosts can accommodate it
        let suitable_hosts: Vec<usize> = hosts.iter()
            .filter(|h| h.can_accommodate(&small_job))
            .map(|h| h.id)
            .collect();
        
        assert!(!suitable_hosts.is_empty(), "At least one host should accommodate small job");
        println!("✓ Can fit on single host(s): {:?}", suitable_hosts);
        
        // Large job that requires multiple hosts
        let large_job = Job::new(2, 5, 10240, 10, 0.0);
        println!("\nLarge job: {} cores, {}MB memory", 
                 large_job.cores_required, large_job.memory_required);
        
        let suitable_single: Vec<usize> = hosts.iter()
            .filter(|h| h.can_accommodate(&large_job))
            .map(|h| h.id)
            .collect();
        
        assert!(suitable_single.is_empty(), "No single host should accommodate large job");
        println!("✓ Cannot fit on any single host");
        
        // Check if total resources are sufficient
        let total_cores: u32 = hosts.iter().map(|h| h.total_cores).sum();
        let total_memory: u32 = hosts.iter().map(|h| h.total_memory).sum();
        
        let can_fit_distributed = total_cores >= large_job.cores_required && 
                                 total_memory >= large_job.memory_required;
        
        assert!(can_fit_distributed, "Should fit when distributed");
        println!("✓ Can fit when distributed across multiple hosts");
    }
}
#!/usr/bin/env python3
"""
Debug test for multi-host scheduling to see what's happening.
"""

import numpy as np
from lsf_env_rust import ClusterSchedulerEnv

def debug_multihost():
    print("=" * 60)
    print("Debug Multi-Host Scheduling (CPU-only distribution)")
    print("=" * 60)
    
    # Create environment where multi-host CPU distribution is required
    env = ClusterSchedulerEnv(
        num_hosts=2,
        host_cores_range=(1, 1),       # Each host has only 1 core
        host_memory_range=(2048, 2048), # Each host has 2GB (enough for job memory)
        job_cores_range=(2, 2),         # Jobs need 2 cores (requires both hosts' CPUs!)
        job_memory_range=(1500, 1500),  # Jobs need 1.5GB (fits on one host's memory)
        job_duration_range=(5, 5),
        max_jobs_per_step=1,
        max_time=3,
        seed=42
    )
    
    print("\nEnvironment Configuration:")
    print("  2 hosts, each with 1 core and 2GB memory")
    print("  Jobs require 2 cores and 1.5GB memory")
    print("  => CPUs must be distributed, memory from one host\n")
    
    # Get host configs
    hosts = env.get_host_configs()
    print("Host Configurations:")
    for host in hosts:
        print(f"  Host {host['host_id']}: {host['total_cores']} cores, {host['total_memory']}MB")
    
    # Reset and check initial state
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"State values (first 10): {state[:10]}")
    
    # Check initial queue state
    queue_states = env.get_queue_states()
    print(f"\nInitial Queue State:")
    print(f"  Job queue: {queue_states['job_queue_length']}")
    print(f"  Submission queue: {queue_states['submission_queue_length']}")
    print(f"  Deferred jobs: {queue_states['deferred_jobs_length']}")
    print(f"  Total jobs generated: {queue_states['total_jobs_generated']}")
    
    # Take first step with equal priority
    print("\n" + "-" * 40)
    print("Taking first step with equal host priorities...")
    action = np.ones(2, dtype=np.float32)
    print(f"Action: {action}")
    
    state, reward, done, info = env.step(action)
    
    print(f"\nStep Result:")
    print(f"  Jobs scheduled: {info['jobs_scheduled']}")
    print(f"  Active jobs: {info['active_jobs']}")
    print(f"  Queue length: {info['queue_length']}")
    print(f"  Submission queue: {info['submission_queue_length']}")
    print(f"  Needs decision: {info['needs_decision']}")
    print(f"  Reward: {reward}")
    
    # Check queue state after step
    queue_states = env.get_queue_states()
    print(f"\nQueue State After Step:")
    print(f"  Job queue: {queue_states['job_queue_length']}")
    print(f"  Submission queue: {queue_states['submission_queue_length']}")
    print(f"  Deferred jobs: {queue_states['deferred_jobs_length']}")
    print(f"  Active jobs: {queue_states['active_jobs_count']}")
    
    if info['jobs_scheduled'] == 0:
        print("\n⚠️  Job was not scheduled!")
        print("Possible reasons:")
        print("  1. Job was deferred (waiting for resources)")
        print("  2. Multi-host scheduling logic not triggered")
        print("  3. Bug in the implementation")
        
        if queue_states['deferred_jobs_length'] > 0:
            print("\n❗ Job was DEFERRED instead of using multi-host scheduling")
            print("   This suggests the multi-host logic is not being triggered correctly")
    else:
        print("\n✓ Job was scheduled successfully!")
    
    # Continue for a few more steps to see what happens
    print("\n" + "-" * 40)
    print("Taking additional steps to observe behavior...")
    
    for i in range(5):
        if done:
            print(f"Episode ended at step {i+1}")
            break
            
        state, reward, done, info = env.step(action)
        print(f"\nStep {i+2}:")
        print(f"  Jobs scheduled: {info['jobs_scheduled']}")
        print(f"  Active jobs: {info['active_jobs']}")
        
        queue_states = env.get_queue_states()
        print(f"  Deferred: {queue_states['deferred_jobs_length']}")
        print(f"  Current time: {queue_states['current_time']}")
        
        if info['jobs_scheduled'] > 0:
            print("  ✓ Job scheduled!")
    
    # Final metrics
    metrics = env.get_metrics()
    print("\n" + "-" * 40)
    print("Final Metrics:")
    print(f"  Jobs completed: {metrics['total_jobs_completed']}")
    print(f"  Jobs in deferred queue: {queue_states['deferred_jobs_length']}")
    
    print("\n" + "=" * 60)
    print("Debug Complete")
    print("=" * 60)

if __name__ == "__main__":
    debug_multihost()
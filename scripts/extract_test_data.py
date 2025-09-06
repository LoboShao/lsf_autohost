#!/usr/bin/env python3
"""
Extract the EXACT test data used during PPO testing.
This replicates the exact flow: train_env.create_test_env(seed) -> test_env.reset()
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.wrapper.gym_wrapper import make_lsf_env

def extract_exact_test_data(training_env_config, log_dir=None):
    """Extract the exact test data that would be used during PPO testing."""
    
    # Step 1: Create the training environment (this sets up the base config)
    train_env = make_lsf_env(num_envs=1, **training_env_config)
    
    # Step 2: Extract test data for each seed using the exact same flow as PPO test()
    test_seeds = [42, 43, 44]
    test_data = {
        "training_env_config": training_env_config,
        "test_environments": {}
    }
    
    for seed in test_seeds:
        print(f"Extracting test data for seed {seed}...")
        
        # EXACT same flow as ppo.py test() function:
        # test_env = train_env.create_test_env(seed)
        test_env = train_env.create_test_env(seed)
        
        # obs, _ = test_env.reset()  
        obs, _ = test_env.reset()  # This triggers the deterministic generation
        
        # Now extract the generated data
        rust_env = test_env.rust_env
        
        # Host data (after reset() with seed)
        hosts = []
        for i, host in enumerate(rust_env.hosts):
            hosts.append({
                "host_id": i,
                "total_cores": host.total_cores,
                "total_memory": host.total_memory
            })
        
        # Job schedule data (after reset() with seed)
        job_data = {
            "job_arrival_schedule": rust_env.job_arrival_schedule.copy(),
            "job_cores_schedule": rust_env.job_cores_schedule.copy(), 
            "job_memory_schedule": rust_env.job_memory_schedule.copy(),
            "job_duration_schedule": rust_env.job_duration_schedule.copy(),
            "total_jobs_in_pool": rust_env.total_jobs_in_pool,
            "max_time": rust_env.max_time,
            "max_jobs_per_step": rust_env.max_jobs_per_step
        }
        
        test_data["test_environments"][seed] = {
            "hosts": hosts,
            "job_schedule": job_data
        }
        
        test_env.close()
    
    train_env.close()
    
    # Save to file - use project root logs directory if not specified
    if log_dir is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        log_dir = os.path.join(project_root, "logs")
    
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, "exact_test_data.json")
    
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Exact test data saved to: {output_file}")
    return output_file

def load_test_data(file_path):
    """Load the saved test data."""
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Use the same config as typical training
    training_config = {
        "num_hosts": 30,
        "max_time": 150,  
        "max_jobs_per_step": 30,
        "host_cores_range": (16, 32),
        "host_memory_range": (64*1024, 128*1024),
        "job_cores_range": (1, 4), 
        "job_memory_range": (1*512, 4*1024),
        "job_duration_range": (5, 90)
    }
    
    extract_exact_test_data(training_config)
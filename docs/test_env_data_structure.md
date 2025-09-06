# Test Environment Data Structure

## Overview
The `test_env_data.json` file contains deterministic test environment configurations used for reproducible evaluation of trained PPO models. This data is automatically generated during the first test run of PPO training and saved to `logs/{experiment_name}/test_env_data.json`.

## File Structure

```json
{
  "test_environments": {
    "42": { ... },    // Test environment with seed 42
    "43": { ... },    // Test environment with seed 43  
    "44": { ... }     // Test environment with seed 44
  }
}
```

## Per-Environment Structure

Each test environment (seeds 42, 43, 44) contains:

### 1. Host Configurations (`hosts`)
Array of host objects, each containing:
- **`host_id`** (int): Unique identifier for the host (0, 1, 2, ...)
- **`total_cores`** (int): Number of CPU cores available on this host
- **`total_memory`** (int): Amount of memory available on this host (in MB)

Example:
```json
"hosts": [
  {"host_id": 0, "total_cores": 24, "total_memory": 98304},
  {"host_id": 1, "total_cores": 18, "total_memory": 81920},
  ...
]
```

### 2. Job Schedule (`job_schedule`)
Dictionary containing all job generation parameters:

#### Arrival Pattern
- **`job_arrival_schedule`** (array of int): Number of jobs arriving at each timestep
- **`max_time`** (int): Maximum simulation time (episode length)
- **`max_jobs_per_step`** (int): Maximum jobs that can arrive in one timestep

#### Job Requirements
- **`job_cores_schedule`** (array of int): CPU cores required for each job (in arrival order)
- **`job_memory_schedule`** (array of int): Memory required for each job in MB (in arrival order)
- **`job_duration_schedule`** (array of int): Runtime duration for each job in seconds (in arrival order)

#### Environment Configuration
- **`total_jobs_in_pool`** (int): Total number of jobs that will be generated
- **`num_hosts`** (int): Number of hosts in the cluster
- **`host_cores_range`** (tuple): Min/max cores per host
- **`host_memory_range`** (tuple): Min/max memory per host (MB)
- **`job_cores_range`** (tuple): Min/max cores per job
- **`job_memory_range`** (tuple): Min/max memory per job (MB)
- **`job_duration_range`** (tuple): Min/max job duration (seconds)

## Data Relationships

1. **Job Ordering**: Jobs are indexed consistently across all arrays:
   - `job_cores_schedule[0]` = cores for 1st job
   - `job_memory_schedule[0]` = memory for 1st job
   - `job_duration_schedule[0]` = duration for 1st job

2. **Arrival Timing**: 
   - `job_arrival_schedule[t]` = number of jobs arriving at timestep `t`
   - Jobs arrive in the order specified by the requirement arrays

3. **Host Assignment**: During testing, the PPO agent decides which host each job runs on based on the host configurations.

## Usage

This data ensures that:
- **Reproducible Testing**: Same job sequences and host configs for consistent evaluation
- **Fair Comparison**: All models tested on identical environments 
- **Deterministic Results**: Fixed seeds guarantee repeatable experiments

## Example Access Pattern

```python
import json

# Load test data
with open('logs/experiment/test_env_data.json', 'r') as f:
    data = json.load(f)

# Get host info for seed 42
hosts = data['test_environments']['42']['hosts']
print(f"Host 0: {hosts[0]['total_cores']} cores, {hosts[0]['total_memory']} MB")

# Get job schedule for seed 42
schedule = data['test_environments']['42']['job_schedule']
total_jobs = schedule['total_jobs_in_pool']
first_10_arrivals = schedule['job_arrival_schedule'][:10]
print(f"Total jobs: {total_jobs}")
print(f"Jobs arriving in first 10 seconds: {first_10_arrivals}")
```

## File Size
Typical file size: ~500KB - 2MB depending on:
- Number of hosts (default: 30)
- Episode length (default: 150 timesteps)  
- Jobs per timestep (default: up to 30)
- Total jobs (~4,000-5,000 per environment)
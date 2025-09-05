# LSF Cluster Scheduler Environment Documentation

## Overview

This environment simulates an IBM LSF (Load Sharing Facility) external scheduler plugin for semiconductor EDA (Electronic Design Automation) workloads. It models job scheduling decisions in a cluster with multiple hosts, considering only CPU cores and memory constraints as per LSF external plugin limitations.

## Key Design Principles

### 1. Time Model
- **Integer Time Steps**: Time advances in whole seconds (no fractional time)
- **Multiple Jobs per Second**: Multiple jobs can arrive within the same second
- **Scheduling Cycles**: One scheduling cycle = one second
- **Event-Driven Advancement**: Time advances only when all jobs for current second have been attempted

### 2. Job Queue Management
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   New Jobs      │ --> │   Job Queue      │ --> │ Submission Queue│
│ (arrive at t=T) │     │ (FIFO ordering)  │     │  (one at a time)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               ^                           |
                               |                           v
                        ┌──────────────────┐      ┌─────────────────┐
                        │  Deferred Jobs   │      │  Agent Decision │
                        │ (from t=T-1)     │      │  (host ranking) │
                        └──────────────────┘      └─────────────────┘
```

### 3. Deferred Job Handling
- Jobs that cannot be scheduled are moved to `deferred_jobs` queue
- At the start of each new second:
  1. New jobs for current second are added to `job_queue`
  2. Deferred jobs from previous second are appended to back of `job_queue`
- This matches LSF behavior where deferred jobs retry in next scheduling cycle

## Environment Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_hosts` | 1000 | Number of hosts in the cluster |
| `max_queue_length` | max_time × max_jobs_per_step | Maximum jobs that can be queued |
| `host_cores_range` | (32, 128) | Range of CPU cores per host |
| `host_memory_range` | (131072, 524288) | Range of memory per host in MB (128GB-512GB) |
| `job_cores_range` | (1, 32) | Range of CPU cores required per job |
| `job_memory_range` | (2048, 65536) | Range of memory required per job in MB (2GB-64GB) |
| `job_duration_range` | (1, 60) | Job duration range in seconds |
| `max_jobs_per_step` | 50 | Maximum jobs that can arrive per second |
| `max_time` | 4096 | Time after which no new jobs arrive |
| `seed` | None | Random seed for deterministic behavior |

### Memory Units
- **All memory values are in MB** - this is the basic unit throughout the environment
- Host memory example: 128GB = 131072 MB
- Job memory example: 4GB = 4096 MB
- Default ranges represent typical small test configurations

## State Space

**Size**: `num_hosts * 4 + 2`

**Format**: `[host_features..., job_features]`

### Per-Host Features (4 per host)
1. **Core Utilization** (0-1): Current CPU usage from last second
2. **Memory Utilization** (0-1): Current memory usage from last second  
3. **Normalized Cores**: `total_cores / max_cores_in_env`
4. **Normalized Memory**: `total_memory / max_memory_in_env`

### Job Features (2 total)
1. **Normalized Cores Required**: `job_cores / max_job_cores`
2. **Normalized Memory Required**: `job_memory / max_job_memory`

## Action Space

**Size**: `num_hosts`

**Format**: Priority values `[0, 1]` for each host

The agent provides a priority ranking for all hosts. The environment tries hosts in descending priority order until it finds one that can accommodate the job.

## Reward Function

### Makespan-Aware Reward
The reward function changes behavior based on the scheduling phase:

```python
if submission_queue.is_empty():  # Cleanup phase
    reward = -utilization_score   # Penalize high utilization
else:  # Normal phase
    reward = utilization_score    # Reward high utilization
```

**Rationale**: 
- During normal operation, high utilization means efficient resource usage
- During cleanup (no new jobs), high utilization means jobs are taking too long
- This prevents the agent from artificially extending job durations

### Utilization Score Components
- **Balance Score** (50%): Rewards balanced CPU/memory usage within hosts
- **Efficiency Score** (50%): Rewards effective utilization (min of CPU/memory)

## Episode Dynamics

### Episode Start
1. Hosts are initialized with realistic configurations
2. Deterministic job schedule is generated
3. First batch of jobs added to queue

### During Episode
1. Agent makes scheduling decisions one job at a time
2. Time advances when all jobs for current second are attempted
3. Host utilization tracked per second
4. Deferred jobs retry in next cycle

### Episode Termination
Episode ends when:
- `current_time >= max_time` (no new jobs arriving)
- AND all queues are empty (job_queue, submission_queue, deferred_jobs)
- AND no active jobs remain

### Makespan
- Recorded when episode ends (all jobs completed)
- Represents total time to process all jobs

## Metrics

### Performance Metrics
- **Completion Rate**: Jobs completed / jobs generated
- **Makespan**: Total time to complete all jobs
- **Average Waiting Time**: Mean time jobs spend in queues
- **Resource Utilization**: Average CPU and memory usage

### Balance Metrics
- **Host Imbalance STD**: Standard deviation of resource imbalance
- **Effective Utilization STD**: Uniformity of utilization across hosts

## Implementation Details

### Deterministic Job Generation
- Job arrivals, requirements, and durations are pre-generated at environment creation
- Ensures reproducible episodes when using seeds
- Total jobs = sum of arrivals across all timesteps

### Host Configuration
- Uses realistic EDA cluster configurations
- Common core counts: 8, 16, 24, 32, 48, 64, 96, 128
- Common memory sizes (in MB): 32768, 65536, 131072, 262144, 524288 (32GB-512GB)
- Filtered to specified ranges
- All values stored and compared in MB

### Job Memory Patterns
- Based on real EDA workloads (all values in MB):
  - Small (512-1024): Lint, quick synthesis
  - Medium (1024-2048): Block-level place & route
  - Large (2048-4096): Full-chip operations
  - Very Large (6144+): Complex simulations
- All memory comparisons and allocations use MB

## Usage Example

```python
from lsf_env_rust import ClusterSchedulerEnv

# Create environment
env = ClusterSchedulerEnv(
    num_hosts=50,
    max_jobs_per_step=40,
    host_cores_range=(16, 32),
    host_memory_range=(65536, 131072),  # 64-128 GB in MB
    job_cores_range=(1, 4),
    job_memory_range=(512, 4096),       # 512MB-4GB in MB
    seed=42  # For deterministic behavior
)

# Reset environment
state = env.reset()

# Step through environment
while True:
    if env.needs_decision():
        action = agent.predict(state)  # Host priorities
        state, reward, done, info = env.step(action)
        if done:
            break

# Get metrics
metrics = env.get_metrics()
```

## Design Rationale

### Why Integer Time?
- Avoids floating-point precision issues
- Matches real LSF scheduling cycle behavior
- Simplifies completion event handling

### Why Deferred Jobs to Back of Queue?
- Gives new arrivals first chance (fairness)
- Prevents starvation of new jobs
- Matches typical LSF scheduling policies

### Why Change Reward in Cleanup Phase?
- Prevents gaming where agent extends makespan
- Encourages quick job completion when no new work
- Results in more realistic scheduling behavior
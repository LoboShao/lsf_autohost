# LSF Reinforcement Learning Scheduler

A reinforcement learning-based job scheduler for IBM LSF environments using Proximal Policy Optimization (PPO). The trained models can be deployed via IBM LSF external plugins for real-time scheduling decisions.

## Project Overview

This system implements a deep reinforcement learning approach to optimize job scheduling in LSF clusters. The core simulation environment is built in Rust for performance, with Python-based PPO training and comprehensive evaluation tools. The resulting models are designed for integration with IBM LSF through external scheduling plugins.

## Architecture

### Core Components

```
src/
├── environment/          # Rust-based cluster simulation
│   ├── src/env.rs       # Main environment logic
│   ├── src/host.rs      # Host resource management
│   ├── src/job.rs       # Job lifecycle management
│   └── src/event.rs     # Event-driven simulation
├── training/             # PPO implementation
│   ├── lsf_train.py     # Main training script
│   ├── ppo.py           # PPO algorithm implementation
│   └── variable_host_model.py  # Neural network architecture
└── wrapper/              # Environment interfaces
    └── gym_wrapper.py   # Gymnasium compatibility
```

### Rust Environment Core

The simulation environment is implemented in Rust for high performance and exposes Python bindings via PyO3:

**Key Characteristics**:
- **Integer time model**: Time advances in whole seconds, multiple jobs can arrive per second
- **Event-driven simulation**: Job arrivals, completions, and resource updates processed efficiently
- **Memory consistency**: All memory values in MB throughout codebase (64GB = 65536MB)
- **Deterministic execution**: Pre-generated job schedules for reproducible evaluation

**Job Flow**:
```
New Jobs → Job Queue → Submission Queue → Agent Decision → Host Assignment
              ↑                                            ↓
         Deferred Jobs ← ← ← ← ← ← ← ← ← ← ← (if insufficient resources)
```

**State Representation** (size: `num_hosts * 4 + 2`):
- Per-host features: CPU utilization, memory utilization, normalized cores, normalized memory
- Global job features: normalized core/memory requirements for current job

**Action Interface**:
The agent receives host priority vectors as input and must select which host to assign the current job. Invalid assignments (insufficient resources) are automatically handled.

**Core Environment Methods**:
- `reset()`: Initialize cluster state and job schedules
- `step(action)`: Process scheduling decision and advance simulation
- `needs_decision()`: Check if agent input required
- `get_state()`: Return current normalized state vector
- `get_metrics()`: Extract performance statistics

## Installation

### Prerequisites
- Python 3.8+
- Rust 1.70+ with Cargo
- PyTorch 1.12+
- Gymnasium 0.26+

### Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Build Rust environment**:
   ```bash
   cd src/environment
   cargo build --release
   cd ../..
   ```

3. **Verify installation**:
   ```bash
   python src/training/lsf_train.py --help
   ```

## Training

### Basic Training Command

```bash
python src/training/lsf_train.py --log-dir experiment_name
```

### Environment Parameters

**Core Environment Settings**:
```bash
--num-hosts 10              # Number of compute hosts in cluster
--simulation-time 500       # Simulation time in seconds per episode
--max-jobs-per-step 30      # Maximum jobs arriving per timestep
```

**Resource Configuration**:
```bash
--host-cores-range 32 32    # Host CPU capacity
--host-memory-range 65536 65536    # Host memory in MB
--job-cores-range 1 8       # Job CPU requirements
--job-memory-range 4096 16384      # Job memory requirements (MB)
--job-duration-range 10 90  # Job duration in seconds
```

### PPO Hyperparameters

**Core PPO Settings**:
```bash
--lr 3e-4                   # Learning rate (CRITICAL PARAMETER)
--gamma 0.995               # Discount factor for future rewards
--lam 0.95                  # GAE lambda for advantage estimation
--clip-coef 0.1             # PPO clipping coefficient
--ent-coef 0.01             # Entropy coefficient for exploration
--vf-coef 0.5               # Value function loss coefficient
```

**Training Scale Parameters**:
```bash
--total-timesteps 33000000  # Total training steps
--rollout-steps 2048        # Steps per rollout collection
--num-envs 8                # Parallel environments
--minibatch-size 512        # Batch size for SGD updates
--update-epochs 4           # SGD epochs per rollout
```

### Advanced Training Features

**Learning Rate Scheduling**:
```bash
--lr-schedule cosine        # Options: constant, linear, exponential, cosine, warmup_cosine
--lr-warmup-steps 1000      # Gradual LR increase at start (~5-10% of total updates)
--lr-decay-factor 0.995     # For exponential decay only
```

**Training Control**:
```bash
--early-stopping-patience 200      # Updates to wait for improvement (10-20% of total)
--early-stopping-threshold 0.01    # Minimum improvement threshold
--save-freq 250                     # Save every N updates
--resume-from path/to/checkpoint.pt # Resume training
```

**Advanced Parameters**:
```bash
--value-norm-decay 0.99         # Value normalization decay
--exploration-noise-decay 0.998 # Policy exploration noise decay
```

### Hyperparameter Tuning Guide

#### 1. Monitor Key Metrics
- **Training reward**: Should increase over time
- **Policy loss**: Should decrease and stabilize
- **Value loss**: Should decrease
- **Entropy**: Should decrease gradually
- **Test completion rate**: Should improve

#### 2. Common Issues & Fixes

**Slow Learning:**
- Increase `--lr` to 5e-4
- Increase `--ent-coef` to 0.02
- Check if `--rollout-steps` captures enough rewards

**Unstable Training:**
- Decrease `--lr` to 1e-4
- Decrease `--clip-coef` to 0.05
- Increase `--minibatch-size`

**Poor Exploration:**
- Increase `--ent-coef`
- Slower `--exploration-noise-decay` to 0.999
- Check action space coverage

**Plateauing Performance:**
- Use learning rate schedule with decay (`--lr-schedule cosine`)
- Increase `--total-timesteps`
- Check if agent has converged to local optimum

#### 3. Parameter Relationships

**Learning Rate (`--lr`)**:
- Too high (>1e-3): Unstable training, loss spikes
- Too low (<1e-4): Slow learning, plateau early
- For batch rewards: Start with 3e-4 to 5e-4

**Rollout Steps (`--rollout-steps`)**:
- Must be long enough to capture multiple rewards
- Should be 2-5x your typical reward interval
- For batch reward system: 2048+ recommended

**Environment Scaling**:
- More hosts → larger action space → may need lower LR
- Higher `--max-jobs-per-step` → more scheduling pressure → harder learning
- Longer `--simulation-time` → more rewards per episode but slower rollouts

### Deterministic Testing Data

The system automatically generates deterministic test environment data for reproducible evaluation during training. This data is saved to `logs/{experiment_name}/test_env_data.json` and contains:

**Test Environment Seeds**: Fixed seeds (42, 43, 44) ensure consistent evaluation across experiments.

**Data Structure**:
```json
{
  "test_environments": {
    "42": {
      "hosts": [
        {"host_id": 0, "total_cores": 24, "total_memory": 98304},
        {"host_id": 1, "total_cores": 18, "total_memory": 81920}
      ],
      "job_schedule": {
        "job_arrival_schedule": [2, 1, 0, 3, ...],  // Jobs per SECOND (not timestep)
        "job_cores_schedule": [4, 2, 1, 8, ...],    // CPU requirements
        "job_memory_schedule": [8192, 4096, ...],   // Memory requirements (MB)
        "job_duration_schedule": [45, 30, 15, ...], // Job durations (seconds)
        "total_jobs_in_pool": 4500,
        "max_time": 150,
        "num_hosts": 30
      }
    }
  }
}
```

**Time vs Timestep Distinction**:
- **Time** (seconds): Simulation time that advances when all jobs in current second are processed
- **Timestep**: Individual agent decision steps - multiple timesteps can occur within one second
- **Job Arrivals**: `job_arrival_schedule[t]` = number of jobs arriving at simulation second `t`
- **Scheduling Process**: If 3 jobs arrive in second 5, this creates 3 timesteps for scheduling decisions before advancing to second 6

**Generation Process**:
1. **Automatic Creation**: Generated during first test run of PPO training
2. **Consistent Jobs**: All job requirements pre-determined for reproducibility
3. **Host Configurations**: Fixed cluster layouts for fair comparison
4. **Arrival Patterns**: Deterministic job arrival timing

**Usage**:
- Training automatically tests against these environments at intervals
- Ensures fair comparison between different training runs and hyperparameters

### Training Example

Large cluster training with advanced features:
```bash
python src/training/lsf_train.py \
  --num-hosts 100 \
  --simulation-time 300 \
  --max-jobs-per-step 80 \
  --total-timesteps 100000000 \
  --rollout-steps 16384 \
  --lr 1e-4 \
  --lr-schedule cosine \
  --early-stopping-patience 500 \
  --log-dir large_cluster_experiment
```

## Implementation Details

### Environment Implementation

**Resource Management**:
```python
# Host configuration with realistic EDA cluster values
host_cores_range = (16, 32)        # CPU cores per host
host_memory_range = (64*1024, 128*1024)  # Memory in MB (64GB-128GB)

# Job characteristics matching EDA workloads
job_cores_range = (1, 4)           # Typical EDA job core requirements
job_memory_range = (512, 4*1024)   # 512MB-4GB memory requirements
```

**Time and Job Management**:
- Jobs arrive in bursts up to `max_jobs_per_step` per second
- Agent makes scheduling decisions for each job in submission queue
- Deferred jobs (insufficient resources) retry in subsequent steps
- Episode ends when all generated jobs complete or time limit reached

**Performance Optimizations**:
- Pre-calculated host normalization factors (cores/max_cores, memory/max_memory)
- Cached state vector construction eliminates repeated calculations
- Efficient event-driven time advancement

### Key Environment Parameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `num_hosts` | 50-100 | Cluster size |
| `max_jobs_per_step` | 30-80 | Job arrival rate (scheduling pressure) |
| `max_time` | 150-300 | Simulation time in seconds |
| `host_cores_range` | (16, 32) | CPU diversity |
| `host_memory_range` | (64GB, 128GB) | Memory diversity |

### Training Algorithm Features

**PPO Enhancements**:
- Generalized Advantage Estimation (GAE) for variance reduction
- Value function normalization for training stability
- Learning rate scheduling (cosine annealing, warmup)
- Early stopping with patience-based convergence detection
- Exploration noise decay for improved exploitation over time



## Configuration Files

Training configurations can be saved and loaded:

```python
# Example configuration
config = {
    "num_hosts": 100,
    "simulation_time": 300,
    "rollout_steps": 16384,
    "learning_rate": 1e-4,
    # ... additional parameters
}
```

## Monitoring

### TensorBoard Integration

Monitor training progress:
```bash
tensorboard --logdir logs/
```

**Available metrics**:
- Training losses and gradients
- Environment performance metrics
- PPO vs baseline comparisons
- Learning rate schedules
- Exploration statistics



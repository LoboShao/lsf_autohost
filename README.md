# LSF Reinforcement Learning Scheduler

A reinforcement learning-based job scheduler for IBM LSF environments using Proximal Policy Optimization (PPO). The system provides multiple scheduling strategies through specialized sub-environments.

## Project Overview

This system implements a deep reinforcement learning approach to optimize job scheduling in LSF clusters. The core simulation is built in Rust for performance, with Python-based PPO training. The modular architecture supports multiple scheduling strategies through specialized sub-environments.

## Architecture

### Environment Structure

The project provides **two specialized sub-environments** built on a shared base environment:

1. **Host Sorting Environment** (`host_sorting_env`): Agent learns to prioritize hosts for each job
2. **Job Ordering Environment** (`job_ordering_env`): Agent learns to prioritize job buckets for scheduling

Additional environments can be created by extending the base environment implementation in `src/environment/`.

### Core Components

```
src/
├── environment/          # Rust-based simulation
│   └── src/
│       ├── base_env.rs           # Base environment (shared functionality)
│       ├── host_sorting_env.rs   # Host sorting environment
│       ├── job_ordering_env.rs   # Job ordering environment
│       ├── host.rs               # Host resource management
│       ├── job.rs                # Job lifecycle management
│       ├── event.rs              # Event-driven simulation
│       └── lib.rs                # Python bindings
├── training/             # PPO implementation
│   ├── lsf_train_host_sorting.py    # Host sorting training
│   ├── lsf_train_job_ordering.py    # Job ordering training
│   ├── ppo.py                        # PPO algorithm
│   ├── simple_mlp_policy.py          # Simple MLP policy
│   └── utils.py                      # Training utilities
└── wrapper/              # Gymnasium wrappers
    ├── gym_wrapper.py                # Host sorting wrapper
    └── job_ordering_wrapper.py       # Job ordering wrapper
```

## Sub-Environments

### 1. Host Sorting Environment

**Strategy**: RL agent decides **which host** to assign each job to by outputting host priorities.

**State Representation** (size: `num_hosts * 2 + 8`):
- Per-host features (2 per host):
  - Available cores (normalized, from previous cycle)
  - Available memory (normalized, from previous cycle)
- Job features (8 features):
  - Job cores/memory/duration (normalized)
  - Deferred flag, batch progress
  - Queue pressure, core/memory pressure

**Action Space**: Priority values for each host (continuous, 0-1)

**Training Script**: `lsf_train_host_sorting.py`

**Use Case**: Fine-grained control over host selection, suitable for heterogeneous clusters

### 2. Job Ordering Environment

**Strategy**: RL agent decides **which job bucket** to schedule first by outputting bucket priorities. Host selection uses first-available heuristic.

**State Representation** (size: `max_buckets * 2 + 2`):
- Per-bucket features (2 per bucket):
  - Bucket core requirement (normalized)
  - Job count in bucket
- Global features (2 features):
  - Available cores ratio
  - Available memory ratio

**Action Space**: Priority values for each bucket (continuous, 0-1)

**Model**: Simple 2-layer MLP (512→512) - input is straightforward

**Training Script**: `lsf_train_job_ordering.py`

**Use Case**: High-level job prioritization, faster decision-making, suitable for large-scale clusters

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
   python src/training/lsf_train_host_sorting.py --help
   python src/training/lsf_train_job_ordering.py --help
   ```

## Training

### Host Sorting Training

Train agent to select optimal hosts for each job:

```bash
python src/training/lsf_train_host_sorting.py \
  --num-hosts 30 \
  --max-time 500 \
  --max-jobs-per-step 50 \
  --log-dir host_sorting_exp1
```

**Key Parameters**:
```bash
--num-hosts 30              # Number of compute hosts
--max-time 500              # Simulation time in seconds
--max-jobs-per-step 50      # Jobs arriving per second
--host-cores-range 32 128   # Host CPU capacity
--host-memory-range 131072 524288  # Host memory (MB)
--job-cores-range 1 32      # Job CPU requirements
--job-memory-range 2048 65536      # Job memory (MB)
```

### Job Ordering Training

Train agent to prioritize job buckets:

```bash
python src/training/lsf_train_job_ordering.py \
  --num-hosts 30 \
  --max-buckets 100 \
  --simulation-time 500 \
  --max-jobs-per-step 3 \
  --log-dir job_ordering_exp1
```

**Key Parameters**:
```bash
--num-hosts 30              # Number of compute hosts
--max-buckets 100           # Maximum job buckets
--simulation-time 500       # Simulation time in seconds
--max-jobs-per-step 3       # Jobs arriving per timestep
--job-cores-range 1 8       # Job CPU requirements
--job-memory-range 1024 8192       # Job memory (MB)
```

### PPO Hyperparameters (Both Environments)

**Core PPO Settings**:
```bash
--lr 3e-4                   # Learning rate
--gamma 0.99                # Discount factor
--lam 0.98                  # GAE lambda
--clip-coef 0.2             # PPO clipping coefficient
--ent-coef 0.01             # Entropy coefficient
--vf-coef 0.5               # Value function loss coefficient
```

**Training Scale**:
```bash
--total-timesteps 67108864  # Total training steps
--rollout-steps 2048        # Steps per rollout
--num-envs 4                # Parallel environments
--minibatch-size 512        # Batch size for updates
--update-epochs 2           # SGD epochs per rollout
```

### Advanced Features

**Learning Rate Scheduling**:
```bash
--lr-schedule linear        # Options: constant, linear, exponential, cosine
--lr-warmup-steps 200       # Warmup steps
--use-kl-adaptive-lr        # KL-divergence adaptive LR
```

**Training Control**:
```bash
--early-stopping-patience 50       # Updates to wait
--save-freq 250                    # Save every N updates
--resume-from path/to/checkpoint.pt  # Resume training
```

## Monitoring

### TensorBoard

Monitor training progress:
```bash
tensorboard --logdir logs/
```

**Available metrics**:
- Training losses (policy, value, entropy)
- Environment performance (completion rate, makespan, utilization)
- PPO vs baseline comparisons
- Learning rate schedules

## Implementation Details

### Rust Environment Core

**Key Characteristics**:
- **Integer time model**: Time advances in whole seconds
- **Event-driven simulation**: Efficient job processing
- **LSF-style updates**: Resource utilization updates per scheduling cycle (not real-time)
- **Deterministic execution**: Pre-generated job schedules for reproducible evaluation

**State Updates**:
- Host availability uses **historical utilization** from previous cycle
- Simulates real LSF behavior where resource info has periodic updates
- See `env.rs:get_state()` - uses `get_core_utilization()` from history

### Job Bucketing

Jobs are grouped into buckets based on `generate_bucket_key()` in `env.rs`:

**Current strategy**:
```rust
format!("c_{}_m_{}_d_{}", cores, memory, duration)
```

**Alternative strategies**:
```rust
// Unique bucket per job
format!("job_{}", job_id)

// Group by resources only
format!("c_{}_m_{}", cores, memory)
```

## Deterministic Testing

The system automatically generates deterministic test data for reproducible evaluation:

**Test Seeds**: Fixed seeds (42, 43, 44) ensure consistent evaluation

**Data Storage**: `logs/{experiment_name}/test_env_data.json`

**Structure**:
```json
{
  "test_environments": {
    "42": {
      "hosts": [...],
      "job_schedule": {
        "job_arrival_schedule": [...],
        "job_cores_schedule": [...],
        "job_memory_schedule": [...],
        "job_duration_schedule": [...]
      }
    }
  }
}
```

## Model Deployment

Trained models can be deployed as REST API services for LSF integration. See `src/model_deploy/` for deployment scripts.

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
    ├── gym_wrapper.py   # Gymnasium compatibility
    └── vectorized_gym_wrapper.py  # Multi-environment support
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

### Key Training Parameters

**Environment Configuration**:
```bash
--num-hosts 50              # Cluster size
--episode-length 200        # Timesteps per episode
--max-jobs-per-step 50      # Job arrival rate
--host-cores-range 16 32    # CPU cores per host
--host-memory-range 65536 131072  # Memory (MB) per host
--job-cores-range 1 4       # Job CPU requirements
--job-memory-range 512 4096 # Job memory requirements (MB)
```

**PPO Hyperparameters**:
```bash
--total-timesteps 50000000  # Total training steps
--rollout-steps 12288       # Steps per policy update
--lr 3e-4                   # Learning rate
--gamma 0.995               # Discount factor
--lam 0.99                  # GAE lambda
--clip-coef 0.2             # PPO clip coefficient
--ent-coef 0.01             # Entropy coefficient
--vf-coef 2.0               # Value function coefficient
```

### Advanced Training Features

**Learning Rate Scheduling**:
```bash
--lr-schedule cosine        # Options: constant, linear, exponential, cosine, warmup_cosine
--lr-warmup-steps 100       # Warmup steps for warmup schedules
```

**Early Stopping**:
```bash
--early-stopping-patience 200   # Updates to wait for improvement
--early-stopping-threshold 0.01 # Minimum improvement threshold
```

**Checkpointing**:
```bash
--save-freq 250             # Save every N updates
--resume-from path/to/checkpoint.pt  # Resume training
```

### Training Example

Large cluster training with advanced features:
```bash
python src/training/lsf_train.py \
  --num-hosts 100 \
  --episode-length 300 \
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
| `max_time` | 150-300 | Episode length in seconds |
| `host_cores_range` | (16, 32) | CPU diversity |
| `host_memory_range` | (64GB, 128GB) | Memory diversity |

### Training Algorithm Features

**PPO Enhancements**:
- Generalized Advantage Estimation (GAE) for variance reduction
- Value function normalization for training stability
- Learning rate scheduling (cosine annealing, warmup)
- Early stopping with patience-based convergence detection
- Exploration noise decay for improved exploitation over time

## Evaluation and Testing

### Deterministic Testing

The system includes comprehensive evaluation tools:

```bash
# Extract deterministic test environments
python scripts/extract_test_data.py

# Test trained model against baselines
# (automatically done during training at specified intervals)
```

**Test Seeds**: Fixed seeds (42, 43, 44) ensure reproducible evaluation across experiments.

**Baseline Comparison**: First-Available scheduler provides performance baseline.

### Metrics

**Environment Metrics**:
- Completion rate: Jobs successfully scheduled and completed
- Makespan: Total episode completion time
- Resource utilization: Average CPU/memory usage
- Queue wait time: Average job queueing delay
- Load imbalance: Variance in host utilization

**Training Metrics**:
- Policy loss: PPO clipped objective
- Value loss: Critic MSE
- Entropy: Policy exploration measure
- Gradient norm: Training stability indicator

## Model Deployment

### Checkpoint Format

Trained models are saved as PyTorch state dictionaries containing:
- Policy network parameters
- Value network parameters
- Normalization statistics
- Training metadata

### LSF Integration

The trained models are designed for deployment through IBM LSF external scheduling plugins:

1. **Model Loading**: Load trained PyTorch model in plugin environment
2. **State Extraction**: Convert LSF cluster state to model input format
3. **Inference**: Run forward pass to get host selection probabilities
4. **Action Selection**: Choose host based on model output
5. **Job Submission**: Submit job to selected host via LSF APIs

### Inference Pipeline

```python
# Pseudocode for LSF plugin integration
model = VariableHostPolicy.load(checkpoint_path)
model.eval()

# Convert LSF state to model input
state = extract_lsf_state(lsf_cluster_info)
normalized_state = normalize_state(state)

# Get scheduling decision
with torch.no_grad():
    action_probs = model.get_action_probs(normalized_state)
    selected_host = choose_action(action_probs, available_hosts)

# Submit job to selected host
lsf_submit_job(job, selected_host)
```

## Performance Characteristics

### Training Performance
- **Small clusters** (20-50 hosts): 2-4 hours on modern CPU
- **Large clusters** (100+ hosts): 8-16 hours, GPU acceleration recommended
- **Memory usage**: ~2-8GB depending on cluster size and rollout length

### Inference Performance
- **Latency**: <1ms per scheduling decision on modern hardware
- **Throughput**: 1000+ decisions/second for real-time scheduling
- **Memory footprint**: ~50-200MB depending on model size

## Configuration Files

Training configurations can be saved and loaded:

```python
# Example configuration
config = {
    "num_hosts": 100,
    "episode_length": 300,
    "rollout_steps": 16384,
    "learning_rate": 1e-4,
    # ... additional parameters
}
```

## Debugging and Monitoring

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

### Debug Scripts

```bash
# Detailed training analysis
python scripts/debug/debug_detailed.py

# PPO algorithm debugging
python scripts/debug/debug_ppo.py

# Architecture benchmarking
python scripts/debug/benchmark_hidden_sizes.py
```

## Troubleshooting

### Common Issues

**Rust compilation errors**:
- Update Rust: `rustup update`
- Install Python dev headers: `apt-get install python3-dev`

**Training instability**:
- Reduce learning rate: `--lr 1e-4`
- Increase rollout steps: `--rollout-steps 8192`
- Enable gradient clipping (default: enabled)

**Memory issues**:
- Reduce rollout steps or number of environments
- Use CPU training for small models

**Import errors**:
- Verify Python path includes project root
- Check that Rust library compiled successfully

### Performance Tuning

**For faster training**:
- Use larger minibatch sizes with GPU
- Enable multi-environment training
- Increase rollout length for better sample efficiency

**For memory efficiency**:
- Reduce rollout steps
- Use single environment training
- Enable gradient accumulation
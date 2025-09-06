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

### State Representation

The environment provides a normalized state vector optimized for neural network training:

**State Size**: `num_hosts * 4 + 2`

**Per-Host Features (4 per host)**:
- `host_core_util`: Current CPU utilization (0-1)
- `host_mem_util`: Current memory utilization (0-1)
- `host_cores_norm`: Host cores / environment max cores (normalized capacity)
- `host_mem_norm`: Host memory / environment max memory (normalized capacity)

**Global Job Features (2 total)**:
- `job_core_norm`: Current job cores / max job cores
- `job_mem_norm`: Current job memory / max job memory

**Performance Optimizations**:
- Host capacity ratios are pre-calculated and cached in the Host struct
- Eliminates 60 runtime divisions per state query, replaced with cached field accesses
- Critical for training performance at scale

### Action Space

The policy network outputs a single integer representing the target host ID for the current job. Invalid actions (insufficient resources) are masked during training and evaluation.

### Reward Function

The reward structure balances multiple scheduling objectives:
- Job completion rewards
- Resource utilization efficiency
- Queue wait time penalties
- Load balancing incentives

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

### Neural Network Architecture

The `VariableHostPolicy` implements a specialized architecture for heterogeneous cluster scheduling:

**Host Encoder**:
- Processes 4 features per host through a shared MLP
- Outputs host embeddings capturing resource state and capacity
- Handles variable numbers of hosts dynamically

**Job Encoder**:
- Processes global job features (2 features)
- Provides context about current scheduling decision

**Policy Head**:
- Combines host and job representations
- Outputs action probabilities over available hosts
- Includes exploration noise with decay schedule

**Value Head**:
- Estimates state values for advantage computation
- Shares representations with policy head

### Environment Simulation

The Rust environment implements a discrete-event simulation with:

**Job Generation**:
- Realistic EDA workload patterns
- Configurable resource requirements
- Deterministic schedules for reproducible evaluation

**Host Modeling**:
- Heterogeneous resource configurations
- Realistic core/memory combinations
- Efficient resource tracking

**Event Processing**:
- Job arrivals, completions, and resource updates
- Optimized for high-frequency state queries during training

### Training Algorithm

PPO implementation with modern enhancements:

**Generalized Advantage Estimation (GAE)**:
- Reduces variance in advantage estimates
- Configurable lambda parameter for bias-variance tradeoff

**Value Function Normalization**:
- Running mean/variance normalization
- Improves training stability with sparse rewards

**Gradient Clipping**:
- Prevents training instability
- Automatic scaling based on gradient norms

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
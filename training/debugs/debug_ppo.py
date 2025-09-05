import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wrapper.gym_wrapper import make_lsf_env
from variable_host_model import VariableHostPolicy
from ppo import PPOTrainer

# Create environment
env = make_lsf_env(num_hosts=50, max_jobs_per_step=40)

# Create policy
policy = VariableHostPolicy(
    obs_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    num_hosts=50
)

# Create trainer
trainer = PPOTrainer(
    policy=policy,
    env=env,
    buffer_size=256,  # Small buffer for quick test
    minibatch_size=64,
    update_epochs=2,
    tensorboard_log_dir="logs/debug_test",
    device="cpu"  # Comment out to use MPS
)

print("Starting PPO training test...")
print(f"Device: {trainer.device}")
print()

# Train for just a few updates to see FPS
trainer.train(
    total_timesteps=2048,  # Just 2 updates worth
    rollout_steps=1024,
    log_interval=1,
    test_interval=1000  # Don't run tests
)
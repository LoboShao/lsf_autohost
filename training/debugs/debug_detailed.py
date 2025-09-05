import time
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper.gym_wrapper import LsfEnvWrapper
from training.variable_host_model import VariableHostPolicy

# Check device
print("=== Device Info ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")
print()

# Test environment
print("=== Environment Speed Test ===")
env = LsfEnvWrapper(num_hosts=50, max_jobs_per_step=40)

# Time reset
start = time.time()
obs, _ = env.reset()
reset_time = time.time() - start
print(f"Reset time: {reset_time:.3f}s")
print(f"State size: {obs.shape}")

# Time raw env steps
start = time.time()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, _ = env.reset()
step_time = time.time() - start
print(f"100 env steps: {step_time:.3f}s ({100/step_time:.1f} steps/sec)")
print()

# Test tensor operations
print("=== Tensor Operation Speed Test ===")
obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

# Time numpy to tensor conversion
start = time.time()
for _ in range(1000):
    _ = torch.tensor(obs, dtype=torch.float32, device=device)
tensor_time = time.time() - start
print(f"1000 numpy->tensor conversions: {tensor_time:.3f}s")

# Time tensor operations on device
start = time.time()
for _ in range(1000):
    _ = torch.matmul(obs_tensor, torch.randn(obs_tensor.shape[0], 100, device=device))
matmul_time = time.time() - start
print(f"1000 tensor matmuls on {device}: {matmul_time:.3f}s")
print()

# Test policy if available
print("=== Policy Speed Test ===")
try:
    # Create small policy
    policy = VariableHostPolicy(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        num_hosts=50
    ).to(device)
    
    # Time policy forward pass
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action, _, _, _ = policy.get_action_and_value(obs_tensor)
    policy_time = time.time() - start
    print(f"100 policy forward passes: {policy_time:.3f}s ({100/policy_time:.1f} passes/sec)")
    
    # Time with CPU conversion
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            action, _, _, _ = policy.get_action_and_value(obs_tensor)
            _ = action.cpu().numpy()
    policy_cpu_time = time.time() - start
    print(f"100 policy passes with CPU conversion: {policy_cpu_time:.3f}s ({100/policy_cpu_time:.1f} passes/sec)")
    
except Exception as e:
    print(f"Could not test policy: {e}")

print("\nDone!")
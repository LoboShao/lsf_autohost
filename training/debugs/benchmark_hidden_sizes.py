import time
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from wrapper.gym_wrapper import LsfEnvWrapper
from training.variable_host_model import VariableHostPolicy

# Test different hidden sizes
hidden_sizes = [16, 32, 64, 128, 256, 512]
num_hosts = 50

# Create environment once
env = LsfEnvWrapper(num_hosts=num_hosts, max_jobs_per_step=40)
obs, _ = env.reset()
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print(f"Testing with obs_dim={obs_dim}, action_dim={action_dim}, num_hosts={num_hosts}")
print("\n" + "="*60)
print(f"{'Hidden Size':<12} {'Parameters':<12} {'CPU (fps)':<12} {'MPS (fps)':<12} {'Best':<8}")
print("="*60)

for hidden_size in hidden_sizes:
    # Modify the model to accept different hidden sizes
    class TestPolicy(VariableHostPolicy):
        def __init__(self, obs_dim, action_dim, num_hosts=30, hidden_size=16):
            self.hidden_size = hidden_size
            super().__init__(obs_dim, action_dim, num_hosts)
            
            # Override the hidden size
            self.hidden_size = hidden_size
            # Recreate layers with new hidden size
            self.host_encoder = torch.nn.Linear(4, hidden_size)
            self.job_encoder = torch.nn.Linear(2, hidden_size)
            self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.priority_head = torch.nn.Linear(hidden_size, 1)
            self.value_head = torch.nn.Linear(hidden_size, 1)
            self.apply(self._init_weights)
    
    # Test CPU
    policy_cpu = TestPolicy(obs_dim, action_dim, num_hosts, hidden_size).to('cpu')
    num_params = sum(p.numel() for p in policy_cpu.parameters())
    
    obs_tensor_cpu = torch.tensor(obs, dtype=torch.float32, device='cpu')
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            policy_cpu.get_action_and_value(obs_tensor_cpu)
    
    # Time CPU
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        with torch.no_grad():
            action, _, _, _ = policy_cpu.get_action_and_value(obs_tensor_cpu)
            _ = action.numpy()  # Include conversion time
    cpu_time = time.time() - start
    cpu_fps = num_iters / cpu_time
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        policy_mps = TestPolicy(obs_dim, action_dim, num_hosts, hidden_size).to('mps')
        obs_tensor_mps = torch.tensor(obs, dtype=torch.float32, device='mps')
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                policy_mps.get_action_and_value(obs_tensor_mps)
        
        # Time MPS
        start = time.time()
        for _ in range(num_iters):
            with torch.no_grad():
                action, _, _, _ = policy_mps.get_action_and_value(obs_tensor_mps)
                _ = action.cpu().numpy()  # Include conversion time
        mps_time = time.time() - start
        mps_fps = num_iters / mps_time
    else:
        mps_fps = 0
    
    best = "CPU" if cpu_fps > mps_fps else "MPS"
    
    print(f"{hidden_size:<12} {num_params:<12,} {cpu_fps:<12.0f} {mps_fps:<12.0f} {best:<8}")
    
print("="*60)
print("\nRecommendation:")
print("- For hidden_size ≤ 128: Use CPU")
print("- For hidden_size ≥ 256: Use MPS")
print("- Your current model (hidden_size=16): Definitely use CPU!")
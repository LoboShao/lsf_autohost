import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wrapper.gym_wrapper import LsfEnvWrapper

# Create env
env = LsfEnvWrapper(num_hosts=50, max_jobs_per_step=40)

# Time the reset
start = time.time()
obs, _ = env.reset()
reset_time = time.time() - start
print(f"Reset time: {reset_time:.3f}s")

# Time steps
start = time.time()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, _ = env.reset()
step_time = time.time() - start
print(f"100 steps time: {step_time:.3f}s")
print(f"Steps per second: {100/step_time:.1f}")
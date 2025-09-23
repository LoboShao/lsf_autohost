import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lsf_env_rust import ClusterSchedulerEnv
from typing import List, Any, Tuple


class LsfEnvWrapper(gym.Env):
    """Gymnasium wrapper for the Rust cluster scheduler environment."""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # Store original constructor kwargs for test environment creation
        self._constructor_kwargs = kwargs.copy()
        
        # Create the Rust environment
        self.rust_env = ClusterSchedulerEnv(**kwargs)
        
        # Store environment parameters
        self.num_hosts = kwargs.get('num_hosts', 1000)
        self.max_jobs_scheduled_per_step = kwargs.get('max_jobs_scheduled_per_step', 20)
        
        # Define action space: priority values for each host (0-1)
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.num_hosts,), 
            dtype=np.float32
        )
        
        # Define observation space - simplified to 2 features per host + 7 job features
        state_size = self.num_hosts * 2 + 7
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(state_size,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        obs = self.rust_env.reset()
        info = {}
        return obs, info
    
    def step(self, action):
        obs, reward, done, info = self.rust_env.step(action)
        terminated = done
        truncated = False
        
        # Auto-reset when episode is done (gymnasium standard behavior)
        if terminated:
            # Store final metrics before reset (important for testing)
            if hasattr(self.rust_env, 'get_metrics'):
                info['final_metrics'] = self.rust_env.get_metrics()
            obs, _ = self.reset()

        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        # Optional: implement visualization
        pass
    
    def close(self):
        # Cleanup if needed
        pass
    
    def get_metrics(self):
        return self.rust_env.get_metrics()
    
    def set_random_seed(self, seed=None):
        """Set the random seed for the underlying Rust environment.
        
        Args:
            seed: Random seed (int) for deterministic behavior, or None for random seeding
        """
        self.rust_env.set_random_seed(seed)
    
    def create_test_env(self, seed: int):
        """Create a fresh test environment with the same configuration but different seed.
        
        Args:
            seed: Seed for deterministic testing
            
        Returns:
            New LsfEnvWrapper instance with same config but specified seed
        """
        # Use the original constructor kwargs with updated seed
        kwargs = self._constructor_kwargs.copy()
        kwargs['seed'] = seed  # Override seed for deterministic testing
        
        return LsfEnvWrapper(**kwargs)


class VectorizedLSFEnv:
    """Simple vectorized environment for parallel LSF training."""
    
    def __init__(self, num_envs: int, **kwargs):
        self.num_envs = num_envs
        self.envs = [LsfEnvWrapper(**kwargs) for _ in range(num_envs)]
        
        # Get observation and action spaces from first environment
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
    
    def reset(self, seed=None, options=None):
        """Reset all environments."""
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = [None] * self.num_envs
            
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seeds[i], options=options)
            observations.append(obs)
            infos.append(info)
        
        return np.array(observations), infos
    
    def step(self, actions):
        """Step all environments."""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            infos
        )
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    def get_metrics(self):
        """Get aggregated metrics from all environments."""
        # Return metrics from the first environment as a representative sample
        if self.envs:
            return self.envs[0].get_metrics()
        return {}
    
    def set_random_seed(self, seed=None):
        """Set random seeds for all environments.
        
        Args:
            seed: Base random seed (int) for deterministic behavior, or None for random seeding.
                  If int, each env gets seed+i. If None, all envs use random seeding.
        """
        for i, env in enumerate(self.envs):
            if seed is not None:
                env.set_random_seed(seed + i)
            else:
                env.set_random_seed(None)


def make_lsf_env(num_envs: int = 1, **kwargs):
    """
    Create an LSF scheduler environment.
    
    Args:
        num_envs: Number of parallel environments (default: 1)
        **kwargs: Arguments passed to ClusterSchedulerEnv
    
    Returns:
        gym.Env: The environment instance (vectorized if num_envs > 1)
    """
    if num_envs == 1:
        return LsfEnvWrapper(**kwargs)
    else:
        return VectorizedLSFEnv(num_envs, **kwargs)


if __name__ == "__main__":
    print("Testing direct Rust env call:")
    rust_env = ClusterSchedulerEnv(num_hosts=10, episode_length=10)
    obs = rust_env.reset()
    print(f"Initial obs: {obs}")
    for i in range(5):
        action = np.random.random(10)
        obs, reward, done, info = rust_env.step(action)
        print(f"[RustEnv] Step {i+1}: reward={reward}")
        if done:
            break

    print("\nTesting Gym wrapper:")
    env = LsfEnvWrapper(num_hosts=10, episode_length=10)
    obs, info = env.reset()
    print(f"Initial obs: {obs}")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[Wrapper] Step {i+1}: reward={reward}")
        if terminated or truncated:
            break
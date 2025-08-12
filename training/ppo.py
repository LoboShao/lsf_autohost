import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, CosineAnnealingLR, SequentialLR
import numpy as np
from collections import defaultdict
import time
from typing import Dict, List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter
import json
import os


class PPOBuffer:
    def __init__(self, size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False

        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)

    def store(self, obs, action, reward, value, log_prob, done):
        self.obs[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.bool, device=self.device)

        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def compute_advantages(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        buffer_size = self.size if self.full else self.ptr

        rewards = self.rewards[:buffer_size].cpu().numpy()
        values = self.values[:buffer_size].cpu().numpy()
        dones = self.dones[:buffer_size].cpu().numpy()

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        returns = advantages + values

        adv_mean = advantages.mean()
        adv_std = advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.advantages[:buffer_size] = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        self.returns[:buffer_size] = torch.tensor(returns, dtype=torch.float32, device=self.device)

    def get_batch(self):
        buffer_size = self.size if self.full else self.ptr
        return {
            'obs': self.obs[:buffer_size],
            'actions': self.actions[:buffer_size],
            'log_probs': self.log_probs[:buffer_size],
            'advantages': self.advantages[:buffer_size],
            'returns': self.returns[:buffer_size],
            'values': self.values[:buffer_size]
        }

    def clear(self):
        self.ptr = 0
        self.full = False


class MetricsReporter:
    def __init__(self, writer: SummaryWriter, test_interval: int = 10):
        self.writer = writer
        self.test_interval = test_interval
        self.training_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

    def log_training_metrics(self, update_count: int, rollout_metrics: Dict, update_metrics: Dict, timesteps_collected: int, fps: float):
        avg_reward = np.mean(rollout_metrics['rewards'])
        avg_value = np.mean(rollout_metrics['values'])
        avg_entropy = np.mean(rollout_metrics['entropies'])
        avg_total_loss = np.mean(update_metrics['total_loss'])
        avg_policy_loss = np.mean(update_metrics['policy_loss'])
        avg_value_loss = np.mean(update_metrics['value_loss'])
        avg_entropy_loss = np.mean(update_metrics['entropy_loss'])
        avg_approx_kl = np.mean(update_metrics['approx_kl'])

        print(f"Update {update_count:4d} | Timesteps {timesteps_collected:7d} | FPS {fps:6.0f} | Reward {avg_reward:7.3f} | Loss {avg_total_loss:7.4f}")

        self.writer.add_scalar('Training/Total_Loss', avg_total_loss, update_count)
        self.writer.add_scalar('Training/Policy_Loss', avg_policy_loss, update_count)
        self.writer.add_scalar('Training/Value_Loss', avg_value_loss, update_count)
        self.writer.add_scalar('Training/Entropy_Loss', avg_entropy_loss, update_count)
        self.writer.add_scalar('Training/Approx_KL', avg_approx_kl, update_count)
        self.writer.add_scalar('Performance/Reward', avg_reward, update_count)
        self.writer.add_scalar('Performance/Value', avg_value, update_count)
        self.writer.add_scalar('Performance/Entropy', avg_entropy, update_count)
        self.writer.add_scalar('Performance/FPS', fps, update_count)

        self.training_metrics['update_count'].append(update_count)
        self.training_metrics['timesteps'].append(timesteps_collected)
        self.training_metrics['reward'].append(avg_reward)
        self.training_metrics['total_loss'].append(avg_total_loss)
        self.training_metrics['policy_loss'].append(avg_policy_loss)
        self.training_metrics['value_loss'].append(avg_value_loss)
        self.training_metrics['fps'].append(fps)

    def log_environment_metrics(self, env, update_count: int, prefix: str = 'Environment'):
        if hasattr(env, 'get_metrics'):
            env_metrics = env.get_metrics()
            if isinstance(env_metrics, dict):
                for key, value in env_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'{prefix}/{key}', value, update_count)

    def should_run_test(self, update_count: int) -> bool:
        return update_count % self.test_interval == 0

    def log_test_results(self, test_rewards: List[float], env_metrics: Dict, update_count: int):
        avg_test_reward = np.mean(test_rewards)
        std_test_reward = np.std(test_rewards)
        min_test_reward = np.min(test_rewards)
        max_test_reward = np.max(test_rewards)

        print(f"[TEST] Update {update_count:4d} | Avg: {avg_test_reward:.3f} ± {std_test_reward:.3f} | Range: [{min_test_reward:.3f}, {max_test_reward:.3f}]")

        self.writer.add_scalar('Test/Avg_Reward', avg_test_reward, update_count)
        self.writer.add_scalar('Test/Std_Reward', std_test_reward, update_count)
        self.writer.add_scalar('Test/Min_Reward', min_test_reward, update_count)
        self.writer.add_scalar('Test/Max_Reward', max_test_reward, update_count)

        for metric_name, values in env_metrics.items():
            if values:
                avg_value = np.mean(values)
                self.writer.add_scalar(f'Test/{metric_name}', avg_value, update_count)

        self.test_metrics['update_count'].append(update_count)
        self.test_metrics['avg_reward'].append(avg_test_reward)
        self.test_metrics['std_reward'].append(std_test_reward)

        return avg_test_reward


class FirstAvailableBaseline:
    """Baseline that always selects first available host (highest priority to host 0)"""
    def __init__(self, num_hosts):
        self.num_hosts = num_hosts
    
    def get_action_and_value(self, obs, deterministic=True):
        # Return decreasing priorities: host 0 gets highest priority (1.0), host 1 gets 0.9, etc.
        action = torch.linspace(1.0, 0.0, self.num_hosts)
        return action, None, None, None


class PPOTrainer:
    def test(self, num_episodes: int = 5, update_count: int = 0, 
             test_seeds: List[int] = None) -> float:
        """Run policy in evaluation mode with deterministic job sequences for full episodes.
        
        Args:
            num_episodes: Number of test episodes to run
            update_count: Current training update count (for logging)
            test_seeds: List of seeds for deterministic testing. If None, uses [42, 43, 44, 45, 46]
        """
        if test_seeds is None:
            test_seeds = [42, 43, 44, 45, 46]  # Default deterministic seeds
            
        self.policy.eval()
        rewards = []
        episode_steps = []
        env_metrics_accum = defaultdict(list)
        
        
        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            
            if self.is_vectorized:
                train_env = self.env.envs[0]
            else:
                train_env = self.env
                
            test_env = train_env.create_test_env(seed)
            
            obs, _ = test_env.reset()
            ep_reward = 0.0
            ep_steps = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    try:
                        action, _, _, _ = self.policy.get_action_and_value(obs_tensor, deterministic=True)
                    except TypeError:
                        action, _, _, _ = self.policy.get_action_and_value(obs_tensor)
                obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())
                ep_reward += reward
                ep_steps += 1
                
            rewards.append(ep_reward)
            episode_steps.append(ep_steps)
            
            if hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            env_metrics_accum[k].append(v)
        
        avg_test_reward = self.metrics_reporter.log_test_results(rewards, env_metrics_accum, update_count)

        if episode_steps:
            avg_steps = np.mean(episode_steps)
            total_steps = sum(episode_steps)
            print(f"[TEST] Timesteps: {avg_steps:.0f} avg ({total_steps} total) across {len(episode_steps)} episodes")
        # Check for early stopping
        if avg_test_reward > self.best_test_reward + self.early_stopping_threshold:
            self.best_test_reward = avg_test_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.early_stopping_patience:
            self.should_stop = True
            print(f"Early stopping triggered after {self.patience_counter} updates without improvement")
            
        self.policy.train()
        return avg_test_reward

    def test_with_metrics(self, num_episodes: int = 5, update_count: int = 0, 
                         test_seeds: List[int] = None, policy_name: str = "PPO") -> Dict:
        """Test policy and collect environment metrics"""
        if test_seeds is None:
            test_seeds = [42, 43, 44, 45, 46]
            
        self.policy.eval()
        episode_metrics = []
        
        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env
        
        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            test_env = train_env.create_test_env(seed)
            
            obs, _ = test_env.reset()
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action, _, _, _ = self.policy.get_action_and_value(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, _ = test_env.step(action.cpu().numpy())
                
            # Collect environment metrics
            if hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()
                if isinstance(metrics, dict):
                    episode_metrics.append(metrics)
            
            test_env.close()
        
        # Average metrics across episodes
        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}
        
        self.policy.train()
        return avg_metrics

    def test_baseline_with_metrics(self, num_episodes: int = 5, update_count: int = 0, 
                                  test_seeds: List[int] = None) -> Dict:
        """Test baseline policy and collect environment metrics"""
        if test_seeds is None:
            test_seeds = [42, 43, 44, 45, 46]
            
        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env
        
        num_hosts = train_env.num_hosts
        baseline = FirstAvailableBaseline(num_hosts)
        episode_metrics = []
        
        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            test_env = train_env.create_test_env(seed)
            
            obs, _ = test_env.reset()
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, _, _, _ = baseline.get_action_and_value(obs_tensor)
                obs, reward, terminated, truncated, _ = test_env.step(action.cpu().numpy())
                
            # Collect environment metrics
            if hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()
                if isinstance(metrics, dict):
                    episode_metrics.append(metrics)
            
            test_env.close()
        
        # Average metrics across episodes
        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}
        
        return avg_metrics

    def log_metric_comparison(self, ppo_metrics: Dict, baseline_metrics: Dict, update_count: int):
        """Log comparison between PPO and baseline performance metrics"""
        print(f"\n=== Performance Comparison (Update {update_count}) ===")
        
        # Define key metrics to compare
        key_metrics = [
            'total_jobs_completed',
            'completion_rate', 
            'avg_host_core_utilization',
            'avg_host_memory_utilization',
            'jobs_in_progress'
        ]
        
        for metric in key_metrics:
            if metric in ppo_metrics and metric in baseline_metrics:
                ppo_val = ppo_metrics[metric]
                baseline_val = baseline_metrics[metric]
                
                if baseline_val != 0:
                    improvement = ((ppo_val - baseline_val) / abs(baseline_val)) * 100
                    print(f"  {metric}: PPO {ppo_val:.3f} vs Baseline {baseline_val:.3f} ({improvement:+.1f}%)")
                else:
                    print(f"  {metric}: PPO {ppo_val:.3f} vs Baseline {baseline_val:.3f}")
                
                if self.writer:
                    self.writer.add_scalars(f'Metrics/{metric}', {
                        'PPO': ppo_val,
                        'Baseline': baseline_val
                    }, update_count)
                    
                    # Also log improvement percentage separately
                    if baseline_val != 0:
                        improvement = ((ppo_val - baseline_val) / abs(baseline_val)) * 100
                        self.writer.add_scalar(f'Improvement/{metric}_percent', improvement, update_count)
        
        print()

    def __init__(
        self,
        policy,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        buffer_size: int = 2048,
        device: str = "auto",
        clip_value_loss: bool = True,
        tensorboard_log_dir: str = "logs/ppo_training",
        lr_schedule: str = "constant",
        lr_decay_factor: float = 0.99,
        lr_warmup_steps: int = 0,
        early_stopping_patience: int = 50,
        early_stopping_threshold: float = 0.01,
        value_norm_decay: float = 0.99,
        checkpoint_dir: str = None,
        save_freq: int = 100
    ):
        self.policy = policy
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.clip_value_loss = clip_value_loss
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.policy.to(self.device)
        
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.lr_decay_factor = lr_decay_factor
        self.lr_warmup_steps = lr_warmup_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.value_norm_decay = value_norm_decay
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = self._create_lr_scheduler()
        
        # Early stopping
        self.best_test_reward = float('-inf')
        self.patience_counter = 0
        self.should_stop = False
        
        self.value_mean = 0.0
        self.value_var = 1.0
        
        # Checkpointing
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check if vectorized environment
        self.is_vectorized = hasattr(env, 'num_envs')
        
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # Buffer(s) - separate buffers for vectorized environments
        if self.is_vectorized:
            obs_dim = env.single_observation_space.shape[0]
            action_dim = env.single_action_space.shape[0]
            # Create separate buffer for each environment
            self.buffers = [PPOBuffer(buffer_size, obs_dim, action_dim, self.device) 
                           for _ in range(self.num_envs)]
        else:
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            self.buffer = PPOBuffer(buffer_size, obs_dim, action_dim, self.device)
        
        self.metrics = defaultdict(list)
        self.writer = SummaryWriter(tensorboard_log_dir)
        self.metrics_reporter = MetricsReporter(self.writer, test_interval=10)
        self.grad_norms = []
        self.lr_history = []
    
    def set_test_interval(self, interval: int):
        """Set the interval for running test episodes"""
        self.metrics_reporter.test_interval = interval
    
    def get_training_summary(self) -> Dict:
        """Get summary of training and test metrics"""
        return {
            'training_metrics': dict(self.metrics_reporter.training_metrics),
            'test_metrics': dict(self.metrics_reporter.test_metrics)
        }
    
    def collect_rollouts(self, num_steps: int) -> Dict:
        """Collect rollouts from the environment"""
        if self.is_vectorized:
            return self._collect_rollouts_vectorized(num_steps)
        else:
            return self._collect_rollouts_single(num_steps)
    
    def _collect_rollouts_single(self, num_steps: int) -> Dict:
        """Collect rollouts from single environment"""
        # Use random seed for training diversity
        self.env.set_random_seed(None)
        obs, _ = self.env.reset()
        rollout_metrics = defaultdict(list)
        episode_ended = False
        
        for step in range(num_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                action, log_prob, entropy, value = self.policy.get_action_and_value(obs_tensor)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.cpu().numpy())
            done = terminated or truncated
            
            
            # Store in buffer
            self.buffer.store(obs, action.cpu().numpy(), reward, value.cpu().item(), 
                            log_prob.cpu().item(), done)
            
            # Metrics
            rollout_metrics['rewards'].append(reward)
            rollout_metrics['values'].append(value.cpu().item())
            rollout_metrics['entropies'].append(entropy.cpu().item())
            
            obs = next_obs
            
            if done:
                if hasattr(self.env, 'get_metrics'):
                    env_metrics = self.env.get_metrics()
                    for key, val in env_metrics.items():
                        if isinstance(val, (int, float)):
                            rollout_metrics[f'env_{key}'].append(val)
                # Use random seed for next training episode
                self.env.set_random_seed(None)
                obs, _ = self.env.reset()
                episode_ended = True
            else:
                episode_ended = False
        
        # Compute advantages with proper bootstrapped value
        if episode_ended:
            # If last episode ended, use terminal value of 0
            last_value = 0.0
        else:
            # If episode is ongoing, use value of current state
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                last_value = self.policy.get_value(obs_tensor).cpu().item()
        
        self.buffer.compute_advantages(last_value, self.gamma, self.lam)
        return rollout_metrics
    
    def _collect_rollouts_vectorized(self, num_steps: int) -> Dict:
        """Collect rollouts from vectorized environments"""
        # Use random seeds for training diversity
        self.env.set_random_seed(None)
        obs, _ = self.env.reset()  # Shape: (num_envs, obs_dim)
        rollout_metrics = defaultdict(list)
        
        for step in range(num_steps):
            # Batch process all environments
            obs_batch = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                # Batch process all environments at once
                batch_actions, batch_log_probs, batch_entropies, batch_values = self.policy.get_action_and_value(obs_batch)
                
                # Convert to lists for env.step() and storage
                actions = batch_actions.cpu().numpy()
                log_probs = batch_log_probs.cpu().numpy()
                entropies = batch_entropies.cpu().numpy()
                values = batch_values.cpu().numpy()
            
            # Step all environments together
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            for env_idx in range(self.num_envs):
                env_obs = obs[env_idx]
                env_action = actions[env_idx]
                env_reward = rewards[env_idx]
                env_done = terminated[env_idx] or truncated[env_idx]
                
                # Store in environment-specific buffer
                self.buffers[env_idx].store(env_obs, env_action, env_reward, values[env_idx], 
                                          log_probs[env_idx], bool(env_done))
                
                rollout_metrics['rewards'].append(env_reward)
                rollout_metrics['values'].append(values[env_idx])
                rollout_metrics['entropies'].append(entropies[env_idx])
            
            obs = next_obs
        
        # Compute advantages separately for each environment
        obs_batch = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            bootstrap_values = self.policy.get_value(obs_batch).cpu().numpy()
        
        # Compute GAE for each environment buffer separately
        for env_idx in range(self.num_envs):
            # Each environment gets its own bootstrap value
            self.buffers[env_idx].compute_advantages(bootstrap_values[env_idx], self.gamma, self.lam)
        
        return rollout_metrics
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler based on configuration"""
        if self.lr_schedule == "constant":
            return None
        elif self.lr_schedule == "linear":
            estimated_updates = 1000
            return LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=estimated_updates)
        elif self.lr_schedule == "exponential":
            return ExponentialLR(self.optimizer, gamma=self.lr_decay_factor)
        elif self.lr_schedule == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=self.lr * 0.01)
        elif self.lr_schedule == "warmup_cosine":
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.lr_warmup_steps)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=self.lr * 0.01)
            return SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[self.lr_warmup_steps])
        else:
            raise ValueError(f"Unknown learning rate schedule: {self.lr_schedule}")
    
    def _update_value_normalization(self, values):
        """Update running statistics for value normalization"""
        batch_mean = values.mean().item()
        batch_var = values.var().item()
        
        # Update running mean and variance
        self.value_mean = self.value_norm_decay * self.value_mean + (1 - self.value_norm_decay) * batch_mean
        self.value_var = self.value_norm_decay * self.value_var + (1 - self.value_norm_decay) * batch_var
    
    def _normalize_values(self, values):
        """Normalize values using running statistics"""
        return (values - self.value_mean) / (torch.sqrt(torch.tensor(self.value_var)) + 1e-8)
    
    def save_checkpoint(self, update_count: int, metrics: Dict):
        """Save model checkpoint with training state"""
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'update_count': update_count,
            'best_test_reward': self.best_test_reward,
            'patience_counter': self.patience_counter,
            'value_mean': self.value_mean,
            'value_var': self.value_var,
            'metrics': metrics,
            'config': {
                'lr': self.lr,
                'lr_schedule': self.lr_schedule,
                'lr_decay_factor': self.lr_decay_factor,
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_coef': self.clip_coef,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{update_count}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_test_reward = checkpoint['best_test_reward']
        self.patience_counter = checkpoint['patience_counter']
        self.value_mean = checkpoint['value_mean']
        self.value_var = checkpoint['value_var']
        
        update_count = checkpoint['update_count']
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from update {update_count}")
        return update_count
    
    def update_policy(self) -> Dict:
        """Update policy using PPO"""
        update_metrics = defaultdict(list)
        
        for epoch in range(self.update_epochs):
            # Combine batches from all environment buffers
            if self.is_vectorized:
                # Collect data from all environment buffers
                all_batches = []
                for env_idx in range(self.num_envs):
                    env_batch = self.buffers[env_idx].get_batch()
                    all_batches.append(env_batch)
                
                # Concatenate all environment data
                batch = {}
                for key in all_batches[0].keys():
                    batch[key] = torch.cat([b[key] for b in all_batches], dim=0)
            else:
                batch = self.buffer.get_batch()
            
            # Split into minibatches
            batch_size = batch['obs'].shape[0]
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]
                
                mb_obs = batch['obs'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                mb_advantages = batch['advantages'][mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_old_values = batch['values'][mb_indices]
                
                # Get new policy outputs
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(mb_obs, mb_actions)
                
                # Policy loss
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.clip_value_loss:
                    value_clipped = mb_old_values + torch.clamp(
                        new_values - mb_old_values, -self.clip_coef, self.clip_coef
                    )
                    value_loss1 = (new_values - mb_returns).pow(2)
                    value_loss2 = (value_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                
                # Enhanced gradient clipping with monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.grad_norms.append(grad_norm.item())
                
                self.optimizer.step()
                
                # Metrics
                update_metrics['policy_loss'].append(policy_loss.item())
                update_metrics['value_loss'].append(value_loss.item())
                update_metrics['entropy_loss'].append(entropy_loss.item())
                update_metrics['total_loss'].append(loss.item())
                update_metrics['approx_kl'].append(((ratio - 1) - torch.log(ratio)).mean().item())
        
        return update_metrics
    
    def train(self, total_timesteps: int, rollout_steps: int = 2048, log_interval: int = 10, test_interval: Optional[int] = None):
        """Main training loop"""
        timesteps_collected = 0
        update_count = 0
        
        # Update scheduler with actual training parameters
        estimated_updates = total_timesteps // (rollout_steps * self.num_envs)
        if self.lr_schedule == "linear":
            self.scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=estimated_updates)
        elif self.lr_schedule == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=estimated_updates, eta_min=self.lr * 0.01)
        elif self.lr_schedule == "warmup_cosine":
            warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.lr_warmup_steps)
            cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=estimated_updates-self.lr_warmup_steps, eta_min=self.lr * 0.01)
            self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[self.lr_warmup_steps])
        
        # Set test interval if provided
        if test_interval is not None:
            self.set_test_interval(test_interval)
        
        print(f"Starting PPO training on {self.device}")
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"TensorBoard logs: {self.writer.log_dir}")
        print(f"Test episodes will run every {self.metrics_reporter.test_interval} updates")
        
        start_time = time.time()
        
        while timesteps_collected < total_timesteps:
            # Collect rollouts
            rollout_metrics = self.collect_rollouts(rollout_steps)
            timesteps_collected += rollout_steps * self.num_envs

            # Update policy with value normalization
            if self.is_vectorized:
                # Combine values from all environment buffers for normalization
                all_values = []
                for env_idx in range(self.num_envs):
                    env_batch = self.buffers[env_idx].get_batch()
                    all_values.append(env_batch['values'])
                combined_values = torch.cat(all_values, dim=0)
                self._update_value_normalization(combined_values)
            else:
                batch = self.buffer.get_batch()
                self._update_value_normalization(batch['values'])
            
            update_metrics = self.update_policy()
            update_count += 1

            # Step learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_history.append(current_lr)
                update_metrics['learning_rate'] = [current_lr]
            
            # Decay exploration noise
            if hasattr(self.policy, 'decay_exploration_noise'):
                self.policy.decay_exploration_noise()
                update_metrics['exploration_noise'] = [self.policy.current_exploration_noise]

            # Clear buffers
            if self.is_vectorized:
                for env_idx in range(self.num_envs):
                    self.buffers[env_idx].clear()
            else:
                self.buffer.clear()
            
            # Check for early stopping
            if self.should_stop:
                print("Early stopping triggered, ending training...")
                break

            # Log metrics
            if update_count % log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = timesteps_collected / elapsed_time
                
                self.metrics_reporter.log_training_metrics(
                    update_count, rollout_metrics, update_metrics, timesteps_collected, fps
                )

                if self.grad_norms:
                    avg_grad_norm = np.mean(self.grad_norms[-10:])  # Last 10 updates
                    self.writer.add_scalar('Training/Grad_Norm', avg_grad_norm, update_count)
                
                if self.lr_history:
                    self.writer.add_scalar('Training/Learning_Rate', self.lr_history[-1], update_count)
                
                if 'exploration_noise' in update_metrics:
                    self.writer.add_scalar('Training/Exploration_Noise', update_metrics['exploration_noise'][0], update_count)
                
                self.writer.add_scalar('Training/Value_Mean', self.value_mean, update_count)
                self.writer.add_scalar('Training/Value_Var', self.value_var, update_count)
                
                avg_reward = np.mean(rollout_metrics['rewards'])
                avg_value = np.mean(rollout_metrics['values'])
                avg_total_loss = np.mean(update_metrics['total_loss'])
                self.metrics['timesteps'].append(timesteps_collected)
                self.metrics['rewards'].append(avg_reward)
                self.metrics['values'].append(avg_value)
                self.metrics['total_loss'].append(avg_total_loss)
            
            # Save checkpoint at specified intervals
            if self.checkpoint_dir and update_count % self.save_freq == 0:
                current_metrics = {
                    'timesteps': timesteps_collected,
                    'avg_reward': np.mean(rollout_metrics['rewards']) if rollout_metrics['rewards'] else 0,
                    'avg_loss': np.mean(update_metrics['total_loss']) if update_metrics['total_loss'] else 0
                }
                self.save_checkpoint(update_count, current_metrics)
            
            # Run test episodes at specified intervals
            if self.metrics_reporter.should_run_test(update_count):
                ppo_metrics = self.test_with_metrics(num_episodes=5, update_count=update_count, policy_name="PPO")
                baseline_metrics = self.test_baseline_with_metrics(num_episodes=5, update_count=update_count)
                
                # Compare performance metrics
                self.log_metric_comparison(ppo_metrics, baseline_metrics, update_count)

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Total updates: {update_count}")
        print(f"Final FPS: {timesteps_collected / training_time:.0f}")
        
        # Log final summary
        if self.metrics_reporter.training_metrics['reward']:
            final_avg_reward = self.metrics_reporter.training_metrics['reward'][-1]
            print(f"Final training reward: {final_avg_reward:.3f}")
        if self.metrics_reporter.test_metrics['avg_reward']:
            final_test_reward = self.metrics_reporter.test_metrics['avg_reward'][-1]
            print(f"Final test reward: {final_test_reward:.3f}")
        
        self.writer.close()
        return self.get_training_summary()
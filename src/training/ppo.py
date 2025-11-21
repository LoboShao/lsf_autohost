import torch
import torch.optim as optim
import numpy as np
from collections import defaultdict
import time
from typing import Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
import json
import os

# Import utility classes
from .utils import PPOBuffer, MetricsReporter, FirstAvailableBaseline, LRSchedulerManager


class PPOTrainer:

    @torch.inference_mode()
    def test_with_metrics(self, num_episodes: int = 1, update_count: int = 0, 
                         test_seeds: List[int] = None, policy_name: str = "PPO") -> Dict:
        """Test policy and collect environment metrics"""
        if test_seeds is None:
            test_seeds = [42]
        
        # Save test environment data on first test run
        if self.first_test_run:
            self.save_test_env_data(test_seeds)
            self.first_test_run = False
            
        self.policy.eval()
        episode_metrics = []
        seed_to_metrics = {}  # Store per-seed metrics
        
        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env
        
        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            test_env = train_env.create_test_env(seed)
            
            # Get and print cluster information (only on first test)
            if hasattr(test_env, 'get_cluster_info') and update_count == 0:
                cluster_info = test_env.get_cluster_info()
                print(f"\n[TEST {policy_name}] Cluster Info for seed {seed}:")
                print(f"  Total Cluster Cores: {cluster_info.get('total_cluster_cores', 'N/A')}")
                print(f"  Total Cluster Memory: {cluster_info.get('total_cluster_memory', 'N/A')} MB")
                print(f"  Number of Hosts: {cluster_info.get('num_hosts', 'N/A')}")
                print(f"  Host Cores Range: {cluster_info.get('host_cores_range', 'N/A')}")
                print(f"  Host Memory Range: {cluster_info.get('host_memory_range', 'N/A')} MB")
            
            obs, _ = test_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            
            final_info = None
            while not (terminated or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, _, _, _ = self.policy.get_action_and_value(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())
                episode_return += reward
                episode_length += 1
                if terminated or truncated:
                    final_info = info
                
            # Collect environment metrics (use final_info if available, fallback to get_metrics)
            metrics = None
            if final_info and 'final_metrics' in final_info:
                metrics = final_info['final_metrics']
            elif hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()
                
            if isinstance(metrics, dict):
                # Add episode return to metrics
                metrics['episode_return'] = episode_return
                metrics['episode_length'] = episode_length
                episode_metrics.append(metrics)
                seed_to_metrics[seed] = metrics.copy()  # Store per-seed metrics
            
            test_env.close()
        
        # Average metrics across episodes
        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m and m.get(key) is not None]
                if values:
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}
        
        # Return both average and per-seed metrics
        result = {
            'average': avg_metrics,
            'per_seed': seed_to_metrics
        }
        
        self.policy.train()
        return result

    @torch.inference_mode()
    def test_baseline_with_metrics(self, num_episodes: int = 1, update_count: int = 0,
                                  test_seeds: List[int] = None) -> Dict:
        """Test baseline policy and collect environment metrics"""
        if test_seeds is None:
            test_seeds = [42]

        # Skip baseline testing if no baseline policy provided
        if self.baseline_policy is None:
            return {'average': {}, 'per_seed': {}}

        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env

        baseline = self.baseline_policy
        episode_metrics = []
        seed_to_metrics = {}  # Store per-seed metrics
        
        for ep in range(num_episodes):
            seed = test_seeds[ep % len(test_seeds)]
            test_env = train_env.create_test_env(seed)
            
            obs, _ = test_env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0
            
            final_info = None
            while not (terminated or truncated):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                action, _, _, _ = baseline.get_action_and_value(obs_tensor)
                obs, reward, terminated, truncated, info = test_env.step(action.cpu().numpy())
                episode_return += reward
                episode_length += 1
                if terminated or truncated:
                    final_info = info
                
            # Collect environment metrics (use final_info if available, fallback to get_metrics)
            metrics = None
            if final_info and 'final_metrics' in final_info:
                metrics = final_info['final_metrics']
            elif hasattr(test_env, 'get_metrics'):
                metrics = test_env.get_metrics()
                
            if isinstance(metrics, dict):
                # Add episode return to metrics
                metrics['episode_return'] = episode_return
                metrics['episode_length'] = episode_length
                episode_metrics.append(metrics)
                seed_to_metrics[seed] = metrics.copy()  # Store per-seed metrics
            
            test_env.close()
        
        # Average metrics across episodes
        if episode_metrics:
            avg_metrics = {}
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m and m.get(key) is not None]
                if values:
                    avg_metrics[key] = np.mean(values)
        else:
            avg_metrics = {}
        
        # Return both average and per-seed metrics
        result = {
            'average': avg_metrics,
            'per_seed': seed_to_metrics
        }
        
        return result

    def log_metric_comparison(self, ppo_metrics: Dict, baseline_metrics: Dict, update_count: int):
        """Log comparison between PPO and baseline performance metrics"""
        
        # Extract average and per-seed metrics
        ppo_avg = ppo_metrics.get('average', ppo_metrics)  # Backward compatibility
        baseline_avg = baseline_metrics.get('average', baseline_metrics)
        ppo_per_seed = ppo_metrics.get('per_seed', {})
        baseline_per_seed = baseline_metrics.get('per_seed', {})
        
        # Log per-seed metrics
        if ppo_per_seed and baseline_per_seed:
            seeds = list(ppo_per_seed.keys())
            print(f"\n[Test Results - Update {update_count}]")
            print("=" * 60)
            
            # Log per-seed average waiting time and jobs completed
            for seed in seeds:
                ppo_seed_metrics = ppo_per_seed.get(seed, {})
                baseline_seed_metrics = baseline_per_seed.get(seed, {})
                
                # Average waiting time per seed
                ppo_wait = ppo_seed_metrics.get('avg_waiting_time', 0)
                baseline_wait = baseline_seed_metrics.get('avg_waiting_time', 0)
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Avg_Waiting_Time', ppo_wait, update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Avg_Waiting_Time', baseline_wait, update_count)
                
                # Log waiting time comparison
                print(f"  Seed {seed} Avg Waiting Time: PPO={ppo_wait:.1f}, Baseline={baseline_wait:.1f}")
                
                # Jobs completed per seed
                ppo_jobs = ppo_seed_metrics.get('total_jobs_completed', 0)
                baseline_jobs = baseline_seed_metrics.get('total_jobs_completed', 0)
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Jobs_Completed', ppo_jobs, update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Jobs_Completed', baseline_jobs, update_count)
                
                # Log jobs completed comparison
                print(f"  Seed {seed} Jobs Completed: PPO={ppo_jobs}, Baseline={baseline_jobs}")
                
                # Log other per-seed metrics
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Episode_Return', 
                                          ppo_seed_metrics.get('episode_return', 0), update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Episode_Return', 
                                               baseline_seed_metrics.get('episode_return', 0), update_count)
                
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Core_Utilization', 
                                          ppo_seed_metrics.get('avg_host_core_utilization', 0), update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Core_Utilization', 
                                               baseline_seed_metrics.get('avg_host_core_utilization', 0), update_count)
                
                self.ppo_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Memory_Utilization', 
                                          ppo_seed_metrics.get('avg_host_memory_utilization', 0), update_count)
                self.baseline_writer.add_scalar(f'Comparison_Seed_{seed}/Host_Memory_Utilization', 
                                               baseline_seed_metrics.get('avg_host_memory_utilization', 0), update_count)
            
            print("-" * 60)
        
        # Log average metrics (existing behavior)
        # Episode return comparison
        ppo_return = ppo_avg.get('episode_return', 0)
        baseline_return = baseline_avg.get('episode_return', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Episode_Return', ppo_return, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Episode_Return', baseline_return, update_count)
        
        
        # Average waiting time comparison (IMPORTANT metric)
        ppo_wait = ppo_avg.get('avg_waiting_time', 0)
        baseline_wait = baseline_avg.get('avg_waiting_time', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Avg_Waiting_Time', ppo_wait, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Avg_Waiting_Time', baseline_wait, update_count)
        
        print(f"  AVERAGE Waiting Time: PPO={ppo_wait:.1f}, Baseline={baseline_wait:.1f}")
        
        # Jobs completed comparison (IMPORTANT metric)
        ppo_jobs = ppo_avg.get('total_jobs_completed', 0)
        baseline_jobs = baseline_avg.get('total_jobs_completed', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Jobs_Completed', ppo_jobs, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Jobs_Completed', baseline_jobs, update_count)
        
        print(f"  AVERAGE Jobs Completed: PPO={ppo_jobs}, Baseline={baseline_jobs}")
        
        # Makespan comparison (if available)
        ppo_makespan = ppo_avg.get('makespan')
        baseline_makespan = baseline_avg.get('makespan')
        if ppo_makespan is not None and baseline_makespan is not None:
            self.ppo_writer.add_scalar('Comparison_Avg/Makespan', ppo_makespan, update_count)
            self.baseline_writer.add_scalar('Comparison_Avg/Makespan', baseline_makespan, update_count)
            
            print(f"  AVERAGE Makespan: PPO={ppo_makespan:.0f}, Baseline={baseline_makespan:.0f}")
        
        print("=" * 60)
        
        # Host core utilization comparison
        ppo_core_util = ppo_avg.get('avg_host_core_utilization', 0)
        baseline_core_util = baseline_avg.get('avg_host_core_utilization', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Host_Core_Utilization', ppo_core_util, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Host_Core_Utilization', baseline_core_util, update_count)
        
        # Host memory utilization comparison
        ppo_mem_util = ppo_avg.get('avg_host_memory_utilization', 0)
        baseline_mem_util = baseline_avg.get('avg_host_memory_utilization', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Host_Memory_Utilization', ppo_mem_util, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Host_Memory_Utilization', baseline_mem_util, update_count)
        
        # Average reward per step
        ppo_avg_reward = ppo_avg.get('episode_return', 0) / max(ppo_avg.get('episode_length', 1), 1)
        baseline_avg_reward = baseline_avg.get('episode_return', 0) / max(baseline_avg.get('episode_length', 1), 1)
        self.ppo_writer.add_scalar('Comparison_Avg/Avg_Reward_Per_Step', ppo_avg_reward, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Avg_Reward_Per_Step', baseline_avg_reward, update_count)
        
        # Deferral rate comparison
        ppo_defer = ppo_avg.get('defer_rate', 0)
        baseline_defer = baseline_avg.get('defer_rate', 0)
        self.ppo_writer.add_scalar('Comparison_Avg/Defer_Rate', ppo_defer, update_count)
        self.baseline_writer.add_scalar('Comparison_Avg/Defer_Rate', baseline_defer, update_count)

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
        save_freq: int = 100,
        test_seeds: List[int] = None,
        use_kl_adaptive_lr: bool = False,
        kl_target: float = 0.02,
        combine_kl_with_scheduler: bool = False,
        baseline_policy = None  # Add baseline policy parameter
    ):
        self.policy = policy
        self.env = env
        self.baseline_policy = baseline_policy  # Store baseline policy
        self.gamma = gamma
        self.lam = lam
        self.test_seeds = test_seeds if test_seeds is not None else [42, 43, 44]
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.clip_value_loss = clip_value_loss
        
        # Device setup - optimized for M3 Max
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.policy.to(self.device)
        
        self.lr = lr
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.value_norm_decay = value_norm_decay
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Initialize the unified LR scheduler manager
        # Will be properly initialized in train() with actual total_updates
        self.lr_manager = LRSchedulerManager(
            optimizer=self.optimizer,
            base_lr=lr,
            schedule_type=lr_schedule,
            total_updates=1000,  # Will be updated in train()
            warmup_steps=lr_warmup_steps,
            use_kl_adaptive=use_kl_adaptive_lr,
            kl_target=kl_target,
            combine_kl_with_scheduler=combine_kl_with_scheduler,
            lr_decay_factor=lr_decay_factor,
            lr_min_factor=0.01
        )
        
        # Early stopping
        self.best_test_reward = float('-inf')
        self.best_update = 0  # Track which update had the best performance
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
        
        self.writer = SummaryWriter(tensorboard_log_dir)
        
        # Create separate writers for PPO and Baseline comparison
        self.ppo_writer = SummaryWriter(f"{tensorboard_log_dir}/PPO")
        self.baseline_writer = SummaryWriter(f"{tensorboard_log_dir}/Baseline")
        
        self.metrics_reporter = MetricsReporter(self.writer, self.ppo_writer, self.baseline_writer, test_interval=10)
        self.grad_norms = []
        self.lr_history = []
        self.first_test_run = True  # Flag to save test env data on first test
    
    def set_test_interval(self, interval: int):
        """Set the interval for running test episodes"""
        self.metrics_reporter.test_interval = interval
    
    def save_test_env_data(self, test_seeds: List[int] = None):
        """Save the exact test environment data used during testing"""
        if test_seeds is None:
            test_seeds = [42]
        
        if self.is_vectorized:
            train_env = self.env.envs[0]
        else:
            train_env = self.env
        
        test_data = {
            "test_environments": {}
        }
        
        print("Saving test environment data...")
        for seed in test_seeds:
            test_env = train_env.create_test_env(seed)
            obs, _ = test_env.reset()  # Triggers deterministic generation
            
            rust_env = test_env.rust_env
            
            # Extract host data using the new method
            hosts = rust_env.get_host_configs()
            
            # Extract job schedule data using the new method
            job_schedule = rust_env.get_job_schedule()
            
            test_data["test_environments"][seed] = {
                "hosts": hosts,
                "job_schedule": job_schedule
            }
            
            test_env.close()
        
        # Save to experiment log directory (same as TensorBoard logs)
        output_file = os.path.join(self.writer.log_dir, "test_env_data.json")
        
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Test environment data saved to: {output_file}")
        return output_file
    
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
        
        # Episode tracking
        current_episode_return = 0.0
        current_episode_length = 0
        
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
            
            # Metrics - batch CPU conversion at the end instead
            rollout_metrics['rewards'].append(reward)
            rollout_metrics['values'].append(value.cpu().item())
            rollout_metrics['entropies'].append(entropy.cpu().item())
            
            # Track episode return
            current_episode_return += reward
            current_episode_length += 1
            
            obs = next_obs
            
            if done:
                # Record episode return (sum of rewards)
                rollout_metrics['episode_returns'].append(current_episode_return)
                rollout_metrics['episode_lengths'].append(current_episode_length)
                
                if hasattr(self.env, 'get_metrics'):
                    env_metrics = self.env.get_metrics()
                    for key, val in env_metrics.items():
                        if isinstance(val, (int, float)):
                            rollout_metrics[f'env_{key}'].append(val)
                
                # Reset episode tracking
                current_episode_return = 0.0
                current_episode_length = 0
                
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
        
        # Episode tracking for each environment
        episode_returns = [0.0] * self.num_envs
        episode_lengths = [0] * self.num_envs
        
        for step in range(num_steps):
            # Batch process all environments
            obs_batch = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                # Batch process all environments at once
                batch_actions, batch_log_probs, batch_entropies, batch_values = self.policy.get_action_and_value(obs_batch)
                
                # Only convert actions for env.step()
                actions_np = batch_actions.cpu().numpy()
            
            # Step all environments together
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions_np)
            for env_idx in range(self.num_envs):
                env_obs = obs[env_idx]
                env_action = actions_np[env_idx]
                env_reward = rewards[env_idx]
                env_done = terminated[env_idx] or truncated[env_idx]
                
                # Store in environment-specific buffer
                self.buffers[env_idx].store(env_obs, env_action, env_reward, 
                                          batch_values[env_idx].cpu().item(), 
                                          batch_log_probs[env_idx].cpu().item(), bool(env_done))
                
                rollout_metrics['rewards'].append(env_reward)
                rollout_metrics['values'].append(batch_values[env_idx].cpu().item())
                rollout_metrics['entropies'].append(batch_entropies[env_idx].cpu().item())
                
                # Track episode return
                episode_returns[env_idx] += env_reward
                episode_lengths[env_idx] += 1
                
                # Episode ended for this environment
                if env_done:
                    rollout_metrics['episode_returns'].append(episode_returns[env_idx])
                    rollout_metrics['episode_lengths'].append(episode_lengths[env_idx])
                    episode_returns[env_idx] = 0.0
                    episode_lengths[env_idx] = 0
            
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
    
    def save_checkpoint(self, update_count: int, metrics: Dict, checkpoint_path: str = None):
        """Save model checkpoint with training state"""
        if not self.checkpoint_dir:
            return
            
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_manager_state_dict': self.lr_manager.state_dict(),
            'update_count': update_count,
            'best_test_reward': self.best_test_reward,
            'patience_counter': self.patience_counter,
            'value_mean': self.value_mean,
            'value_var': self.value_var,
            'metrics': metrics,
            'config': {
                'lr': self.lr,
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_coef': self.clip_coef,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm
            },
            'lr_config': {
                'strategy': self.lr_manager.get_description()
            }
        }
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{update_count}.pt')
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest if not a custom path
        if 'best_model' not in checkpoint_path:
            latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
            torch.save(checkpoint, latest_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'lr_manager_state_dict' in checkpoint:
            self.lr_manager.load_state_dict(checkpoint['lr_manager_state_dict'])
        elif 'scheduler_state_dict' in checkpoint:
            # Backward compatibility with old checkpoints
            print("Warning: Loading old checkpoint format, LR scheduler state may not be fully restored")
        
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
                
                # Normalize advantages per minibatch (critical for stable training with sparse rewards)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
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
        
        # Re-initialize LR manager with actual total updates
        estimated_updates = total_timesteps // (rollout_steps * self.num_envs)
        self.lr_manager.total_updates = estimated_updates
        # Recreate the scheduler with the correct total updates
        self.lr_manager.scheduler = self.lr_manager._create_scheduler()
        
        # Set test interval if provided
        if test_interval is not None:
            self.set_test_interval(test_interval)
        
        print(f"Starting PPO training on {self.device}")
        print(f"Policy parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"LR Strategy: {self.lr_manager.get_description()}")
        print(f"TensorBoard logs: {self.writer.log_dir}")
        print(f"Test episodes will run every {self.metrics_reporter.test_interval} updates")
        
        # Run initial test before training begins
        print("\nRunning initial performance test...")
        ppo_metrics = self.test_with_metrics(num_episodes=len(self.test_seeds), update_count=0, policy_name="PPO", test_seeds=self.test_seeds)
        baseline_metrics = self.test_baseline_with_metrics(num_episodes=len(self.test_seeds), update_count=0, test_seeds=self.test_seeds)
        
        # Log initial comparison
        self.log_metric_comparison(ppo_metrics, baseline_metrics, 0)
        ppo_makespan = ppo_metrics.get('makespan', 'N/A')
        baseline_makespan = baseline_metrics.get('makespan', 'N/A')
        print(f"Initial PPO performance - Episode Return: {ppo_metrics.get('episode_return', 0):.1f}, Makespan: {ppo_makespan}")
        print(f"Initial Baseline performance - Episode Return: {baseline_metrics.get('episode_return', 0):.1f}, Makespan: {baseline_makespan}")
        
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

            # Step learning rate scheduler with KL divergence if available
            avg_kl = None
            if 'approx_kl' in update_metrics and update_metrics['approx_kl']:
                avg_kl = np.mean(update_metrics['approx_kl'])
            
            # Step the unified LR manager
            current_lr = self.lr_manager.step(kl_divergence=avg_kl, verbose=(update_count % log_interval == 0))
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
                
                # Log KL divergence and adjustment factor
                if 'approx_kl' in update_metrics and update_metrics['approx_kl']:
                    avg_kl = np.mean(update_metrics['approx_kl'])
                    self.writer.add_scalar('Training/KL_Divergence', avg_kl, update_count)
                    if self.lr_manager.kl_adaptive:
                        self.writer.add_scalar('Training/LR_Adjustment_Factor', 
                                             self.lr_manager.kl_adaptive.adjustment_factor, update_count)
                
                if 'exploration_noise' in update_metrics:
                    self.writer.add_scalar('Training/Exploration_Noise', update_metrics['exploration_noise'][0], update_count)
                
                self.writer.add_scalar('Training/Value_Mean', self.value_mean, update_count)
                self.writer.add_scalar('Training/Value_Var', self.value_var, update_count)
                
                # Metrics are tracked by metrics_reporter, no need for redundant tracking
            
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
                ppo_metrics = self.test_with_metrics(num_episodes=len(self.test_seeds), update_count=update_count, policy_name="PPO", test_seeds=self.test_seeds)
                baseline_metrics = self.test_baseline_with_metrics(num_episodes=len(self.test_seeds), update_count=update_count, test_seeds=self.test_seeds)

                # Compare performance metrics
                self.log_metric_comparison(ppo_metrics, baseline_metrics, update_count)
                
                # Check early stopping based on average episode return
                ppo_avg = ppo_metrics.get('average', ppo_metrics)
                current_test_reward = ppo_avg.get('episode_return', 0)
                
                # Check if performance improved (no threshold now)
                if current_test_reward > self.best_test_reward:
                    self.best_test_reward = current_test_reward
                    self.best_update = update_count  # Track which update was best
                    self.patience_counter = 0
                    print(f"  New best test reward: {self.best_test_reward:.3f} at update {update_count}")
                    
                    # Save checkpoint when we find a new best
                    if self.checkpoint_dir:
                        # Save with update number in filename
                        best_path = os.path.join(self.checkpoint_dir, f'best_model_{update_count}.pt')
                        self.save_checkpoint(update_count, self.get_training_summary(), best_path)
                        print(f"  Saved new best model to {best_path}")
                        
                        # Also save/overwrite as 'best_model.pt' for easy access
                        latest_best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                        self.save_checkpoint(update_count, self.get_training_summary(), latest_best_path)
                else:
                    self.patience_counter += 1
                    
                    # Check if performance is significantly dropping
                    performance_drop = (self.best_test_reward - current_test_reward) / abs(self.best_test_reward) if self.best_test_reward != 0 else 0
                    
                    if performance_drop > 0.1:  # More than 10% drop from best
                        print(f"  WARNING: Performance dropped {performance_drop*100:.1f}% from best. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                        
                        # Accelerate early stopping if performance is consistently dropping
                        if performance_drop > 0.2:  # More than 20% drop
                            self.patience_counter += 2  # Count as 3 strikes instead of 1
                            print(f"  SEVERE performance drop detected. Accelerating early stopping.")
                    else:
                        print(f"  No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
                    
                if self.patience_counter >= self.early_stopping_patience:
                    self.should_stop = True
                    print(f"  Early stopping triggered.")
                    print(f"  Best model was at update {self.best_update} with reward {self.best_test_reward:.3f}")
                    
                    # Load the best model before stopping
                    if self.checkpoint_dir:
                        best_path = os.path.join(self.checkpoint_dir, f'best_model_{self.best_update}.pt')
                        if os.path.exists(best_path):
                            self.load_checkpoint(best_path)
                            print(f"  Loaded best model from {best_path}")
                        else:
                            # Fallback to generic best_model.pt if specific version not found
                            fallback_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                            if os.path.exists(fallback_path):
                                self.load_checkpoint(fallback_path)
                                print(f"  Loaded best model from {fallback_path}")

        training_time = time.time() - start_time
        print(f"\n{'='*60}")
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
        
        # Report best model information
        print(f"\n{'='*60}")
        print(f"BEST MODEL SUMMARY:")
        print(f"  Update: {self.best_update}")
        print(f"  Test Reward: {self.best_test_reward:.3f}")
        if self.checkpoint_dir:
            print(f"  Saved as: best_model_{self.best_update}.pt")
        print(f"{'='*60}")
        
        self.writer.close()
        self.ppo_writer.close()
        self.baseline_writer.close()
        return self.get_training_summary()
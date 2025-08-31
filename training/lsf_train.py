#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import numpy as np
from wrapper.gym_wrapper import make_lsf_env
from training.mlp_model import MLPPolicy
from training.variable_host_model import VariableHostPolicy
from training.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train PPO agent on LSF scheduling environment with arrival-time batching')
    
    # Environment args - OPTIMIZED FOR JOB CYCLES  
    parser.add_argument('--num-hosts', type=int, default=30, help='Number of hosts in the cluster')
    parser.add_argument('--episode-length', type=int, default=150, help='Episode length in steps (matches rollout steps)')
    parser.add_argument('--max-jobs-per-step', type=int, default=30, help='Maximum jobs per step - much higher load for resource pressure')
    parser.add_argument('--max-queue-length', type=int, default=100*100, help='Maximum queue length')
    
    # Host/Job resource ranges - INCREASED LOAD FOR SCHEDULING PRESSURE
    parser.add_argument('--host-cores-min', type=int, default=16, help='Minimum host cores')
    parser.add_argument('--host-cores-max', type=int, default=32, help='Maximum host cores')
    parser.add_argument('--host-memory-min', type=int, default=64*1024, help='Minimum host memory (MB)')
    parser.add_argument('--host-memory-max', type=int, default=128*1024, help='Maximum host memory (MB)')
    parser.add_argument('--job-cores-min', type=int, default=1, help='Minimum job cores - bigger minimum jobs')
    parser.add_argument('--job-cores-max', type=int, default=4, help='Maximum job cores - larger synthesis/PnR jobs')
    parser.add_argument('--job-memory-min', type=int, default=1*512, help='Minimum job memory (MB) - 4GB realistic minimum')
    parser.add_argument('--job-memory-max', type=int, default=4*1024, help='Maximum job memory (MB) - 16GB for larger jobs')
    parser.add_argument('--job-duration-min', type=int, default=5, help='Minimum job duration (seconds) - shorter for more turnover')
    parser.add_argument('--job-duration-max', type=int, default=90, help='Maximum job duration (seconds) - moderate length jobs')
    
    # Training args - OPTIMIZED FOR BATCH REWARDS & JOB CYCLES
    # Formula: total_timesteps = rollout_steps × num_envs × num_updates
    # Default: 6144 × 3 × 4096 = 33,554,432 timesteps (~2048 updates)
    parser.add_argument('--total-timesteps', type=int, default=6144*2*2048, help='Total training timesteps: rollout_steps × num_envs × 2048_updates')
    parser.add_argument('--rollout-steps', type=int, default=6144, help='Steps per rollout - capture multiple job cycles')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate optimized for batch rewards')
    parser.add_argument('--gamma', type=float, default=0.995, help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.98, help='GAE lambda')
    parser.add_argument('--clip-coef', type=float, default=0.3, help='PPO clip coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient - higher for exploration')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value function coefficient')
    parser.add_argument('--update-epochs', type=int, default=2, help='Number of update epochs')
    parser.add_argument('--minibatch-size', type=int, default=512, help='Minibatch size - larger for stable gradients')
    parser.add_argument('--buffer-size', type=int, default=6144, help='Rollout buffer size - MATCHES ROLLOUT STEPS')

    # System args
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=5, help='Logging interval - more frequent for small scale')
    parser.add_argument('--save-model', type=str, default=None, help='Path to save the trained model')
    
    # Advanced RL techniques
    parser.add_argument('--lr-schedule', type=str, default='cosine', 
                       choices=['constant', 'linear', 'exponential', 'cosine', 'warmup_cosine'],
                       help='Learning rate schedule')
    parser.add_argument('--lr-decay-factor', type=float, default=0.995, help='Learning rate decay factor for exponential schedule')
    parser.add_argument('--lr-warmup-steps', type=int, default=100, help='Warmup steps for warmup schedules')
    parser.add_argument('--early-stopping-patience', type=int, default=200, help='Early stopping patience - reasonable for longer training')
    parser.add_argument('--early-stopping-threshold', type=float, default=0.01, help='Early stopping improvement threshold')
    parser.add_argument('--value-norm-decay', type=float, default=0.99, help='Value normalization decay factor')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--save-freq', type=int, default=250, help='Checkpoint save frequency - more frequent for experiments')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--exploration-noise-decay', type=float, default=0.998, help='Exploration noise decay factor - slower decay')
    parser.add_argument('--min-exploration-noise', type=float, default=0.01, help='Minimum exploration noise')
    parser.add_argument('--num-envs', type=int, default=2, help='Number of parallel environments - more for better sampling')
    parser.add_argument('--tensorboard-dir', type=str, default="config-save", help='TensorBoard log directory (auto-generated if not specified)')
    
    return parser.parse_args()


def create_model(obs_dim, action_dim, num_hosts, exploration_noise_decay=0.995, min_exploration_noise=0.01):
    """Create the actor-critic policy."""
    # return ActorCriticPolicy(
    #     obs_dim=obs_dim, 
    #     action_dim=action_dim, 
    #     num_hosts=num_hosts,
    #     exploration_noise_decay=exploration_noise_decay,
    #     min_exploration_noise=min_exploration_noise
    # )
    # return MLPPolicy(
    #     obs_dim=obs_dim, 
    #     action_dim=action_dim, 
    #     num_hosts=num_hosts,
    #     exploration_noise_decay=exploration_noise_decay,
    #     min_exploration_noise=min_exploration_noise
    # )
    return VariableHostPolicy(
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        num_hosts=num_hosts,
        exploration_noise_decay=exploration_noise_decay,
        min_exploration_noise=min_exploration_noise
    )


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.tensorboard_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.tensorboard_dir = f"logs/ppo_training_{timestamp}"
    else:
        # Ensure tensorboard_dir is always under logs/
        args.tensorboard_dir = f"logs/{args.tensorboard_dir}"
    
    if args.checkpoint_dir is None:
        # Extract the experiment name from tensorboard_dir and put under checkpoints/
        experiment_name = args.tensorboard_dir.replace("logs/", "")
        args.checkpoint_dir = f"checkpoints/{experiment_name}"
    
    print("=== LSF Scheduler PPO Training (Advanced) ===")
    print(f"Environment config:")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Hosts: {args.num_hosts}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Max jobs per step: {args.max_jobs_per_step}")
    print(f"  Max queue length: {args.max_queue_length}")
    print(f"  Host cores: {args.host_cores_min}-{args.host_cores_max}")
    print(f"  Host memory: {args.host_memory_min}-{args.host_memory_max} MB")
    print(f"  Job cores: {args.job_cores_min}-{args.job_cores_max}")
    print(f"  Job memory: {args.job_memory_min}-{args.job_memory_max} MB")
    print(f"  Job duration: {args.job_duration_min}-{args.job_duration_max} seconds")
    print(f"Training config:")
    print(f"  Total timesteps: {args.total_timesteps}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Learning rate schedule: {args.lr_schedule}")
    print(f"  TensorBoard logs: {args.tensorboard_dir}")
    print()
    
    # Create environment using our updated wrapper
    env_kwargs = {
        'num_hosts': args.num_hosts,
        'max_queue_length': args.max_queue_length,
        'host_cores_range': (args.host_cores_min, args.host_cores_max),
        'host_memory_range': (args.host_memory_min, args.host_memory_max),
        'job_cores_range': (args.job_cores_min, args.job_cores_max),
        'job_memory_range': (args.job_memory_min, args.job_memory_max),
        'job_duration_range': (args.job_duration_min, args.job_duration_max),
        'max_jobs_per_step': args.max_jobs_per_step,
        'max_time': args.episode_length,
        'seed': args.seed,
    }
    
    env = make_lsf_env(num_envs=args.num_envs, **env_kwargs)
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print()
    
    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy = create_model(obs_dim, action_dim, args.num_hosts, 
                         args.exploration_noise_decay, args.min_exploration_noise)
    
    print(f"Policy created with {sum(p.numel() for p in policy.parameters()):,} parameters")
    print()
    
    # Create trainer with advanced features
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        buffer_size=args.buffer_size,
        device=args.device,
        tensorboard_log_dir=args.tensorboard_dir,
        lr_schedule=args.lr_schedule,
        lr_decay_factor=args.lr_decay_factor,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        value_norm_decay=args.value_norm_decay,
        checkpoint_dir=args.checkpoint_dir,
        save_freq=args.save_freq
    )
    
    # Resume from checkpoint if specified
    start_update = 0
    if args.resume_from:
        start_update = trainer.load_checkpoint(args.resume_from)
        print(f"Resumed training from update {start_update}")
    
    # Train
    try:
        print("Starting advanced PPO training with:")
        print(f"- Learning rate schedule: {args.lr_schedule}")
        print(f"- Early stopping patience: {args.early_stopping_patience}")
        print(f"- Value normalization decay: {args.value_norm_decay}")
        print(f"- Exploration noise decay: {args.exploration_noise_decay}")
        if args.checkpoint_dir:
            print(f"- Checkpoints saved to: {args.checkpoint_dir}")
        print()
        
        metrics = trainer.train(
            total_timesteps=args.total_timesteps,
            rollout_steps=args.rollout_steps,
            log_interval=args.log_interval
        )
        
        print("\nTraining completed successfully!")
        
        # Save model if requested
        if args.save_model:
            torch.save(policy.state_dict(), args.save_model)
            print(f"Model saved to {args.save_model}")
        
        if 'training_metrics' in metrics and 'reward' in metrics['training_metrics']:
            if metrics['training_metrics']['reward']:
                final_reward = metrics['training_metrics']['reward'][-1]
                print(f"Final average reward: {final_reward:.3f}")
        elif 'rewards' in metrics and metrics['rewards']:
            final_reward = metrics['rewards'][-1]
            print(f"Final average reward: {final_reward:.3f}")
        
        try:
            env_metrics = env.get_metrics()
            print(f"\nFinal environment metrics:")
            
            metric_names = {
                'total_jobs_completed': 'Total jobs completed',
                'completion_rate': 'Completion rate', 
                'avg_host_core_utilization': 'Average core utilization',
                'avg_host_memory_utilization': 'Average memory utilization',
                'avg_waiting_time': 'Average waiting time'
            }
            
            for key, label in metric_names.items():
                if key in env_metrics:
                    value = env_metrics[key]
                    if isinstance(value, float):
                        print(f"  {label}: {value:.3f}")
                    else:
                        print(f"  {label}: {value}")
                        
        except Exception as e:
            print(f"Environment metrics not available: {e}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
        if args.save_model:
            save_path = args.save_model.replace('.pth', '_interrupted.pth')
            torch.save(policy.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    
    finally:
        env.close()


if __name__ == "__main__":
    main()
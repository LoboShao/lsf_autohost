#!/usr/bin/env python3
"""
Simple episode runner - just get data from env.rs every step
"""
import sys
import os
import json
import time
import argparse

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import torch
import numpy as np
from src.wrapper.gym_wrapper import make_lsf_env
from src.training.variable_host_model import VariableHostPolicy

def run_episode_simple(log_dir, seed=42):
    """Simple episode runner - get all data from env.rs every step."""
    
    try:
        # Load test data
        test_data_path = os.path.join(project_root, log_dir, 'test_env_data.json')
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Extract config
        env_data = test_data['test_environments']['42']
        hosts = env_data['hosts']
        job_schedule = env_data['job_schedule']
        
        env_config = {
            'num_hosts': len(hosts),
            'max_time': job_schedule['max_time'],
            'max_jobs_per_step': job_schedule['max_jobs_per_step'],
            'host_cores_range': (min(h['total_cores'] for h in hosts), max(h['total_cores'] for h in hosts)),
            'host_memory_range': (min(h['total_memory'] for h in hosts), max(h['total_memory'] for h in hosts)),
            'job_cores_range': (min(job_schedule['job_cores_schedule']), max(job_schedule['job_cores_schedule'])),
            'job_memory_range': (min(job_schedule['job_memory_schedule']), max(job_schedule['job_memory_schedule'])),
            'job_duration_range': (min(job_schedule['job_duration_schedule']), max(job_schedule['job_duration_schedule']))
        }
        
        # Load model
        checkpoint_dir = os.path.join(project_root, log_dir, 'checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        num_hosts = env_config['num_hosts']
        model = VariableHostPolicy(obs_dim=num_hosts * 4 + 2, action_dim=num_hosts, num_hosts=num_hosts)
        
        if 'policy_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        # Create environment
        env = make_lsf_env(num_envs=1, **env_config)
        test_env = env.create_test_env(seed)
        obs, info = test_env.reset()
        
        start_data = {'seed': seed, 'num_hosts': num_hosts}
        print("EPISODE_START:" + json.dumps(start_data), flush=True)
        
        # Simple episode loop - just step and get data from env.rs
        visualizer_step = 0
        done = False
        
        while not done:
            visualizer_step += 1
            
            # Make decision if needed
            if test_env.rust_env.needs_decision():
                # Get model action
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)
                    # Ensure proper format: 1-D numpy array of float32
                    action_array = action.squeeze().cpu().numpy().astype(np.float32)
                
                # Step with action
                obs, reward, terminated, truncated, info = test_env.step(action_array)
            else:
                # Step without decision
                dummy_action = np.zeros(num_hosts, dtype=np.float32)
                obs, reward, terminated, truncated, info = test_env.step(dummy_action)
            
            done = terminated or truncated
            
            # Use get_visualization_data() which provides ALL data the visualizer needs!
            viz_data = None
            try:
                if hasattr(test_env, 'rust_env') and hasattr(test_env.rust_env, 'get_visualization_data'):
                    viz_data = test_env.rust_env.get_visualization_data()
            except Exception as e:
                pass  # Silently fall back to info dict
            
            # Extract all data from get_visualization_data() 
            if viz_data:
                env_time = int(viz_data.get('current_time', 0))
                total_jobs = int(viz_data.get('total_jobs_generated', 0))
                completed_jobs = int(viz_data.get('total_jobs_completed', 0))
                failed_jobs = int(viz_data.get('total_jobs_failed', 0))
                active_jobs = int(viz_data.get('active_jobs_count', 0))
                job_queue = int(viz_data.get('job_queue_length', 0))
                submission_queue = int(viz_data.get('submission_queue_length', 0))
                deferred_jobs = int(viz_data.get('deferred_jobs_length', 0))
                needs_decision = bool(viz_data.get('needs_decision', False))
                episode_done = bool(viz_data.get('episode_done', False))
                hosts_data = viz_data.get('hosts', [])
                time_source = "get_visualization_data()"
            else:
                # Fallback to info dict and state
                env_time = info.get('current_time', 0) if info else 0
                total_jobs = info.get('total_jobs_generated', info.get('jobs_scheduled', 0)) if info else 0
                completed_jobs = info.get('total_jobs_completed', 0) if info else 0
                failed_jobs = info.get('total_jobs_failed', 0) if info else 0
                active_jobs = info.get('active_jobs', 0) if info else 0
                job_queue = info.get('queue_length', 0) if info else 0
                submission_queue = info.get('submission_queue_length', 0) if info else 0
                deferred_jobs = 0
                needs_decision = test_env.rust_env.needs_decision() if hasattr(test_env, 'rust_env') else False
                episode_done = done
                time_source = "info dict fallback"
                
                # Generate hosts_data from state
                state = test_env.rust_env.get_state()
                hosts_data = []
                for i in range(num_hosts):
                    base_idx = i * 4
                    hosts_data.append({
                        'id': i,
                        'cpu_util': float(state[base_idx]) * 100,
                        'memory_util': float(state[base_idx + 1]) * 100
                    })
            
            
            # Stream all env.rs data to visualizer
            episode_data = {
                'visualizer_step': visualizer_step,
                'env_time': int(env_time),
                'hosts': hosts_data,
                'env_data': {
                    'total_jobs': int(total_jobs),
                    'completed_jobs': int(completed_jobs),
                    'failed_jobs': int(failed_jobs),
                    'active_jobs': int(active_jobs),
                    'job_queue_length': int(job_queue),
                    'deferred_jobs_length': int(deferred_jobs) if 'deferred_jobs' in locals() else 0,
                    'needs_decision': bool(needs_decision),
                    'episode_done': bool(episode_done) if 'episode_done' in locals() else bool(done)
                }
            }
            print("EPISODE_DATA:" + json.dumps(episode_data), flush=True)
            
            # Update rate (with pause support)
            time.sleep(0.1)
            
            # Safety limit
            if visualizer_step > 10000:
                done = True
        
        test_env.close()
        env.close()
        complete_data = {'final_step': visualizer_step, 'final_time': env_time}
        print("EPISODE_COMPLETE:" + json.dumps(complete_data), flush=True)
        
    except Exception as e:
        error_data = {'error': str(e)}
        print("ERROR:" + json.dumps(error_data), flush=True)
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    run_episode_simple(args.log_dir, args.seed)

if __name__ == "__main__":
    main()
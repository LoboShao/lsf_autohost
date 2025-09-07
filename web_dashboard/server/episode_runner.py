#!/usr/bin/env python3
"""
Episode runner for web dashboard - streams real-time data from env.rs
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

def stream_episode_data(log_dir, seed=42):
    """Run episode and stream data to stdout for web dashboard."""
    
    try:
        # Load test data to get exact configuration
        test_data_path = os.path.join(project_root, log_dir, 'test_env_data.json')
        print(f"DEBUG:Loading test data from {test_data_path}", flush=True)
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        # Extract environment configuration from test data
        env_data = test_data['test_environments']['42']
        hosts = env_data['hosts']
        job_schedule = env_data['job_schedule']
        
        env_config = {
            'num_hosts': len(hosts),
            'max_time': job_schedule['max_time'],
            'max_jobs_per_step': job_schedule['max_jobs_per_step'],
            'host_cores_range': (
                min(h['total_cores'] for h in hosts),
                max(h['total_cores'] for h in hosts)
            ),
            'host_memory_range': (
                min(h['total_memory'] for h in hosts),
                max(h['total_memory'] for h in hosts)
            ),
            'job_cores_range': (
                min(job_schedule['job_cores_schedule']),
                max(job_schedule['job_cores_schedule'])
            ),
            'job_memory_range': (
                min(job_schedule['job_memory_schedule']),
                max(job_schedule['job_memory_schedule'])
            ),
            'job_duration_range': (
                min(job_schedule['job_duration_schedule']),
                max(job_schedule['job_duration_schedule'])
            )
        }
        
        print(f"DEBUG:Environment config: {env_config}", flush=True)
        
        # Find and load model checkpoint
        checkpoint_dir = os.path.join(project_root, log_dir, 'checkpoints')
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        
        checkpoint_path = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
        print(f"DEBUG:Loading model from {checkpoint_path}", flush=True)
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        num_hosts = env_config['num_hosts']
        model = VariableHostPolicy(
            obs_dim=num_hosts * 4 + 2, 
            action_dim=num_hosts, 
            num_hosts=num_hosts
        )
        
        if 'policy_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['policy_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        print(f"DEBUG:Model loaded successfully", flush=True)
        
        # Create environment
        env = make_lsf_env(num_envs=1, **env_config)
        test_env = env.create_test_env(seed)
        obs, info = test_env.reset()
        
        print(f"DEBUG:Environment created and reset with seed {seed}", flush=True)
        
        # Episode execution with real-time streaming
        current_time = 0
        step_count = 0
        done = False
        decisions_made = 0
        episode_length = env_config['max_time']
        
        print(f"EPISODE_START:{json.dumps({'seed': seed, 'max_time': episode_length, 'num_hosts': num_hosts})}", flush=True)
        
        # Episode continues until ALL jobs finish, not just max_time
        # Jobs arrive until max_time, then environment runs until completion
        while not done:
            # Get current state from env.rs
            state = test_env.rust_env.get_state()
            
            # Extract host utilization data from state vector
            hosts_data = []
            for i in range(num_hosts):
                base_idx = i * 4  # 4 features per host: cpu_util, mem_util, cpu_norm, mem_norm
                host_data = {
                    'id': i,
                    'cpu_util': float(state[base_idx]) * 100,  # Convert to percentage
                    'memory_util': float(state[base_idx + 1]) * 100,  # Convert to percentage
                    'cpu_capacity_norm': float(state[base_idx + 2]),  # Normalized capacity
                    'memory_capacity_norm': float(state[base_idx + 3])  # Normalized capacity
                }
                hosts_data.append(host_data)
            
            # Check if scheduling decision is needed
            if test_env.rust_env.needs_decision():
                decisions_made += 1
                
                # Extract job requirements from state (last 2 elements)
                job_features = state[-2:]
                job_cores_norm = float(job_features[0])
                job_memory_norm = float(job_features[1])
                
                # Estimate actual job requirements for display
                job_cores_est = int(job_cores_norm * env_config['job_cores_range'][1])
                job_memory_est = int(job_memory_norm * env_config['job_memory_range'][1])
                
                # Get model decision - using the same format as PPO training
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action, _, _, _ = model.get_action_and_value(obs_tensor, deterministic=True)
                    # Make sure action has correct shape for all hosts
                    if action.dim() > 1:
                        action_array = action.squeeze().cpu().numpy()  # Remove batch dimension
                    else:
                        action_array = action.cpu().numpy()
                    
                    # Ensure action_array has correct length
                    if len(action_array) != num_hosts:
                        print(f"DEBUG:Action shape mismatch - expected {num_hosts}, got {len(action_array)}", flush=True)
                        # Fallback to uniform priorities if shape is wrong
                        action_array = np.ones(num_hosts, dtype=np.float32) / num_hosts
                
                selected_host = int(np.argmax(action_array))
                priority_score = float(action_array[selected_host])
                
                # Get expected job arrivals for this second (if available)
                job_arrivals_this_second = 0
                if current_time < len(job_schedule['job_arrival_schedule']):
                    job_arrivals_this_second = job_schedule['job_arrival_schedule'][current_time]
                
                # Get queue information before using it
                queue_length = info.get('queue_length', 0) if info else 0
                
                # Stream decision data with timing context (ensure all values are serializable)
                decision_data = {
                    'time': int(current_time),
                    'job_cores': int(job_cores_est),
                    'job_memory': int(job_memory_est),
                    'job_cores_norm': float(job_cores_norm),
                    'job_memory_norm': float(job_memory_norm),
                    'selected_host': int(selected_host),
                    'priority_score': float(priority_score),
                    'decisions_made': int(decisions_made),
                    'jobs_this_second': int(job_arrivals_this_second),
                    'queue_length': int(queue_length)
                }
                print(f"DECISION:{json.dumps(decision_data)}", flush=True)
                
                # Step environment with model action
                obs, reward, terminated, truncated, info = test_env.step(action_array)
                done = terminated or truncated
                step_count += 1
                
            else:
                # No decision needed, advance time
                dummy_action = np.zeros(num_hosts, dtype=np.float32)
                obs, reward, terminated, truncated, info = test_env.step(dummy_action)
                done = terminated or truncated
            
            # Get current time from Rust environment (it's always in info dict)
            time_source = "fallback"
            if info and 'current_time' in info:
                current_time = int(info['current_time'])  # Convert to int since it's u64 in Rust
                time_source = "info['current_time']"
            else:
                # Fallback: manually track time (should not happen)
                current_time += 1
                time_source = "manual increment"
            
            # Debug time tracking every 50 steps
            if step_count % 50 == 0:
                print(f"DEBUG:Step {step_count} - env_time:{current_time}, job_queue:{queue_length if 'queue_length' in locals() else 'N/A'}, needs_decision:{test_env.rust_env.needs_decision() if hasattr(test_env, 'rust_env') else 'N/A'}, time_source:{time_source}", flush=True)
            
            # Get comprehensive metrics from env.rs (same as training code)
            env_metrics = test_env.rust_env.get_metrics() if hasattr(test_env, 'rust_env') else {}
            
            # Extract metrics from both info and get_metrics() for completeness
            if info:
                total_jobs = info.get('total_jobs_generated', 0)
                completed_jobs = info.get('total_jobs_completed', 0)
                failed_jobs = info.get('total_jobs_failed', 0)
                active_jobs = info.get('active_jobs', 0)
                queue_length = info.get('queue_length', 0)
                # submission_queue_length removed per user request
                
                # Cross-check with get_metrics() output - use metrics as authoritative source
                if env_metrics:
                    metrics_completed = env_metrics.get('total_jobs_completed', completed_jobs)
                    if metrics_completed != completed_jobs:
                        print(f"DEBUG:Completion mismatch - info:{completed_jobs}, metrics:{metrics_completed} (using metrics)", flush=True)
                    completed_jobs = metrics_completed  # Always use get_metrics() as authoritative source
                
                # Try to get detailed queue states (if new functions are available)
                try:
                    queue_states = test_env.rust_env.get_queue_states()
                    deferred_jobs_count = queue_states.get('deferred_jobs_length', 0)
                    pending_completions = queue_states.get('pending_completions', 0)
                except AttributeError:
                    deferred_jobs_count = 0
                    pending_completions = 0
                
                # Enhanced debug for stuck episodes
                if current_time > episode_length + 50 and active_jobs > 0 and completed_jobs == 0:
                    if current_time % 25 == 0:  # More frequent logging when no jobs completing
                        print(f"DEBUG:No job completions at {current_time}s - active:{active_jobs}, completed:{completed_jobs}, failed:{failed_jobs}", flush=True)
                        print(f"DEBUG:Queue state - job_queue:{queue_length}, submission:{submission_queue_length}, deferred:{deferred_jobs_count}", flush=True)
                        
                        # Check if this is an env.rs completion processing bug
                        if active_jobs > 0:
                            print(f"DEBUG:POTENTIAL BUG - {active_jobs} jobs active but 0 completed after {current_time}s", flush=True)
                            print(f"DEBUG:This suggests jobs are being scheduled but never marked as complete", flush=True)
                            print(f"DEBUG:With job durations 20-120s, jobs should complete by now", flush=True)
                        
                        # Try to get active job details for debugging
                        try:
                            active_jobs_info = test_env.rust_env.get_active_jobs_info()
                            if len(active_jobs_info) > 0:
                                sample_job = active_jobs_info[0]
                                print(f"DEBUG:Sample active job - remaining_time:{sample_job.get('remaining_time', 'N/A')}, expected_completion:{sample_job.get('expected_completion_time', 'N/A')}", flush=True)
                        except AttributeError:
                            pass
            else:
                total_jobs = completed_jobs = failed_jobs = active_jobs = queue_length = 0
                deferred_jobs_count = pending_completions = 0
            
            # Calculate metrics
            completion_rate = (completed_jobs / max(total_jobs, 1)) * 100 if total_jobs > 0 else 0
            
            # Add timing context to episode data
            jobs_arriving_this_second = 0
            if current_time < len(job_schedule['job_arrival_schedule']):
                jobs_arriving_this_second = job_schedule['job_arrival_schedule'][current_time]
            
            # Calculate expected total episode duration for better progress tracking
            # Estimate completion time: arrival phase + average job duration
            avg_job_duration = (env_config['job_duration_range'][0] + env_config['job_duration_range'][1]) / 2
            estimated_completion_time = episode_length + avg_job_duration
            
            # Stream episode update with all real-time data from env.rs
            episode_data = {
                'visualizer_step': step_count,
                'env_time': current_time,
                'time': current_time,  # Backwards compatibility
                'hosts': hosts_data,
                'env_data': {
                    'total_jobs': total_jobs,
                    'completed_jobs': completed_jobs,
                    'failed_jobs': failed_jobs,
                    'active_jobs': active_jobs,
                    'job_queue_length': queue_length,
                    'needs_decision': test_env.rust_env.needs_decision() if hasattr(test_env, 'rust_env') else False,
                    'episode_done': done
                },
                'metrics': {
                    'total_jobs': total_jobs,
                    'completed_jobs': completed_jobs,
                    'failed_jobs': failed_jobs,
                    'active_jobs': active_jobs,
                    'decisions_made': decisions_made,
                    'completion_rate': completion_rate,
                    'jobs_arriving_this_second': jobs_arriving_this_second,
                    'in_arrival_phase': current_time < episode_length,
                    'in_completion_phase': current_time >= episode_length,
                    'estimated_total_duration': int(estimated_completion_time),
                    'job_arrival_end': episode_length,
                    'deferred_jobs_count': deferred_jobs_count,
                    'pending_completions': pending_completions
                }
            }
            print(f"EPISODE_DATA:{json.dumps(episode_data)}", flush=True)
            
            # Control update rate (10 FPS for smooth visualization)
            time.sleep(0.1)
            
            # Safety check to prevent infinite loops (much higher limits)
            if step_count > episode_length * 100:  # Allow much longer episodes
                print(f"DEBUG:Step safety limit reached - steps:{step_count}, time:{current_time}s", flush=True)
                print(f"DEBUG:Final state - active:{active_jobs}, completed:{completed_jobs}, failed:{failed_jobs}", flush=True)
                done = True  # Force episode end
            
            # Log natural episode termination
            if done and current_time < episode_length + 200:
                print(f"DEBUG:Episode ended naturally at {current_time}s - active:{active_jobs}, completed:{completed_jobs}", flush=True)
            
            # Log phase transitions
            if current_time == episode_length:
                print(f"DEBUG:Job arrival phase complete at {current_time}s, entering completion phase", flush=True)
            elif current_time % 50 == 0 and current_time > episode_length:
                print(f"DEBUG:Completion phase continuing at {current_time}s, active_jobs={active_jobs}", flush=True)
        
        # Episode completed
        test_env.close()
        env.close()
        
        final_data = {
            'episode_complete': True,
            'final_time': current_time,
            'total_steps': step_count,
            'decisions_made': decisions_made,
            'completion_rate': completion_rate
        }
        print(f"EPISODE_COMPLETE:{json.dumps(final_data)}", flush=True)
        print(f"DEBUG:Episode finished successfully", flush=True)
        
    except Exception as e:
        error_data = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(f"ERROR:{json.dumps(error_data)}", flush=True)
        import traceback
        print(f"DEBUG:Full traceback: {traceback.format_exc()}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True, help='Log directory relative to project root')
    parser.add_argument('--seed', type=int, default=42, help='Test seed')
    
    args = parser.parse_args()
    
    print(f"DEBUG:Starting episode runner with log_dir={args.log_dir}, seed={args.seed}", flush=True)
    stream_episode_data(args.log_dir, args.seed)

if __name__ == "__main__":
    main()

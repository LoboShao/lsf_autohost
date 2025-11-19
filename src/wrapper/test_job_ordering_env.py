#!/usr/bin/env python3
"""
Test script to verify that JobOrderingEnv can run without errors.
Tests basic functionality: reset, step, state shapes, and episode completion.
"""

import numpy as np
from job_ordering_wrapper import JobOrderingEnvWrapper


def test_basic_functionality(seed=42, num_steps=100, num_hosts=50, max_buckets=100, max_time=100):
    """
    Test basic environment functionality to ensure it runs without errors.

    Args:
        seed: Random seed for deterministic testing
        num_steps: Number of steps to test
        num_hosts: Number of hosts in the cluster
        max_buckets: Maximum number of job buckets
        max_time: Maximum time for the episode

    Returns:
        bool: True if all tests pass, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing JobOrderingEnv basic functionality")
    print(f"Configuration: num_hosts={num_hosts}, max_buckets={max_buckets}, max_time={max_time}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    try:
        # Create environment
        env = JobOrderingEnvWrapper(
            num_hosts=num_hosts,
            max_buckets=max_buckets,
            max_time=max_time,
            seed=seed
        )

        print(f"✓ Environment created successfully")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")

        # Test reset
        obs, info = env.reset(seed=seed)
        print(f"\n✓ Environment reset successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected shape: ({max_buckets * 2 + 2},)")

        # Check observation shape
        expected_obs_shape = (max_buckets * 2 + 2,)
        if obs.shape != expected_obs_shape:
            print(f"❌ ERROR: Observation shape mismatch!")
            print(f"  Expected: {expected_obs_shape}")
            print(f"  Got: {obs.shape}")
            return False

        print(f"  First 10 values: {obs[:10]}")
        print(f"  Last 10 values: {obs[-10:]}")

        # Generate deterministic random actions
        np.random.seed(seed)

        # Track metrics
        rewards = []
        episode_count = 0
        steps_completed = 0

        # Step through environment
        print(f"\nRunning {num_steps} steps...")
        for i in range(num_steps):
            # Generate random action (bucket priorities)
            action = np.random.random(max_buckets).astype(np.float32)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            steps_completed += 1

            # Check observation shape
            if obs.shape != expected_obs_shape:
                print(f"❌ Step {i+1}: Observation shape changed!")
                print(f"  Expected: {expected_obs_shape}")
                print(f"  Got: {obs.shape}")
                return False

            # Track episodes
            if terminated:
                episode_count += 1
                print(f"  Episode {episode_count} completed at step {i+1}")

            # Print progress
            if (i + 1) % 20 == 0:
                print(f"  Step {i+1}: reward={reward:.6f}, terminated={terminated}")

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"✓ All {steps_completed} steps completed successfully")
        print(f"✓ Episodes completed: {episode_count}")
        print(f"✓ Total reward: {sum(rewards):.6f}")
        print(f"✓ Average reward: {np.mean(rewards):.6f}")
        print(f"✓ Min reward: {min(rewards):.6f}")
        print(f"✓ Max reward: {max(rewards):.6f}")

        # Get final metrics
        try:
            metrics = env.get_metrics()
            print(f"\nFinal Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Could not retrieve metrics: {e}")

        # Test cluster info
        try:
            cluster_info = env.get_cluster_info()
            print(f"\nCluster Info:")
            for key, value in cluster_info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"  Could not retrieve cluster info: {e}")

        print(f"\n{'='*60}")
        print(f"✅ ALL TESTS PASSED")
        print(f"{'='*60}")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ ERROR: Test failed with exception!")
        print(f"{'='*60}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_format(seed=42, num_hosts=10, max_buckets=20):
    """
    Test the state format to ensure it matches expected structure.

    State format: [bucket0_cores, bucket0_count, bucket1_cores, bucket1_count, ...,
                   available_cores_ratio, available_mem_ratio]
    """
    print(f"\n{'='*60}")
    print(f"Testing JobOrderingEnv state format")
    print(f"{'='*60}\n")

    try:
        env = JobOrderingEnvWrapper(
            num_hosts=num_hosts,
            max_buckets=max_buckets,
            max_time=50,
            seed=seed
        )

        obs, _ = env.reset(seed=seed)

        print(f"State shape: {obs.shape}")
        print(f"Expected: ({max_buckets * 2 + 2},)")

        # Parse state
        bucket_features = obs[:max_buckets * 2]
        global_features = obs[-2:]

        print(f"\nBucket features (first 10): {bucket_features[:10]}")
        print(f"Global features: {global_features}")
        print(f"  Available cores ratio: {global_features[0]:.4f}")
        print(f"  Available memory ratio: {global_features[1]:.4f}")

        # Check global features are in valid range [0, 1]
        if not (0.0 <= global_features[0] <= 1.0):
            print(f"❌ Available cores ratio out of range: {global_features[0]}")
            return False

        if not (0.0 <= global_features[1] <= 1.0):
            print(f"❌ Available memory ratio out of range: {global_features[1]}")
            return False

        print(f"\n✓ State format looks correct")

        # Run a few steps and check state updates
        print(f"\nTesting state updates over 5 steps...")
        for i in range(5):
            action = np.random.random(max_buckets).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            global_features = obs[-2:]
            print(f"  Step {i+1}: cores_ratio={global_features[0]:.4f}, mem_ratio={global_features[1]:.4f}, reward={reward:.6f}")

        print(f"\n✅ State format test passed")
        return True

    except Exception as e:
        print(f"\n❌ State format test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_tests():
    """Run comprehensive tests with different configurations."""
    test_configs = [
        {"seed": 42, "num_steps": 50, "num_hosts": 10, "max_buckets": 20, "max_time": 50},
        {"seed": 123, "num_steps": 100, "num_hosts": 20, "max_buckets": 50, "max_time": 100},
        {"seed": 999, "num_steps": 200, "num_hosts": 50, "max_buckets": 100, "max_time": 200},
    ]

    all_passed = True

    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}/{len(test_configs)}")

        passed = test_basic_functionality(**config)

        if passed:
            print(f"✅ TEST {i+1} PASSED")
        else:
            print(f"❌ TEST {i+1} FAILED")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("JobOrderingEnv Test Suite")
    print("=" * 60)
    print("\nMake sure the Rust environment is compiled:")
    print("  cd src/environment")
    print("  maturin develop --release")
    print()

    import time
    time.sleep(1)

    try:
        # Test state format first
        print("\n" + "="*60)
        print("PHASE 1: State Format Test")
        print("="*60)
        state_test_passed = test_state_format(seed=42, num_hosts=10, max_buckets=20)

        if not state_test_passed:
            print("\n⚠️  State format test failed. Please fix before continuing.")
            exit(1)

        # Test basic functionality
        print("\n" + "="*60)
        print("PHASE 2: Basic Functionality Test")
        print("="*60)
        basic_test_passed = test_basic_functionality(
            seed=42,
            num_steps=100,
            num_hosts=30,
            max_buckets=50,
            max_time=100
        )

        if not basic_test_passed:
            print("\n⚠️  Basic functionality test failed. Please fix before continuing.")
            exit(1)

        # Run comprehensive tests
        print("\n" + "="*60)
        print("PHASE 3: Comprehensive Tests")
        print("="*60)
        all_passed = run_comprehensive_tests()

        if all_passed:
            print("\n" + "="*60)
            print("🎉 SUCCESS: All tests passed!")
            print("JobOrderingEnv is working correctly.")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("⚠️  Some tests failed. Please review the output above.")
            print("="*60)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease make sure to compile the Rust environment first:")
        print("  cd src/environment")
        print("  maturin develop --release")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

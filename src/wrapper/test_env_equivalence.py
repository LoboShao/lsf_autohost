#!/usr/bin/env python3
"""
Test script to verify that the original environment (ClusterSchedulerEnv)
and the refactored environment (HostSortingEnv) produce identical behavior
when given the same random seed and actions.
"""

import numpy as np
from gym_wrapper import LsfEnvWrapper
from host_sorting_wrapper import HostSortingEnvWrapper


def compare_environments(seed=42, num_steps=100, num_hosts=50, max_time=100):
    """
    Compare the two environments step by step to verify identical behavior.

    Args:
        seed: Random seed for deterministic testing
        num_steps: Number of steps to test
        num_hosts: Number of hosts in the cluster
        max_time: Maximum time for the episode

    Returns:
        bool: True if environments behave identically, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing environment equivalence with seed={seed}")
    print(f"Configuration: num_hosts={num_hosts}, max_time={max_time}")
    print(f"{'='*60}\n")

    # Create environments with same configuration
    env1 = LsfEnvWrapper(
        num_hosts=num_hosts,
        max_time=max_time,
        seed=seed
    )

    env2 = HostSortingEnvWrapper(
        num_hosts=num_hosts,
        max_time=max_time,
        seed=seed
    )

    # Reset both environments
    obs1, _ = env1.reset(seed=seed)
    obs2, _ = env2.reset(seed=seed)

    # Check initial observations
    print(f"Initial observation shapes: env1={obs1.shape}, env2={obs2.shape}")

    if obs1.shape != obs2.shape:
        print(f"❌ ERROR: Initial observation shapes differ!")
        return False

    initial_diff = np.max(np.abs(obs1 - obs2))
    print(f"Initial observation max difference: {initial_diff:.10f}")

    if initial_diff > 1e-6:
        print(f"❌ ERROR: Initial observations differ by more than 1e-6!")
        print(f"First 20 values of obs1: {obs1[:20]}")
        print(f"First 20 values of obs2: {obs2[:20]}")
        return False

    print(f"✓ Initial observations match!\n")

    # Generate deterministic random actions
    np.random.seed(seed)
    actions = [np.random.random(num_hosts).astype(np.float32) for _ in range(num_steps)]

    # Track metrics
    rewards1, rewards2 = [], []
    dones1, dones2 = [], []
    max_obs_diff = 0.0

    # Step through both environments with identical actions
    print(f"Testing {num_steps} steps...")
    for i in range(num_steps):
        action = actions[i]

        # Step environment 1
        obs1, reward1, done1, truncated1, info1 = env1.step(action)
        rewards1.append(reward1)
        dones1.append(done1)

        # Step environment 2
        obs2, reward2, done2, truncated2, info2 = env2.step(action)
        rewards2.append(reward2)
        dones2.append(done2)

        # Compare observations
        obs_diff = np.max(np.abs(obs1 - obs2))
        max_obs_diff = max(max_obs_diff, obs_diff)

        # Compare rewards
        reward_diff = abs(reward1 - reward2)

        # Compare done flags
        if done1 != done2:
            print(f"❌ Step {i+1}: Done flags differ! env1={done1}, env2={done2}")
            return False

        # Print progress and differences
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}: reward_diff={reward_diff:.10f}, obs_diff={obs_diff:.10f}, done={done1}")

        # Check for significant differences
        if obs_diff > 1e-5:
            print(f"❌ Step {i+1}: Observations differ by {obs_diff:.10f} (> 1e-5)")
            print(f"   First differing values: obs1={obs1[:5]}, obs2={obs2[:5]}")
            return False

        if reward_diff > 1e-5:
            print(f"❌ Step {i+1}: Rewards differ by {reward_diff:.10f} (> 1e-5)")
            print(f"   env1={reward1}, env2={reward2}")
            return False

    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"✓ All {num_steps} steps completed successfully")
    print(f"✓ Maximum observation difference: {max_obs_diff:.10f}")
    print(f"✓ Rewards match: env1_sum={sum(rewards1):.6f}, env2_sum={sum(rewards2):.6f}")
    print(f"✓ Episode completions match: env1={sum(dones1)}, env2={sum(dones2)}")

    # Get final metrics if available
    metrics_match = True
    try:
        metrics1 = env1.get_metrics()
        metrics2 = env2.get_metrics()

        print(f"\nFinal Metrics Comparison:")
        for key in metrics1.keys():
            if key in metrics2:
                val1 = metrics1[key]
                val2 = metrics2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = abs(val1 - val2)
                    # For deterministic tests, we expect EXACT matches
                    if diff > 1e-10:  # Very small tolerance for floating point
                        print(f"  ❌ {key}: env1={val1:.10f}, env2={val2:.10f}, diff={diff:.10f}")
                        metrics_match = False
                    else:
                        print(f"  ✓ {key}: {val1:.6f} (exact match)")
                else:
                    print(f"  {key}: env1={val1}, env2={val2}")
                    if val1 != val2:
                        print(f"     ❌ Non-numeric values differ!")
                        metrics_match = False
    except Exception as e:
        print(f"Could not compare metrics: {e}")
        metrics_match = False

    if not metrics_match:
        print(f"\n❌ ERROR: Metrics don't match! Environments are NOT behaviorally identical!")
        return False

    return True


def run_multiple_tests():
    """Run multiple test configurations to ensure equivalence."""
    test_configs = [
        {"seed": 42, "num_steps": 100, "num_hosts": 10, "max_time": 50},
        {"seed": 123, "num_steps": 100, "num_hosts": 20, "max_time": 50},
        {"seed": 999, "num_steps": 200, "num_hosts": 50, "max_time": 100},
    ]

    all_passed = True

    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}/{len(test_configs)}")

        passed = compare_environments(**config)

        if passed:
            print(f"✅ TEST {i+1} PASSED")
        else:
            print(f"❌ TEST {i+1} FAILED")
            all_passed = False

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")

    if all_passed:
        print("✅ ALL TESTS PASSED - Environments are behaviorally equivalent!")
    else:
        print("❌ SOME TESTS FAILED - Environments have different behavior!")

    return all_passed


if __name__ == "__main__":
    # First compile the Rust environment if needed
    print("Make sure the Rust environment is compiled:")
    print("  cd src/environment")
    print("  maturin develop --release")
    print()

    import time
    time.sleep(2)  # Give user time to see the message

    try:
        # Test single configuration
        single_test_passed = compare_environments(
            seed=42,
            num_steps=1000,
            num_hosts=30,
            max_time=300
        )

        if single_test_passed:
            print("\n" + "="*60)
            print("Running comprehensive tests...")
            print("="*60)
            all_passed = run_multiple_tests()

            if all_passed:
                print("\n🎉 SUCCESS: The refactored environment (HostSortingEnv) is")
                print("behaviorally identical to the original environment (ClusterSchedulerEnv)!")
            else:
                print("\n⚠️  Some differences were detected. Please review the output above.")
        else:
            print("\n⚠️  Basic test failed. Please fix the issues before running comprehensive tests.")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease make sure to compile the Rust environment first:")
        print("  cd src/environment")
        print("  maturin develop --release")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
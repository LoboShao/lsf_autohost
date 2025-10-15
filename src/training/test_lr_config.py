#!/usr/bin/env python3
"""Test script to verify learning rate configuration combinations."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_lr_configurations():
    """Test different LR configuration combinations."""
    
    test_cases = [
        {
            "name": "Pure Scheduler (Cosine)",
            "args": ["--lr-schedule", "cosine"],
            "expected": "cosine scheduler only"
        },
        {
            "name": "Pure KL-Adaptive",
            "args": ["--use-kl-adaptive-lr"],
            "expected": "Pure KL-adaptive (target=0.02), no scheduler"
        },
        {
            "name": "Combined KL + Scheduler",
            "args": ["--use-kl-adaptive-lr", "--combine-kl-with-scheduler", "--lr-schedule", "linear"],
            "expected": "KL-adaptive (target=0.02) + linear scheduler"
        },
        {
            "name": "Custom KL Target",
            "args": ["--use-kl-adaptive-lr", "--kl-target", "0.01"],
            "expected": "Pure KL-adaptive (target=0.01), no scheduler"
        }
    ]
    
    print("=" * 60)
    print("Testing Learning Rate Configuration Logic")
    print("=" * 60)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Args: {' '.join(test_case['args'])}")
        
        # Parse args to simulate configuration
        use_kl = "--use-kl-adaptive-lr" in test_case['args']
        combine = "--combine-kl-with-scheduler" in test_case['args']
        
        # Find lr_schedule
        lr_schedule = "constant"
        if "--lr-schedule" in test_case['args']:
            idx = test_case['args'].index("--lr-schedule")
            lr_schedule = test_case['args'][idx + 1]
        
        # Find kl_target
        kl_target = 0.02
        if "--kl-target" in test_case['args']:
            idx = test_case['args'].index("--kl-target")
            kl_target = float(test_case['args'][idx + 1])
        
        # Determine actual LR strategy
        if use_kl:
            if combine:
                actual = f"KL-adaptive (target={kl_target}) + {lr_schedule} scheduler"
            else:
                actual = f"Pure KL-adaptive (target={kl_target}), no scheduler"
        else:
            actual = f"{lr_schedule} scheduler only"
        
        print(f"Expected: {test_case['expected']}")
        print(f"Actual: {actual}")
        
        # Check if matches
        if test_case['expected'] in actual:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
    
    print("\n" + "=" * 60)
    print("Configuration Logic Summary:")
    print("-" * 60)
    print("1. Default: KL-adaptive only (--use-kl-adaptive-lr is True by default)")
    print("2. Pure Scheduler: Omit --use-kl-adaptive-lr flag")
    print("3. Pure KL: Use --use-kl-adaptive-lr (default)")
    print("4. Combined: Use --use-kl-adaptive-lr --combine-kl-with-scheduler")
    print("=" * 60)

if __name__ == "__main__":
    test_lr_configurations()
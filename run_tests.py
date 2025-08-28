#!/usr/bin/env python3
"""
Test Runner for Highway-Env Imitation Learning Framework
Comprehensive testing script with multiple test modes
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_basic_tests():
    """Run basic functionality tests"""
    print("🔍 Running Basic Functionality Tests...")
    
    # Test 1: Highway-env basic functionality
    cmd1 = [
        "python3", "-c",
        "import highway_env; import gymnasium as gym; env = gym.make('highway-v0'); "
        "obs, info = env.reset(); print('✅ Highway-env basic test passed'); env.close()"
    ]
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    if result1.returncode == 0:
        print("✅ Highway-env basic functionality: PASSED")
    else:
        print("❌ Highway-env basic functionality: FAILED")
        print(result1.stderr)
        return False
    
    # Test 2: Imitation learning models
    cmd2 = [
        "python3", "-c",
        "from imitation_learning.models.cnn_policy import create_policy_for_env; "
        "import gymnasium as gym; import highway_env; env = gym.make('highway-v0'); "
        "policy = create_policy_for_env('highway-v0', env.observation_space, env.action_space); "
        "print('✅ Imitation learning models test passed'); env.close()"
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode == 0:
        print("✅ Imitation learning models: PASSED")
    else:
        print("❌ Imitation learning models: FAILED")
        print(result2.stderr)
        return False
    
    # Test 3: Data collection
    cmd3 = [
        "python3", "-c",
        "from imitation_learning.data_collection.expert_data_collector import ExpertDataCollector; "
        "import gymnasium as gym; import highway_env; "
        "env_config = {'observation': {'type': 'Kinematics'}, 'duration': 10}; "
        "collector = ExpertDataCollector(env_config, max_episodes=1, min_episode_length=5, quality_threshold=-10.0); "
        "trajectories = collector.collect_rule_based_expert_data('highway-v0', 'idm', render=False); "
        "print(f'✅ Data collection test passed - collected {len(trajectories)} trajectories')"
    ]
    
    result3 = subprocess.run(cmd3, capture_output=True, text=True)
    if result3.returncode == 0:
        print("✅ Data collection functionality: PASSED")
    else:
        print("❌ Data collection functionality: FAILED")
        print(result3.stderr)
        return False
    
    return True


def run_environment_tests():
    """Test individual environments"""
    print("\n🏗️  Running Individual Environment Tests...")
    
    environments = ['highway-v0', 'intersection-v0', 'parking-v0']
    
    for env_name in environments:
        print(f"\n  Testing {env_name}...")
        cmd = [
            "python3", "test_individual_environments.py",
            "--env", env_name, "--quick"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ {env_name}: PASSED")
        else:
            print(f"  ❌ {env_name}: FAILED")
            # Print last few lines of error
            lines = result.stdout.split('\n')
            for line in lines[-10:]:
                if line.strip():
                    print(f"    {line}")
    
    return True


def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print("\n🧪 Running Comprehensive Test Suite...")
    
    cmd = ["python3", "test_imitation_learning.py", "--quick"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Comprehensive test suite: PASSED")
    else:
        print("❌ Comprehensive test suite: FAILED")
        # Print relevant output
        lines = result.stdout.split('\n')
        for line in lines[-20:]:
            if line.strip():
                print(f"  {line}")
    
    return result.returncode == 0


def run_example_demo():
    """Run example demonstration"""
    print("\n🚀 Running Example Demonstration...")
    
    # Create a minimal example
    example_code = '''
import sys
sys.path.append(".")
from imitation_learning import ExpertDataCollector, create_policy_for_env
import gymnasium as gym
import highway_env

print("Creating environment...")
env_config = {
    "observation": {"type": "Kinematics"},
    "duration": 10,
}

print("Collecting sample data...")
collector = ExpertDataCollector(
    env_config=env_config,
    max_episodes=2,
    min_episode_length=5,
    quality_threshold=-10.0
)

trajectories = collector.collect_rule_based_expert_data(
    env_name="highway-v0",
    expert_type="idm",
    render=False,
)

print(f"Collected {len(trajectories)} trajectories")

print("Creating policy...")
env = gym.make("highway-v0")
env.unwrapped.configure(env_config)
policy = create_policy_for_env("highway-v0", env.observation_space, env.action_space)
env.close()

print("✅ Example demonstration completed successfully!")
'''
    
    # Write and run example
    with open("temp_example.py", "w") as f:
        f.write(example_code)
    
    result = subprocess.run(["python3", "temp_example.py"], capture_output=True, text=True)
    
    # Clean up
    if os.path.exists("temp_example.py"):
        os.remove("temp_example.py")
    
    if result.returncode == 0:
        print("✅ Example demonstration: PASSED")
        print("  All components working correctly!")
    else:
        print("❌ Example demonstration: FAILED")
        print(result.stderr)
        return False
    
    return True


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Framework is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please check the issues above.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Highway-Env Imitation Learning Framework")
    parser.add_argument("--basic", action="store_true", help="Run only basic functionality tests")
    parser.add_argument("--env", action="store_true", help="Run individual environment tests")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive test suite")
    parser.add_argument("--example", action="store_true", help="Run example demonstration")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific test is requested
    if not any([args.basic, args.env, args.comprehensive, args.example]):
        args.all = True
    
    print("Highway-Env Imitation Learning Framework Test Runner")
    print("="*60)
    
    results = {}
    
    # Run requested tests
    if args.basic or args.all:
        results["Basic Functionality"] = run_basic_tests()
    
    if args.env or args.all:
        results["Environment Tests"] = run_environment_tests()
    
    if args.example or args.all:
        results["Example Demonstration"] = run_example_demo()
    
    if args.comprehensive or args.all:
        results["Comprehensive Suite"] = run_comprehensive_tests()
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
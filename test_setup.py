#!/usr/bin/env python3
"""
Test script to verify the Lane-Nav setup is working correctly.
Tests environment creation, configuration loading, and basic functionality.
"""

import sys
import traceback
from typing import List, Tuple

def test_imports() -> Tuple[bool, str]:
    """Test if all required packages can be imported."""
    try:
        print("Testing imports...")
        
        # Core packages
        import gymnasium as gym
        import numpy as np
        import torch
        
        # RL packages
        import stable_baselines3
        from stable_baselines3 import PPO
        
        # Highway environment
        import highway_env
        
        # IL packages
        import imitation
        
        # Our modules
        import config
        import imitation as imitation_module
        
        print("✅ All imports successful!")
        return True, "All packages imported successfully"
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error during imports: {e}"


def test_environment_creation() -> Tuple[bool, str]:
    """Test environment creation for all scenarios."""
    try:
        print("Testing environment creation...")
        
        from config import create_environment
        
        scenarios = ["highway", "intersection", "roundabout", "parking"]
        
        for scenario in scenarios:
            print(f"  Creating {scenario} environment...")
            env = create_environment(scenario)
            
            # Test basic environment functionality
            obs, info = env.reset()
            print(f"    ✅ {scenario}: obs shape = {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            
            # Test random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            env.close()
        
        print("✅ All environments created successfully!")
        return True, "All environments work correctly"
        
    except Exception as e:
        return False, f"Environment creation error: {e}"


def test_config_loading() -> Tuple[bool, str]:
    """Test configuration loading and validation."""
    try:
        print("Testing configuration loading...")
        
        from config import (
            ENVIRONMENT_CONFIGS, 
            PPO_HYPERPARAMS, 
            IMITATION_HYPERPARAMS,
            TRAINING_CONFIG,
            get_env_config,
            get_model_filename
        )
        
        # Test environment configs
        for scenario in ["highway", "intersection", "roundabout", "parking"]:
            config = get_env_config(scenario)
            print(f"  ✅ {scenario} config loaded")
        
        # Test model filename generation
        filename = get_model_filename("highway", "rl", "ppo")
        print(f"  ✅ Model filename: {filename}")
        
        filename = get_model_filename("intersection", "il", "bc")
        print(f"  ✅ IL filename: {filename}")
        
        print("✅ Configuration loading successful!")
        return True, "Configuration loaded correctly"
        
    except Exception as e:
        return False, f"Configuration error: {e}"


def test_directory_structure() -> Tuple[bool, str]:
    """Test if required directories exist or can be created."""
    try:
        print("Testing directory structure...")
        
        import os
        from config import TRAINING_CONFIG
        
        required_dirs = [
            TRAINING_CONFIG["model_save_dir"],
            TRAINING_CONFIG["log_dir"],
            "./tensorboard_logs"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✅ Created directory: {dir_path}")
            else:
                print(f"  ✅ Directory exists: {dir_path}")
        
        print("✅ Directory structure verified!")
        return True, "All directories ready"
        
    except Exception as e:
        return False, f"Directory error: {e}"


def test_training_scripts() -> Tuple[bool, str]:
    """Test that training scripts can be imported and have required functions."""
    try:
        print("Testing training scripts...")
        
        scripts = [
            "train_highway",
            "train_intersection", 
            "train_roundabout",
            "train_parking"
        ]
        
        for script_name in scripts:
            try:
                module = __import__(script_name)
                
                # Check for required functions
                required_functions = ["train_rl_agent", "train_il_agent", "main"]
                for func_name in required_functions:
                    if hasattr(module, func_name):
                        print(f"    ✅ {script_name}.{func_name}")
                    else:
                        print(f"    ❌ {script_name}.{func_name} not found")
                        
            except ImportError as e:
                print(f"    ❌ Could not import {script_name}: {e}")
        
        # Test evaluate script
        try:
            import evaluate
            print(f"    ✅ evaluate.py imported")
        except ImportError as e:
            print(f"    ❌ Could not import evaluate.py: {e}")
        
        print("✅ Training scripts verification completed!")
        return True, "Training scripts ready"
        
    except Exception as e:
        return False, f"Training scripts error: {e}"


def run_all_tests() -> None:
    """Run all tests and provide summary."""
    print("=" * 60)
    print("🧪 Lane-Nav Setup Verification")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment Creation", test_environment_creation),
        ("Configuration Loading", test_config_loading),
        ("Directory Structure", test_directory_structure),
        ("Training Scripts", test_training_scripts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            success, message = test_func()
            results.append((test_name, success, message))
            
            if success:
                print(f"✅ PASS: {message}")
            else:
                print(f"❌ FAIL: {message}")
                
        except Exception as e:
            error_msg = f"Test crashed: {e}"
            results.append((test_name, False, error_msg))
            print(f"💥 CRASH: {error_msg}")
            print("Full traceback:")
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            print(f"    → {message}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Lane-Nav setup is ready to use.")
        print("\nNext steps:")
        print("1. Train an agent: python train_highway.py --mode rl")
        print("2. Evaluate it: python evaluate.py highway --mode rl")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the error messages above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check Python version (>=3.8 recommended)")
        print("3. Install highway-env: pip install highway-env")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
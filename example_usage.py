#!/usr/bin/env python3
"""
Example usage script demonstrating the Lane-Nav framework.
Shows how to train and evaluate agents programmatically.
"""

import os
import sys
from config import TRAINING_CONFIG

def ensure_directories():
    """Create necessary directories."""
    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["log_dir"], exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)

def example_rl_training():
    """Example: Train an RL agent."""
    print("=" * 60)
    print("🚗 Example: Training PPO agent for highway driving")
    print("=" * 60)
    
    try:
        from train_highway import train_rl_agent
        
        print("Training PPO agent with reduced timesteps for demo...")
        
        # Train with fewer timesteps for quick demo
        model = train_rl_agent(total_timesteps=5000)  # Reduced for demo
        
        print("✅ RL training completed!")
        return True
        
    except Exception as e:
        print(f"❌ RL training failed: {e}")
        return False

def example_evaluation():
    """Example: Evaluate a trained agent."""
    print("\n" + "=" * 60)
    print("🎯 Example: Evaluating trained agent")
    print("=" * 60)
    
    try:
        from evaluate import evaluate_agent
        
        print("Evaluating highway agent (without rendering for demo)...")
        
        results = evaluate_agent(
            scenario="highway",
            mode="rl",
            algorithm="ppo",
            n_episodes=3,
            render=False,  # No rendering for demo
            verbose=True
        )
        
        print(f"✅ Evaluation completed! Mean reward: {results['mean_reward']:.2f}")
        return True
        
    except FileNotFoundError:
        print("❌ No trained model found. Train a model first.")
        return False
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False

def example_il_training():
    """Example: Train an IL agent using BC."""
    print("\n" + "=" * 60)
    print("🎓 Example: Training Behavioral Cloning agent")
    print("=" * 60)
    
    try:
        # Check if we have an RL expert first
        expert_path = os.path.join(
            TRAINING_CONFIG["model_save_dir"], 
            "ppo_highway_agent.zip"
        )
        
        if not os.path.exists(expert_path):
            print("❌ No expert model found. Need to train RL agent first.")
            return False
        
        from train_highway import train_il_agent
        
        print("Training BC agent using expert demonstrations...")
        
        model_path = train_il_agent(
            algorithm="bc",
            expert_model_path=expert_path,
            n_demonstrations=5  # Reduced for demo
        )
        
        print(f"✅ IL training completed! Model saved: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ IL training failed: {e}")
        return False

def example_environment_showcase():
    """Example: Showcase all environments."""
    print("\n" + "=" * 60)
    print("🌍 Example: Environment showcase")
    print("=" * 60)
    
    try:
        from config import create_environment
        
        scenarios = ["highway", "intersection", "roundabout", "parking"]
        
        for scenario in scenarios:
            print(f"\nTesting {scenario} environment...")
            
            env = create_environment(scenario)
            obs, info = env.reset()
            
            print(f"  Observation space: {env.observation_space}")
            print(f"  Action space: {env.action_space}")
            print(f"  Initial observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            
            # Take a few random steps
            for step in range(3):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"  Step {step + 1}: reward = {reward:.3f}")
                
                if terminated or truncated:
                    break
            
            env.close()
            print(f"  ✅ {scenario} environment working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment showcase failed: {e}")
        return False

def main():
    """Main example demonstration."""
    print("🎯 Lane-Nav Framework Example Usage")
    print("This script demonstrates the key features of the framework.")
    print("Note: Training uses reduced parameters for quick demonstration.\n")
    
    # Ensure directories exist
    ensure_directories()
    
    # Run examples
    examples = [
        ("Environment Showcase", example_environment_showcase),
        ("RL Training", example_rl_training),
        ("Agent Evaluation", example_evaluation),
        ("IL Training", example_il_training),
    ]
    
    results = []
    
    for example_name, example_func in examples:
        try:
            success = example_func()
            results.append((example_name, success))
        except KeyboardInterrupt:
            print(f"\n🛑 {example_name} interrupted by user")
            break
        except Exception as e:
            print(f"💥 {example_name} crashed: {e}")
            results.append((example_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Example Results Summary")
    print("=" * 60)
    
    for example_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {example_name}")
    
    successful = sum(success for _, success in results)
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} examples")
    
    if successful == total:
        print("\n🎉 All examples completed successfully!")
        print("\nYou can now use the framework with commands like:")
        print("• python train_highway.py --mode rl")
        print("• python evaluate.py highway --mode rl") 
        print("• python train_intersection.py --mode il")
        print("• python evaluate.py intersection --mode il")
    else:
        print(f"\n⚠️ Some examples failed. Check error messages above.")
    
    print("\nFor more details, see README.md or run:")
    print("• python test_setup.py  # Verify installation")
    print("• python train_highway.py --help  # See training options")
    print("• python evaluate.py --help  # See evaluation options")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Example session interrupted")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
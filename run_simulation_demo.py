#!/usr/bin/env python3
"""
Simulation Demo Script - View Final Simulations for All Environments
Shows how to run and visualize simulations for all highway-env environments
"""

import gymnasium as gym
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import highway_env
from gymnasium.wrappers import RecordVideo


def run_random_policy_demo(env_name: str, episodes: int = 3, record_video: bool = False):
    """
    Run a random policy demo for visualization
    
    Args:
        env_name: Name of the environment
        episodes: Number of episodes to run
        record_video: Whether to record video
    """
    print(f"\n{'='*60}")
    print(f"🚗 Running {env_name} simulation with random policy")
    print(f"{'='*60}")
    
    # Create environment with human rendering
    env = gym.make(env_name, render_mode="human")
    
    # Optional: Record video
    if record_video:
        video_folder = f"./simulation_videos/{env_name}"
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)
        print(f"📹 Recording videos to: {video_folder}")
    
    # Configure environment for better visualization
    if hasattr(env.unwrapped, 'config'):
        env.unwrapped.config.update({
            "simulation_frequency": 15,  # Higher FPS for smoother rendering
            "policy_frequency": 5,       # Agent decision frequency
            "duration": 40,              # Episode length
        })
    
    try:
        for episode in range(episodes):
            print(f"\n🎬 Episode {episode + 1}/{episodes}")
            obs, info = env.reset()
            
            episode_reward = 0
            step = 0
            done = truncated = False
            
            while not (done or truncated):
                # Random action for demonstration
                action = env.action_space.sample()
                
                # Take step
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                # Render the environment
                env.render()
                
                # Small delay for better visualization
                time.sleep(0.05)
                
                # Print some info every 20 steps
                if step % 20 == 0:
                    print(f"  Step {step}: Reward = {episode_reward:.2f}")
            
            print(f"✅ Episode {episode + 1} completed - Total Reward: {episode_reward:.2f}, Steps: {step}")
            
            # Pause between episodes
            if episode < episodes - 1:
                print("⏸️  Press Enter to continue to next episode...")
                input()
    
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped by user")
    
    finally:
        env.close()
        print(f"🏁 {env_name} simulation completed\n")


def run_trained_model_demo(env_name: str, model_path: str = None, episodes: int = 3):
    """
    Run simulation with a trained model (if available)
    
    Args:
        env_name: Name of the environment
        model_path: Path to trained model
        episodes: Number of episodes to run
    """
    print(f"\n{'='*60}")
    print(f"🤖 Running {env_name} with trained model")
    print(f"{'='*60}")
    
    try:
        from stable_baselines3 import DQN, PPO
        
        # Create environment
        env = gym.make(env_name, render_mode="human")
        
        # Try to load a trained model
        if model_path and Path(model_path).exists():
            if "dqn" in model_path.lower():
                model = DQN.load(model_path, env=env)
            elif "ppo" in model_path.lower():
                model = PPO.load(model_path, env=env)
            else:
                print("⚠️  Unknown model type, using random policy")
                run_random_policy_demo(env_name, episodes)
                return
        else:
            print("⚠️  No trained model found, using random policy")
            run_random_policy_demo(env_name, episodes)
            return
        
        # Run episodes with trained model
        for episode in range(episodes):
            print(f"\n🎬 Episode {episode + 1}/{episodes}")
            obs, info = env.reset()
            
            episode_reward = 0
            step = 0
            done = truncated = False
            
            while not (done or truncated):
                # Use trained model
                action, _states = model.predict(obs, deterministic=True)
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                env.render()
                time.sleep(0.05)
                
                if step % 20 == 0:
                    print(f"  Step {step}: Reward = {episode_reward:.2f}")
            
            print(f"✅ Episode {episode + 1} completed - Total Reward: {episode_reward:.2f}, Steps: {step}")
            
            if episode < episodes - 1:
                print("⏸️  Press Enter to continue...")
                input()
        
        env.close()
        
    except ImportError:
        print("⚠️  stable-baselines3 not installed, using random policy")
        run_random_policy_demo(env_name, episodes)


def demo_all_environments():
    """Demo all available environments"""
    
    environments = {
        'highway-v0': {
            'description': 'Multi-lane highway driving',
            'config': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'lanes_count': 4,
                'vehicles_count': 30,
                'duration': 40,
            }
        },
        'highway-fast-v0': {
            'description': 'Highway with fast simulation',
            'config': {
                'lanes_count': 3,
                'vehicles_count': 25,
                'duration': 40,
            }
        },
        'intersection-v0': {
            'description': 'Traffic intersection navigation',
            'config': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'duration': 20,
            }
        },
        'roundabout-v0': {
            'description': 'Roundabout navigation',
            'config': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'duration': 20,
            }
        },
        'parking-v0': {
            'description': 'Parking maneuver',
            'config': {
                'observation': {'type': 'KinematicsGoal'},
                'action': {'type': 'ContinuousAction'},
                'duration': 20,
            }
        },
        'merge-v0': {
            'description': 'Highway merging scenario',
            'config': {
                'observation': {'type': 'Kinematics'},
                'action': {'type': 'DiscreteMetaAction'},
                'duration': 30,
            }
        }
    }
    
    print("🌟 Highway-Env Simulation Demo")
    print("=" * 50)
    print("Available environments:")
    
    for i, (env_name, env_info) in enumerate(environments.items(), 1):
        print(f"{i}. {env_name}: {env_info['description']}")
    
    print(f"{len(environments) + 1}. Run all environments")
    print("0. Exit")
    
    while True:
        try:
            choice = input("\nSelect environment to demo (number): ").strip()
            
            if choice == '0':
                print("👋 Goodbye!")
                break
            elif choice == str(len(environments) + 1):
                # Run all environments
                for env_name, env_info in environments.items():
                    try:
                        print(f"\n🚀 Starting {env_name}...")
                        run_random_policy_demo(env_name, episodes=2)
                    except Exception as e:
                        print(f"❌ Error running {env_name}: {e}")
                        continue
                break
            elif choice.isdigit() and 1 <= int(choice) <= len(environments):
                env_list = list(environments.keys())
                selected_env = env_list[int(choice) - 1]
                print(f"\n🚀 Starting {selected_env}...")
                run_random_policy_demo(selected_env, episodes=3)
                break
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Highway-Env Simulation Demo")
    parser.add_argument('--env', type=str, help='Specific environment to run')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--record', action='store_true', help='Record videos')
    parser.add_argument('--all', action='store_true', help='Demo all environments')
    
    args = parser.parse_args()
    
    if args.all:
        demo_all_environments()
    elif args.env:
        if args.model:
            run_trained_model_demo(args.env, args.model, args.episodes)
        else:
            run_random_policy_demo(args.env, args.episodes, args.record)
    else:
        demo_all_environments()


if __name__ == "__main__":
    main()
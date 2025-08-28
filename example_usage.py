#!/usr/bin/env python3
"""
Example Usage Script for Highway-Env Imitation Learning Framework
Demonstrates the complete pipeline from data collection to RL integration
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

import highway_env
from imitation_learning import (
    ExpertDataCollector,
    ImitationLearningTrainer,
    ImitationLearningEvaluator,
    create_policy_for_env,
    create_warm_start_rl_model,
    train_rl_with_il_warmstart
)


def example_highway_kinematics():
    """
    Example 1: Highway environment with kinematics observations
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Highway Environment with Kinematics")
    print("="*60)
    
    # Environment configuration
    env_config = {
        'observation': {'type': 'Kinematics'},
        'action': {'type': 'DiscreteMetaAction'},
        'duration': 20,  # Shorter for example
        'vehicles_count': 20,
        'collision_reward': -1,
        'right_lane_reward': 0.1,
        'high_speed_reward': 0.4,
    }
    
    # Step 1: Collect expert data
    print("\nStep 1: Collecting expert demonstrations...")
    collector = ExpertDataCollector(
        env_config=env_config,
        save_dir="./example_data/highway_kinematics",
        max_episodes=10,  # Small for example
        min_episode_length=10,
    )
    
    trajectories = collector.collect_rule_based_expert_data(
        env_name='highway-v0',
        expert_type='idm',
        render=False,
    )
    
    collector.print_statistics()
    data_path = collector.save_data('highway_kinematics_expert.pkl')
    
    # Step 2: Create and train imitation learning model
    print("\nStep 2: Training imitation learning model...")
    env = gym.make('highway-v0')
    env.unwrapped.configure(env_config)
    
    policy = create_policy_for_env('highway-v0', env.observation_space, env.action_space)
    print(f"Created policy with {sum(p.numel() for p in policy.parameters())} parameters")
    
    trainer = ImitationLearningTrainer(
        policy=policy,
        num_epochs=20,  # Small for example
        batch_size=32,
        learning_rate=1e-3,
        save_dir="./example_models/highway_kinematics",
    )
    
    train_loader, val_loader = trainer.prepare_data(trajectories)
    training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
    
    model_path = trainer.save_model('highway_kinematics_final.pth')
    rl_weights_path = trainer.export_for_rl('highway_kinematics_rl_weights.pth')
    
    env.close()
    
    # Step 3: Evaluate the trained model
    print("\nStep 3: Evaluating trained model...")
    evaluator = ImitationLearningEvaluator(policy)
    
    eval_results = evaluator.evaluate_single_environment(
        env_name='highway-v0',
        env_config=env_config,
        num_episodes=5,
        deterministic=True,
    )
    
    print(f"IL Model Performance:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2f}")
    print(f"  Collision Rate: {eval_results['collision_rate']:.2f}")
    
    # Step 4: Create RL model with IL warm start
    print("\nStep 4: Creating RL model with IL warm start...")
    try:
        rl_model, env = create_warm_start_rl_model(
            env_name='highway-v0',
            algorithm='DQN',
            il_weights_path=rl_weights_path,
            env_config=env_config,
        )
        
        print("✅ RL model created successfully with IL warm start!")
        
        # Quick test of the RL model
        obs, info = env.reset()
        action, _ = rl_model.predict(obs, deterministic=True)
        print(f"RL model prediction test: action = {action}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ RL integration failed: {e}")
    
    return {
        'data_path': data_path,
        'model_path': model_path,
        'rl_weights_path': rl_weights_path,
        'eval_results': eval_results,
    }


def example_highway_cnn():
    """
    Example 2: Highway environment with CNN observations
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Highway Environment with CNN")
    print("="*60)
    
    # Environment configuration with grayscale observations
    env_config = {
        'observation': {
            'type': 'GrayscaleObservation',
            'observation_shape': (64, 32),  # Smaller for faster example
            'stack_size': 4,
            'weights': [0.2989, 0.5870, 0.1140],
            'scaling': 1.75,
        },
        'action': {'type': 'DiscreteMetaAction'},
        'duration': 15,
        'vehicles_count': 15,
    }
    
    # Step 1: Collect expert data
    print("\nStep 1: Collecting expert demonstrations...")
    collector = ExpertDataCollector(
        env_config=env_config,
        save_dir="./example_data/highway_cnn",
        max_episodes=8,  # Small for example
        min_episode_length=8,
    )
    
    trajectories = collector.collect_rule_based_expert_data(
        env_name='highway-fast-v0',
        expert_type='idm',
        render=False,
    )
    
    collector.print_statistics()
    data_path = collector.save_data('highway_cnn_expert.pkl')
    
    # Step 2: Create and train CNN-based model
    print("\nStep 2: Training CNN-based imitation learning model...")
    env = gym.make('highway-fast-v0')
    env.unwrapped.configure(env_config)
    
    policy = create_policy_for_env('highway-fast-v0', env.observation_space, env.action_space)
    print(f"Created CNN policy with {sum(p.numel() for p in policy.parameters())} parameters")
    
    trainer = ImitationLearningTrainer(
        policy=policy,
        num_epochs=15,  # Small for example
        batch_size=16,  # Smaller for CNN
        learning_rate=1e-3,
        save_dir="./example_models/highway_cnn",
    )
    
    train_loader, val_loader = trainer.prepare_data(trajectories)
    training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
    
    model_path = trainer.save_model('highway_cnn_final.pth')
    rl_weights_path = trainer.export_for_rl('highway_cnn_rl_weights.pth')
    
    env.close()
    
    # Step 3: Evaluate the trained model
    print("\nStep 3: Evaluating trained CNN model...")
    evaluator = ImitationLearningEvaluator(policy)
    
    eval_results = evaluator.evaluate_single_environment(
        env_name='highway-fast-v0',
        env_config=env_config,
        num_episodes=5,
        deterministic=True,
    )
    
    print(f"CNN IL Model Performance:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2f}")
    print(f"  Collision Rate: {eval_results['collision_rate']:.2f}")
    
    return {
        'data_path': data_path,
        'model_path': model_path,
        'rl_weights_path': rl_weights_path,
        'eval_results': eval_results,
    }


def example_intersection():
    """
    Example 3: Intersection environment
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Intersection Environment")
    print("="*60)
    
    # Environment configuration
    env_config = {
        'observation': {'type': 'Kinematics'},
        'action': {'type': 'DiscreteMetaAction'},
        'duration': 15,
        'destination': 'o1',
        'collision_reward': -5,
        'reached_goal_reward': 1,
    }
    
    # Step 1: Collect expert data
    print("\nStep 1: Collecting expert demonstrations for intersection...")
    collector = ExpertDataCollector(
        env_config=env_config,
        save_dir="./example_data/intersection",
        max_episodes=12,
        min_episode_length=5,
    )
    
    trajectories = collector.collect_rule_based_expert_data(
        env_name='intersection-v0',
        expert_type='planned',  # Use planned expert for intersection
        render=False,
    )
    
    collector.print_statistics()
    data_path = collector.save_data('intersection_expert.pkl')
    
    # Step 2: Train model
    print("\nStep 2: Training intersection model...")
    env = gym.make('intersection-v0')
    env.unwrapped.configure(env_config)
    
    policy = create_policy_for_env('intersection-v0', env.observation_space, env.action_space)
    
    trainer = ImitationLearningTrainer(
        policy=policy,
        num_epochs=25,
        batch_size=32,
        learning_rate=1e-3,
        save_dir="./example_models/intersection",
    )
    
    train_loader, val_loader = trainer.prepare_data(trajectories)
    training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
    
    model_path = trainer.save_model('intersection_final.pth')
    rl_weights_path = trainer.export_for_rl('intersection_rl_weights.pth')
    
    env.close()
    
    # Step 3: Evaluate
    print("\nStep 3: Evaluating intersection model...")
    evaluator = ImitationLearningEvaluator(policy)
    
    eval_results = evaluator.evaluate_single_environment(
        env_name='intersection-v0',
        env_config=env_config,
        num_episodes=5,
        deterministic=True,
    )
    
    print(f"Intersection IL Model Performance:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2f}")
    print(f"  Collision Rate: {eval_results['collision_rate']:.2f}")
    
    return {
        'data_path': data_path,
        'model_path': model_path,
        'rl_weights_path': rl_weights_path,
        'eval_results': eval_results,
    }


def example_parking():
    """
    Example 4: Parking environment with continuous control
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Parking Environment (Continuous Control)")
    print("="*60)
    
    # Environment configuration
    env_config = {
        'observation': {'type': 'Kinematics'},
        'action': {'type': 'ContinuousAction'},
        'duration': 20,
        'collision_reward': -5,
        'success_goal_reward': 1,
    }
    
    # Step 1: Collect expert data
    print("\nStep 1: Collecting expert demonstrations for parking...")
    collector = ExpertDataCollector(
        env_config=env_config,
        save_dir="./example_data/parking",
        max_episodes=15,
        min_episode_length=5,
    )
    
    trajectories = collector.collect_rule_based_expert_data(
        env_name='parking-v0',
        expert_type='planned',  # Use planned expert for parking
        render=False,
    )
    
    collector.print_statistics()
    data_path = collector.save_data('parking_expert.pkl')
    
    # Step 2: Train model
    print("\nStep 2: Training parking model...")
    env = gym.make('parking-v0')
    env.unwrapped.configure(env_config)
    
    policy = create_policy_for_env('parking-v0', env.observation_space, env.action_space)
    print(f"Created continuous control policy for parking")
    
    trainer = ImitationLearningTrainer(
        policy=policy,
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-3,
        save_dir="./example_models/parking",
    )
    
    train_loader, val_loader = trainer.prepare_data(trajectories)
    training_history = trainer.train_behavioral_cloning(train_loader, val_loader)
    
    model_path = trainer.save_model('parking_final.pth')
    rl_weights_path = trainer.export_for_rl('parking_rl_weights.pth')
    
    env.close()
    
    # Step 3: Evaluate
    print("\nStep 3: Evaluating parking model...")
    evaluator = ImitationLearningEvaluator(policy)
    
    eval_results = evaluator.evaluate_single_environment(
        env_name='parking-v0',
        env_config=env_config,
        num_episodes=5,
        deterministic=True,
    )
    
    print(f"Parking IL Model Performance:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Success Rate: {eval_results['success_rate']:.2f}")
    print(f"  Collision Rate: {eval_results['collision_rate']:.2f}")
    
    return {
        'data_path': data_path,
        'model_path': model_path,
        'rl_weights_path': rl_weights_path,
        'eval_results': eval_results,
    }


def main():
    """
    Run all examples and provide summary
    """
    print("Highway-Env Imitation Learning Framework")
    print("Example Usage Demonstration")
    print("="*80)
    
    # Create output directories
    Path("./example_data").mkdir(exist_ok=True)
    Path("./example_models").mkdir(exist_ok=True)
    
    # Run examples
    results = {}
    
    try:
        print("\n🚗 Running Highway Examples...")
        results['highway_kinematics'] = example_highway_kinematics()
        results['highway_cnn'] = example_highway_cnn()
        
        print("\n🚦 Running Intersection Example...")
        results['intersection'] = example_intersection()
        
        print("\n🅿️  Running Parking Example...")
        results['parking'] = example_parking()
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY OF ALL EXAMPLES")
        print("="*80)
        
        for example_name, result in results.items():
            if 'eval_results' in result:
                eval_data = result['eval_results']
                print(f"\n{example_name.upper()}:")
                print(f"  Mean Reward: {eval_data['mean_reward']:.2f}")
                print(f"  Success Rate: {eval_data['success_rate']:.2f}")
                print(f"  Collision Rate: {eval_data['collision_rate']:.2f}")
                print(f"  Model saved: {Path(result['model_path']).name}")
        
        print(f"\n✅ All examples completed successfully!")
        print(f"   Models saved in: ./example_models/")
        print(f"   Data saved in: ./example_data/")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
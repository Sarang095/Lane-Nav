#!/usr/bin/env python3
"""
Flexible evaluation script for autonomous driving agents
Supports evaluation of RL (PPO) and IL (BC/DAgger) trained models
across all scenarios: highway, intersection, roundabout, parking
"""

import os
import argparse
import numpy as np
import time
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy

from config import (
    create_environment,
    get_model_filename,
    TRAINING_CONFIG,
    EVALUATION_CONFIG
)
from imitation import load_imitation_model, evaluate_imitation_policy


def load_model(scenario: str, mode: str, algorithm: str = "ppo", model_path: Optional[str] = None) -> BasePolicy:
    """
    Load trained model for evaluation.
    
    Args:
        scenario: Environment scenario ('highway', 'intersection', 'roundabout', 'parking')
        mode: Training mode ('rl' or 'il')
        algorithm: Algorithm used ('ppo', 'bc', 'dagger')
        model_path: Custom path to model file
    
    Returns:
        Loaded model/policy
    """
    if model_path is None:
        model_path = os.path.join(
            TRAINING_CONFIG["model_save_dir"],
            get_model_filename(scenario, mode, algorithm)
        )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    
    if mode == "rl":
        # Load RL model (PPO)
        model = PPO.load(model_path)
        return model
    
    elif mode == "il":
        # Load IL model (BC/DAgger)
        # Create environment to get action/observation spaces
        env = create_environment(scenario)
        model = load_imitation_model(model_path, env)
        return model
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rl' or 'il'")


def evaluate_agent(
    scenario: str,
    mode: str,
    algorithm: str = "ppo",
    model_path: Optional[str] = None,
    n_episodes: int = EVALUATION_CONFIG["n_episodes"],
    render: bool = EVALUATION_CONFIG["render"],
    deterministic: bool = EVALUATION_CONFIG["deterministic"],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate trained agent on specified scenario.
    
    Args:
        scenario: Environment scenario
        mode: Training mode ('rl' or 'il')
        algorithm: Algorithm used
        model_path: Custom path to model file
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        verbose: Whether to print episode details
    
    Returns:
        Evaluation results
    """
    print(f"\n=== Evaluating {mode.upper()} agent on {scenario}-v0 ===")
    
    # Load the trained model
    model = load_model(scenario, mode, algorithm, model_path)
    
    # Create environment for evaluation
    render_mode = "human" if render else None
    env = create_environment(scenario, render_mode=render_mode)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        if verbose:
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print("-" * 40)
        
        step_count = 0
        while not done:
            # Get action from model
            if mode == "rl":
                action, _states = model.predict(obs, deterministic=deterministic)
            else:  # IL mode
                action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            if verbose and step_count % 50 == 0:
                print(f"  Step {step_count}: Reward = {reward:.3f}, Total = {episode_reward:.3f}")
            
            # Add small delay for better visualization when rendering
            if render:
                time.sleep(0.05)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Check for success (scenario-specific)
        episode_success = check_episode_success(scenario, info, episode_reward)
        if episode_success:
            success_count += 1
        
        if verbose:
            print(f"  Final reward: {episode_reward:.3f}")
            print(f"  Episode length: {episode_length} steps")
            print(f"  Success: {'Yes' if episode_success else 'No'}")
    
    # Calculate evaluation metrics
    results = {
        "scenario": scenario,
        "mode": mode,
        "algorithm": algorithm,
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": success_count / n_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }
    
    # Print summary
    print(f"\n=== Evaluation Results ===")
    print(f"Scenario: {scenario}-v0")
    print(f"Mode: {mode.upper()} ({algorithm.upper()})")
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Reward Range: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")
    print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate: {results['success_rate']:.2%} ({success_count}/{n_episodes})")
    
    env.close()
    return results


def check_episode_success(scenario: str, info: Dict[str, Any], final_reward: float) -> bool:
    """
    Check if episode was successful based on scenario-specific criteria.
    
    Args:
        scenario: Environment scenario
        info: Final info dict from environment
        final_reward: Final episode reward
    
    Returns:
        Whether episode was successful
    """
    # Scenario-specific success criteria
    if scenario == "highway":
        # Success if no collision and decent reward
        return final_reward > 20 and not info.get("crashed", False)
    
    elif scenario == "intersection":
        # Success if reached destination without collision
        return info.get("arrived_to_destination", False) and not info.get("crashed", False)
    
    elif scenario == "roundabout":
        # Success if completed roundabout without collision
        return final_reward > 5 and not info.get("crashed", False)
    
    elif scenario == "parking":
        # Success if parked correctly (close to goal)
        return info.get("is_success", False) or final_reward > 0.8
    
    else:
        # Default: success if positive final reward
        return final_reward > 0


def main():
    """Main evaluation function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate autonomous driving agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate.py highway --mode rl
    python evaluate.py intersection --mode il
    python evaluate.py roundabout --mode rl --algorithm ppo --episodes 5
    python evaluate.py parking --mode il --algorithm bc --no-render
        """
    )
    
    parser.add_argument(
        "scenario",
        type=str,
        choices=["highway", "intersection", "roundabout", "parking"],
        help="Environment scenario to evaluate"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rl", "il"],
        default="rl",
        help="Training mode of the model to evaluate"
    )
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "bc", "dagger"],
        default="ppo",
        help="Algorithm used to train the model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom path to model file"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVALUATION_CONFIG["n_episodes"],
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable environment rendering"
    )
    
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Validate algorithm for mode
    if args.mode == "rl" and args.algorithm not in ["ppo"]:
        print(f"Warning: Algorithm '{args.algorithm}' not typically used for RL mode. Using 'ppo'.")
        args.algorithm = "ppo"
    elif args.mode == "il" and args.algorithm not in ["bc", "dagger"]:
        print(f"Warning: Algorithm '{args.algorithm}' not typically used for IL mode. Using 'bc'.")
        args.algorithm = "bc"
    
    try:
        # Run evaluation
        results = evaluate_agent(
            scenario=args.scenario,
            mode=args.mode,
            algorithm=args.algorithm,
            model_path=args.model_path,
            n_episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.non_deterministic,
            verbose=not args.quiet
        )
        
        # Save results if requested
        if args.save_results:
            import json
            with open(args.save_results, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = {k: v.tolist() if isinstance(v, np.ndarray) else 
                              float(v) if isinstance(v, np.floating) else v 
                              for k, v in results.items()}
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to {args.save_results}")
        
        print(f"\nEvaluation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nTo train a model first, run:")
        print(f"python train_{args.scenario}.py --mode {args.mode}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
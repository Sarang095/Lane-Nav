#!/usr/bin/env python3
"""
Training script for parking-v0 autonomous driving agent
Supports both Reinforcement Learning (PPO) and Imitation Learning (BC/DAgger)
"""

import os
import argparse
import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from config import (
    create_environment, 
    PPO_HYPERPARAMS, 
    TRAINING_CONFIG, 
    get_model_filename,
    IMITATION_HYPERPARAMS
)
from imitation import (
    generate_expert_demonstrations,
    train_behavioral_cloning,
    train_dagger,
    save_imitation_model
)


def train_rl_agent(
    total_timesteps: int = TRAINING_CONFIG["total_timesteps"],
    save_path: Optional[str] = None
) -> PPO:
    """
    Train PPO agent for parking-v0 environment.
    
    Args:
        total_timesteps: Total training timesteps
        save_path: Path to save the model
    
    Returns:
        Trained PPO agent
    """
    print("Training PPO agent for parking-v0...")
    
    # Create environment
    env = create_environment("parking")
    env = Monitor(env, TRAINING_CONFIG["log_dir"])
    
    # Create evaluation environment
    eval_env = create_environment("parking")
    eval_env = Monitor(eval_env, TRAINING_CONFIG["log_dir"] + "/eval")
    
    # Initialize PPO agent with MlpPolicy for KinematicsGoal observations
    model = PPO(
        "MlpPolicy",  # Use MLP policy for KinematicsGoal observations
        env,
        **PPO_HYPERPARAMS
    )
    
    # Setup callbacks
    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=TRAINING_CONFIG["model_save_dir"],
        log_path=TRAINING_CONFIG["log_dir"],
        eval_freq=TRAINING_CONFIG["eval_freq"],
        n_eval_episodes=TRAINING_CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG["save_freq"],
        save_path=TRAINING_CONFIG["model_save_dir"],
        name_prefix="parking_checkpoint"
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    if save_path is None:
        save_path = os.path.join(
            TRAINING_CONFIG["model_save_dir"],
            get_model_filename("parking", "rl", "ppo")
        )
    
    model.save(save_path)
    print(f"Model saved to {save_path}")
    
    return model


def train_il_agent(
    algorithm: str = "bc",
    expert_model_path: Optional[str] = None,
    n_demonstrations: int = 20
) -> str:
    """
    Train Imitation Learning agent for parking-v0 environment.
    
    Args:
        algorithm: IL algorithm to use ('bc' or 'dagger')
        expert_model_path: Path to expert model for generating demonstrations
        n_demonstrations: Number of expert demonstrations to collect
    
    Returns:
        Path to saved IL model
    """
    print(f"Training {algorithm.upper()} agent for parking-v0...")
    
    # Load expert model if not provided
    if expert_model_path is None:
        expert_model_path = os.path.join(
            TRAINING_CONFIG["model_save_dir"],
            get_model_filename("parking", "rl", "ppo")
        )
    
    if not os.path.exists(expert_model_path):
        print(f"Expert model not found at {expert_model_path}")
        print("Training RL agent first to use as expert...")
        expert_model = train_rl_agent()
    else:
        expert_model = PPO.load(expert_model_path)
    
    # Create environment for demonstrations
    env = create_environment("parking")
    
    if algorithm.lower() == "bc":
        # Generate expert demonstrations
        demonstrations = generate_expert_demonstrations(
            expert_model, 
            env, 
            n_trajectories=n_demonstrations
        )
        
        # Train BC agent
        trainer = train_behavioral_cloning("parking", demonstrations)
        
    elif algorithm.lower() == "dagger":
        # Train DAgger agent
        trainer = train_dagger("parking", expert_model)
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'bc' or 'dagger'")
    
    # Save trained model
    model_path = save_imitation_model(trainer, "parking", algorithm.lower())
    return model_path


def main():
    """Main training function with command line interface."""
    parser = argparse.ArgumentParser(description="Train autonomous driving agent for parking-v0")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["rl", "il"], 
        default="rl",
        help="Training mode: 'rl' for reinforcement learning (PPO) or 'il' for imitation learning"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "bc", "dagger"],
        default="ppo",
        help="Algorithm to use (ppo for RL, bc/dagger for IL)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TRAINING_CONFIG["total_timesteps"],
        help="Total timesteps for RL training"
    )
    parser.add_argument(
        "--expert-model",
        type=str,
        default=None,
        help="Path to expert model for IL training"
    )
    parser.add_argument(
        "--n-demonstrations",
        type=int,
        default=20,
        help="Number of expert demonstrations for IL training"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Custom save path for the trained model"
    )
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)
    os.makedirs(TRAINING_CONFIG["log_dir"], exist_ok=True)
    
    if args.mode == "rl":
        print("Starting Reinforcement Learning training...")
        model = train_rl_agent(
            total_timesteps=args.timesteps,
            save_path=args.save_path
        )
        print("RL training completed successfully!")
        
    elif args.mode == "il":
        print("Starting Imitation Learning training...")
        model_path = train_il_agent(
            algorithm=args.algorithm if args.algorithm in ["bc", "dagger"] else "bc",
            expert_model_path=args.expert_model,
            n_demonstrations=args.n_demonstrations
        )
        print(f"IL training completed! Model saved to: {model_path}")
    
    print("\nTraining finished! You can now evaluate the agent using:")
    print(f"python evaluate.py parking --mode {args.mode}")


if __name__ == "__main__":
    main()
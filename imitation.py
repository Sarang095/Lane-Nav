"""
Imitation Learning utilities for autonomous driving agents
Supports Behavioral Cloning (BC) and DAgger algorithms
"""

import os
import numpy as np
import torch as th
from typing import Dict, Any, Optional, Tuple, List
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
import gymnasium as gym

from config import IMITATION_HYPERPARAMS, TRAINING_CONFIG, create_environment


class ExpertPolicy:
    """Wrapper for expert policy to generate demonstrations."""
    
    def __init__(self, expert_model_path: str):
        """Initialize expert policy from a trained model."""
        self.policy = load_policy("ppo", expert_model_path)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Predict action for given observation."""
        return self.policy.predict(observation, deterministic=deterministic)


class PPOExpertPolicy:
    """Use a trained PPO model as expert policy."""
    
    def __init__(self, model_path: str):
        """Load trained PPO model."""
        self.model = PPO.load(model_path)
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Predict action for given observation."""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, None


def generate_expert_demonstrations(
    expert_policy: PPO,
    env: gym.Env,
    n_trajectories: int = 10,
    max_episode_steps: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Generate expert demonstrations using a trained policy.
    
    Args:
        expert_policy: Trained expert policy
        env: Environment to collect demonstrations from
        n_trajectories: Number of trajectories to collect
        max_episode_steps: Maximum steps per episode
    
    Returns:
        List of demonstrations
    """
    print(f"Generating {n_trajectories} expert demonstrations...")
    
    # Wrap environment for rollout collection
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    
    # Generate rollouts
    trajectories = rollout.rollout(
        expert_policy,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_trajectories),
        rng=np.random.default_rng(42)
    )
    
    print(f"Collected {len(trajectories)} trajectories with {sum(len(traj.obs) for traj in trajectories)} total steps")
    return trajectories


def train_behavioral_cloning(
    scenario: str,
    expert_demonstrations: List[Dict[str, Any]],
    hyperparams: Optional[Dict[str, Any]] = None
) -> bc.BC:
    """
    Train a Behavioral Cloning agent.
    
    Args:
        scenario: Environment scenario name
        expert_demonstrations: Expert demonstrations
        hyperparams: Training hyperparameters
    
    Returns:
        Trained BC agent
    """
    if hyperparams is None:
        hyperparams = IMITATION_HYPERPARAMS["bc"]
    
    print(f"Training Behavioral Cloning agent for {scenario}...")
    
    # Create environment
    env = create_environment(scenario)
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
    
    # Initialize BC trainer
    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=expert_demonstrations,
        batch_size=hyperparams.get("batch_size", 32),
        optimizer_kwargs={"lr": hyperparams.get("learning_rate", 1e-3)},
        device=hyperparams.get("device", "auto")
    )
    
    # Train the agent
    trainer.train(n_epochs=hyperparams.get("n_epochs", 100))
    
    print("Behavioral Cloning training completed!")
    return trainer


def train_dagger(
    scenario: str,
    expert_policy: PPO,
    hyperparams: Optional[Dict[str, Any]] = None
) -> bc.BC:
    """
    Train a DAgger agent.
    
    Args:
        scenario: Environment scenario name  
        expert_policy: Expert policy for generating labels
        hyperparams: Training hyperparameters
    
    Returns:
        Trained DAgger agent
    """
    if hyperparams is None:
        hyperparams = IMITATION_HYPERPARAMS["dagger"]
    
    print(f"Training DAgger agent for {scenario}...")
    
    # Create environment
    env = create_environment(scenario)
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
    
    # Initialize BC trainer for DAgger
    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=hyperparams.get("batch_size", 32),
        optimizer_kwargs={"lr": hyperparams.get("learning_rate", 1e-3)},
        device=hyperparams.get("device", "auto")
    )
    
    # DAgger training loop
    n_rounds = hyperparams.get("n_rounds", 20)
    n_traj = hyperparams.get("n_traj", 10)
    
    for round_idx in range(n_rounds):
        print(f"DAgger Round {round_idx + 1}/{n_rounds}")
        
        # Generate trajectories with current policy (or expert for first round)
        if round_idx == 0:
            # Use expert policy for initial demonstrations
            current_policy = expert_policy
        else:
            # Use current learner policy
            current_policy = trainer.policy
        
        # Collect rollouts
        trajectories = rollout.rollout(
            current_policy,
            env,
            rollout.make_sample_until(min_episodes=n_traj),
            rng=np.random.default_rng(42 + round_idx)
        )
        
        # Label actions with expert policy
        expert_trajectories = []
        for traj in trajectories:
            expert_actions = []
            for obs in traj.obs:
                action, _ = expert_policy.predict(obs, deterministic=True)
                expert_actions.append(action)
            
            # Create new trajectory with expert actions
            expert_traj = rollout.Trajectory(
                obs=traj.obs,
                acts=np.array(expert_actions),
                infos=traj.infos,
                terminal=traj.terminal
            )
            expert_trajectories.append(expert_traj)
        
        # Add to training dataset
        trainer.set_demonstrations(expert_trajectories)
        
        # Train for a few epochs
        trainer.train(n_epochs=hyperparams.get("n_epochs", 10))
    
    print("DAgger training completed!")
    return trainer


def save_imitation_model(trainer: bc.BC, scenario: str, algorithm: str = "bc") -> str:
    """
    Save trained imitation learning model.
    
    Args:
        trainer: Trained imitation learning agent
        scenario: Environment scenario name
        algorithm: Algorithm used ('bc' or 'dagger')
    
    Returns:
        Path to saved model
    """
    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)
    model_path = os.path.join(
        TRAINING_CONFIG["model_save_dir"],
        f"{algorithm}_{scenario}_agent.zip"
    )
    
    # Save the policy
    trainer.policy.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


def load_imitation_model(model_path: str, env: gym.Env) -> BasePolicy:
    """
    Load trained imitation learning model.
    
    Args:
        model_path: Path to saved model
        env: Environment (for action/observation spaces)
    
    Returns:
        Loaded policy
    """
    try:
        # Try to load as SB3 policy first
        policy = load_policy("ppo", model_path, env)
        return policy
    except Exception as e:
        print(f"Error loading imitation model: {e}")
        raise


def evaluate_imitation_policy(
    policy: BasePolicy,
    env: gym.Env,
    n_episodes: int = 5,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate an imitation learning policy.
    
    Args:
        policy: Policy to evaluate
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
    
    Returns:
        Evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths)
    }
    
    return metrics
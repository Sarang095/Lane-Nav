"""
Evaluation Framework for Imitation Learning Models
Tests trained models on highway-env environments
"""

import gymnasium as gym
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

import highway_env


class ImitationLearningEvaluator:
    """
    Comprehensive evaluation framework for imitation learning models
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "auto",
        render: bool = False,
    ):
        self.model = model
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.render = render
        
        # Evaluation metrics
        self.results = defaultdict(list)
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def evaluate_single_environment(
        self,
        env_name: str,
        env_config: Optional[Dict] = None,
        num_episodes: int = 10,
        max_steps: int = 1000,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a single environment
        
        Args:
            env_name: Name of the highway environment
            env_config: Environment configuration
            num_episodes: Number of episodes to run
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic actions
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating on {env_name} for {num_episodes} episodes...")
        
        # Create environment
        env = gym.make(env_name, render_mode="human" if self.render else None)
        if env_config:
            env.unwrapped.configure(env_config)
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        collision_rate = 0
        episode_details = []
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = truncated = False
            collided = False
            
            episode_actions = []
            episode_observations = []
            
            while not (done or truncated) and episode_length < max_steps:
                # Prepare observation
                obs_tensor = self._prepare_observation(obs)
                
                # Get action from model
                with torch.no_grad():
                    if hasattr(self.model, 'predict'):
                        action = self.model.predict(obs_tensor, deterministic=deterministic)
                    else:
                        action_logits, _ = self.model(obs_tensor)
                        if self.model.is_discrete:
                            action = torch.argmax(action_logits, dim=-1)
                        else:
                            action = torch.tanh(action_logits)
                    
                    action = action.cpu().numpy()
                    if action.ndim > 0:
                        action = action[0] if len(action) == 1 else action
                
                # Store data
                episode_actions.append(action)
                episode_observations.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
                
                # Take action
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Check for collision
                if info.get('crashed', False) or reward < -0.5:  # Assuming negative reward for collision
                    collided = True
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if collided:
                collision_rate += 1
            else:
                success_rate += 1
            
            episode_details.append({
                'episode': episode,
                'reward': episode_reward,
                'length': episode_length,
                'collided': collided,
                'actions': episode_actions,
                'final_info': info,
            })
            
            print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, Collided={collided}")
        
        env.close()
        
        # Calculate metrics
        success_rate = success_rate / num_episodes
        collision_rate = collision_rate / num_episodes
        
        results = {
            'env_name': env_name,
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'success_rate': success_rate,
            'collision_rate': collision_rate,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_details': episode_details,
        }
        
        return results
    
    def evaluate_multiple_environments(
        self,
        env_configs: Dict[str, Dict],
        num_episodes: int = 10,
        max_steps: int = 1000,
        deterministic: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model on multiple environments
        
        Args:
            env_configs: Dictionary mapping env_name to env_config
            num_episodes: Number of episodes per environment
            max_steps: Maximum steps per episode
            deterministic: Whether to use deterministic actions
        
        Returns:
            Dictionary mapping environment names to evaluation results
        """
        all_results = {}
        
        for env_name, env_config in env_configs.items():
            results = self.evaluate_single_environment(
                env_name=env_name,
                env_config=env_config,
                num_episodes=num_episodes,
                max_steps=max_steps,
                deterministic=deterministic,
            )
            all_results[env_name] = results
        
        return all_results
    
    def _prepare_observation(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Prepare observation for model input"""
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.FloatTensor(obs)
        else:
            obs_tensor = obs.float()
        
        # Ensure proper batch dimension
        if obs_tensor.ndim == len(self.model.observation_space.shape):
            obs_tensor = obs_tensor.unsqueeze(0)
        
        return obs_tensor.to(self.device)
    
    def benchmark_against_baselines(
        self,
        env_name: str,
        env_config: Optional[Dict] = None,
        num_episodes: int = 10,
        baselines: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark against baseline policies
        
        Args:
            env_name: Environment name
            env_config: Environment configuration
            num_episodes: Number of episodes
            baselines: List of baseline types ['random', 'idm', 'aggressive']
        
        Returns:
            Comparison results
        """
        if baselines is None:
            baselines = ['random', 'idm']
        
        results = {}
        
        # Evaluate trained model
        print("Evaluating trained imitation learning model...")
        il_results = self.evaluate_single_environment(
            env_name, env_config, num_episodes, deterministic=True
        )
        results['imitation_learning'] = il_results
        
        # Evaluate baselines
        for baseline in baselines:
            print(f"Evaluating {baseline} baseline...")
            baseline_results = self._evaluate_baseline(
                env_name, env_config, baseline, num_episodes
            )
            results[baseline] = baseline_results
        
        return results
    
    def _evaluate_baseline(
        self,
        env_name: str,
        env_config: Optional[Dict],
        baseline_type: str,
        num_episodes: int,
    ) -> Dict[str, Any]:
        """Evaluate baseline policy"""
        env = gym.make(env_name)
        if env_config:
            env.unwrapped.configure(env_config)
        
        episode_rewards = []
        episode_lengths = []
        collision_rate = 0
        
        for episode in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            done = truncated = False
            
            while not (done or truncated) and episode_length < 1000:
                # Get baseline action
                if baseline_type == 'random':
                    action = env.action_space.sample()
                elif baseline_type == 'idm':
                    action = self._get_idm_action(env, obs)
                elif baseline_type == 'aggressive':
                    action = self._get_aggressive_action(env, obs)
                else:
                    action = env.action_space.sample()
                
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if info.get('crashed', False) or reward < -0.5:
                    collision_rate += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        env.close()
        
        return {
            'env_name': env_name,
            'baseline_type': baseline_type,
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'collision_rate': collision_rate / num_episodes,
            'episode_rewards': episode_rewards,
        }
    
    def _get_idm_action(self, env, obs):
        """Simple IDM-like baseline action"""
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
            vehicle = env.unwrapped.vehicle
            if vehicle.speed < 20:
                return 0  # FASTER
            elif vehicle.speed > 25:
                return 1  # SLOWER
            else:
                return 3  # IDLE
        return 3
    
    def _get_aggressive_action(self, env, obs):
        """Aggressive baseline action"""
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
            vehicle = env.unwrapped.vehicle
            if vehicle.speed < 30:
                return 0  # FASTER
            elif np.random.random() < 0.4:
                return np.random.choice([2, 4])  # LANE_RIGHT or LANE_LEFT
            else:
                return 3  # IDLE
        return env.action_space.sample()
    
    def save_results(self, results: Dict, filepath: str):
        """Save evaluation results to file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved evaluation results to {filepath}")
    
    def plot_comparison(self, results: Dict[str, Dict], save_path: Optional[str] = None):
        """Plot comparison of different models/baselines"""
        models = list(results.keys())
        rewards = [results[model]['mean_reward'] for model in models]
        reward_stds = [results[model]['std_reward'] for model in models]
        collision_rates = [results[model].get('collision_rate', 0) for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Rewards comparison
        ax1.bar(models, rewards, yerr=reward_stds, capsize=5)
        ax1.set_title('Mean Episode Reward')
        ax1.set_ylabel('Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Collision rates comparison
        ax2.bar(models, collision_rates)
        ax2.set_title('Collision Rate')
        ax2.set_ylabel('Collision Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()


def create_evaluation_configs() -> Dict[str, Dict]:
    """
    Create standard evaluation configurations for different environments
    """
    configs = {
        'highway-v0': {
            'observation': {'type': 'Kinematics'},
            'action': {'type': 'DiscreteMetaAction'},
            'duration': 40,
        },
        'highway-fast-v0': {
            'observation': {
                'type': 'GrayscaleObservation',
                'observation_shape': (128, 64),
                'stack_size': 4,
                'weights': [0.2989, 0.5870, 0.1140],
                'scaling': 1.75,
            },
            'duration': 40,
        },
        'intersection-v0': {
            'observation': {'type': 'Kinematics'},
            'action': {'type': 'DiscreteMetaAction'},
            'duration': 20,
        },
        'roundabout-v0': {
            'observation': {'type': 'Kinematics'},
            'action': {'type': 'DiscreteMetaAction'},
            'duration': 20,
        },
        'parking-v0': {
            'observation': {'type': 'Kinematics'},
            'action': {'type': 'ContinuousAction'},
            'duration': 20,
        },
    }
    
    return configs


def run_comprehensive_evaluation(
    model: torch.nn.Module,
    save_dir: str = "./evaluation_results",
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation across all highway-env scenarios
    
    Args:
        model: Trained imitation learning model
        save_dir: Directory to save results
    
    Returns:
        Complete evaluation results
    """
    evaluator = ImitationLearningEvaluator(model)
    configs = create_evaluation_configs()
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("Starting comprehensive evaluation...")
    
    # Evaluate on all environments
    all_results = evaluator.evaluate_multiple_environments(
        env_configs=configs,
        num_episodes=20,
        deterministic=True,
    )
    
    # Save results
    results_file = save_path / "comprehensive_evaluation.json"
    evaluator.save_results(all_results, str(results_file))
    
    # Create summary
    summary = {}
    for env_name, results in all_results.items():
        summary[env_name] = {
            'mean_reward': results['mean_reward'],
            'success_rate': results['success_rate'],
            'collision_rate': results['collision_rate'],
        }
    
    print("\n=== Evaluation Summary ===")
    for env_name, metrics in summary.items():
        print(f"{env_name}:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f}")
        print(f"  Success Rate: {metrics['success_rate']:.2f}")
        print(f"  Collision Rate: {metrics['collision_rate']:.2f}")
        print()
    
    return all_results
"""
Expert Data Collection Framework for Imitation Learning
Collects expert demonstrations from highway-env environments
"""

import os
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
import gymnasium as gym
from collections import deque
import time
import json
from pathlib import Path

import highway_env


@dataclass
class Trajectory:
    """Single trajectory data structure"""
    observations: List[np.ndarray]
    actions: List[Union[int, np.ndarray]]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    episode_length: int
    total_reward: float
    env_name: str
    metadata: Dict[str, Any]


class ExpertDataCollector:
    """
    Collects expert demonstrations using various strategies:
    1. Human expert (manual control)
    2. Rule-based expert (IDM, planned trajectories)
    3. Pre-trained RL agent
    4. Optimal control solutions
    """
    
    def __init__(
        self,
        env_config: Dict[str, Any],
        save_dir: str = "./expert_data",
        max_episodes: int = 100,
        min_episode_length: int = 10,
        quality_threshold: float = 0.0,  # Minimum reward threshold
    ):
        self.env_config = env_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_episodes = max_episodes
        self.min_episode_length = min_episode_length
        self.quality_threshold = quality_threshold
        
        self.trajectories: List[Trajectory] = []
        
        # Statistics
        self.collection_stats = {
            "total_episodes": 0,
            "valid_episodes": 0,
            "total_steps": 0,
            "avg_reward": 0.0,
            "collection_time": 0.0,
        }
    
    def create_env(self, env_name: str, render_mode: Optional[str] = None) -> gym.Env:
        """Create highway-env environment with specified configuration"""
        env = gym.make(env_name, render_mode=render_mode)
        
        # Apply custom configuration
        if self.env_config:
            env.unwrapped.configure(self.env_config)
        
        return env
    
    def collect_rule_based_expert_data(
        self,
        env_name: str,
        expert_type: str = "idm",
        render: bool = False,
    ) -> List[Trajectory]:
        """
        Collect expert data using rule-based policies
        
        Args:
            env_name: Name of the highway-env environment
            expert_type: Type of expert ('idm', 'planned', 'aggressive', 'conservative')
            render: Whether to render the environment during collection
        """
        env = self.create_env(env_name, render_mode="human" if render else None)
        
        start_time = time.time()
        episode_count = 0
        
        print(f"Collecting expert data using {expert_type} expert for {env_name}")
        print(f"Target episodes: {self.max_episodes}")
        
        while episode_count < self.max_episodes:
            trajectory = self._collect_single_episode(env, expert_type)
            
            # Quality filtering
            if self._is_valid_trajectory(trajectory):
                self.trajectories.append(trajectory)
                self.collection_stats["valid_episodes"] += 1
                print(f"Episode {episode_count + 1}/{self.max_episodes}: "
                      f"Length={trajectory.episode_length}, "
                      f"Reward={trajectory.total_reward:.2f}")
            else:
                print(f"Episode {episode_count + 1} rejected: "
                      f"Length={trajectory.episode_length}, "
                      f"Reward={trajectory.total_reward:.2f}")
            
            episode_count += 1
            self.collection_stats["total_episodes"] += 1
        
        env.close()
        
        # Update statistics
        self.collection_stats["collection_time"] = time.time() - start_time
        self._update_statistics()
        
        return self.trajectories
    
    def _collect_single_episode(self, env: gym.Env, expert_type: str) -> Trajectory:
        """Collect a single trajectory using specified expert type"""
        observations = []
        actions = []
        rewards = []
        dones = []
        infos = []
        
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0.0
        step_count = 0
        
        while not (done or truncated):
            # Get expert action based on type
            action = self._get_expert_action(env, obs, expert_type)
            
            # Store data
            observations.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
            actions.append(action)
            
            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            
            rewards.append(reward)
            dones.append(done or truncated)
            infos.append(info.copy())
            
            total_reward += reward
            step_count += 1
            obs = next_obs
        
        # Create trajectory
        trajectory = Trajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            infos=infos,
            episode_length=step_count,
            total_reward=total_reward,
            env_name=env.spec.id if env.spec else "unknown",
            metadata={
                "expert_type": expert_type,
                "collection_timestamp": time.time(),
            }
        )
        
        return trajectory
    
    def _get_expert_action(self, env: gym.Env, obs: np.ndarray, expert_type: str) -> Union[int, np.ndarray]:
        """Get action from specified expert type"""
        if expert_type == "idm":
            return self._idm_expert_action(env, obs)
        elif expert_type == "planned":
            return self._planned_expert_action(env, obs)
        elif expert_type == "aggressive":
            return self._aggressive_expert_action(env, obs)
        elif expert_type == "conservative":
            return self._conservative_expert_action(env, obs)
        else:
            # Random fallback
            return env.action_space.sample()
    
    def _idm_expert_action(self, env: gym.Env, obs: np.ndarray) -> Union[int, np.ndarray]:
        """IDM (Intelligent Driver Model) based expert action"""
        # Use simple heuristic-based IDM behavior
        try:
            if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
                vehicle = env.unwrapped.vehicle
                
                # Simple IDM-like behavior based on vehicle state
                current_speed = vehicle.speed if hasattr(vehicle, 'speed') else 20
                target_speed = 25  # Target speed for highway
                
                # Speed control based on current speed
                if current_speed < target_speed * 0.8:
                    return 0  # FASTER
                elif current_speed > target_speed * 1.2:
                    return 1  # SLOWER
                else:
                    # Occasionally change lanes (simple lane changing behavior)
                    if hasattr(vehicle, 'lane_index') and np.random.random() < 0.1:
                        # Try to change lanes occasionally
                        if np.random.random() < 0.5:
                            return 2  # LANE_RIGHT
                        else:
                            return 4  # LANE_LEFT
                    else:
                        return 3  # IDLE (maintain current state)
        except Exception:
            pass
        
        # Fallback to simple heuristic
        return self._simple_heuristic_action(env, obs)
    
    def _planned_expert_action(self, env: gym.Env, obs: np.ndarray) -> Union[int, np.ndarray]:
        """Planned trajectory expert action"""
        # Implement planned behavior (e.g., A* or RRT-based)
        # For now, use a simple goal-directed heuristic
        return self._goal_directed_action(env, obs)
    
    def _aggressive_expert_action(self, env: gym.Env, obs: np.ndarray) -> Union[int, np.ndarray]:
        """Aggressive driving expert action"""
        # Implement aggressive behavior (higher speeds, more lane changes)
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
            vehicle = env.unwrapped.vehicle
            if vehicle.speed < 25:
                return 0  # FASTER
            elif np.random.random() < 0.3:  # 30% chance of lane change
                return np.random.choice([2, 4])  # LANE_RIGHT or LANE_LEFT
            else:
                return 3  # IDLE
        return env.action_space.sample()
    
    def _conservative_expert_action(self, env: gym.Env, obs: np.ndarray) -> Union[int, np.ndarray]:
        """Conservative driving expert action"""
        # Implement conservative behavior (lower speeds, fewer lane changes)
        if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
            vehicle = env.unwrapped.vehicle
            if vehicle.speed > 15:
                return 1  # SLOWER
            elif vehicle.speed < 10:
                return 0  # FASTER
            else:
                return 3  # IDLE
        return 3  # Default to IDLE
    
    def _simple_heuristic_action(self, env: gym.Env, obs: np.ndarray) -> Union[int, np.ndarray]:
        """Simple heuristic-based action"""
        # Basic heuristic based on observation
        if isinstance(env.action_space, gym.spaces.Discrete):
            # For discrete action spaces (usually 0-4: FASTER, SLOWER, LANE_RIGHT, IDLE, LANE_LEFT)
            if hasattr(env.unwrapped, 'vehicle') and env.unwrapped.vehicle:
                vehicle = env.unwrapped.vehicle
                target_speed = 20  # Target speed
                
                if vehicle.speed < target_speed * 0.8:
                    return 0  # FASTER
                elif vehicle.speed > target_speed * 1.2:
                    return 1  # SLOWER
                else:
                    return 3  # IDLE
            return 3  # IDLE as default
        else:
            # For continuous action spaces
            return np.array([0.0, 0.0])  # No acceleration, no steering
    
    def _goal_directed_action(self, env: gym.Env, obs: np.ndarray) -> Union[int, np.ndarray]:
        """Goal-directed action based on environment state"""
        # Implement goal-directed behavior
        return self._simple_heuristic_action(env, obs)
    
    def _is_valid_trajectory(self, trajectory: Trajectory) -> bool:
        """Check if trajectory meets quality criteria"""
        return (
            trajectory.episode_length >= self.min_episode_length and
            trajectory.total_reward >= self.quality_threshold
        )
    
    def _update_statistics(self):
        """Update collection statistics"""
        if self.trajectories:
            self.collection_stats["total_steps"] = sum(t.episode_length for t in self.trajectories)
            self.collection_stats["avg_reward"] = sum(t.total_reward for t in self.trajectories) / len(self.trajectories)
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """Save collected trajectories to disk"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"expert_trajectories_{timestamp}.pkl"
        
        filepath = self.save_dir / filename
        
        # Save trajectories
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectories, f)
        
        # Save statistics
        stats_path = filepath.with_suffix('.json')
        with open(stats_path, 'w') as f:
            json.dump(self.collection_stats, f, indent=2)
        
        print(f"Saved {len(self.trajectories)} trajectories to {filepath}")
        print(f"Saved statistics to {stats_path}")
        
        return str(filepath)
    
    def load_data(self, filepath: str) -> List[Trajectory]:
        """Load trajectories from disk"""
        with open(filepath, 'rb') as f:
            self.trajectories = pickle.load(f)
        
        print(f"Loaded {len(self.trajectories)} trajectories from {filepath}")
        return self.trajectories
    
    def print_statistics(self):
        """Print collection statistics"""
        print("\n=== Expert Data Collection Statistics ===")
        print(f"Total episodes collected: {self.collection_stats['total_episodes']}")
        print(f"Valid episodes: {self.collection_stats['valid_episodes']}")
        print(f"Total steps: {self.collection_stats['total_steps']}")
        print(f"Average reward: {self.collection_stats['avg_reward']:.2f}")
        print(f"Collection time: {self.collection_stats['collection_time']:.2f}s")
        
        if self.trajectories:
            lengths = [t.episode_length for t in self.trajectories]
            rewards = [t.total_reward for t in self.trajectories]
            print(f"Episode length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f}")
            print(f"Episode reward - Min: {min(rewards):.2f}, Max: {max(rewards):.2f}, Avg: {np.mean(rewards):.2f}")


class DatasetCreator:
    """Create ML datasets from collected trajectories"""
    
    def __init__(self, trajectories: List[Trajectory]):
        self.trajectories = trajectories
    
    def create_supervised_dataset(
        self,
        normalize_observations: bool = True,
        augment_data: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create supervised learning dataset (observations -> actions)
        
        Returns:
            observations: Array of observations
            actions: Array of corresponding actions
        """
        all_observations = []
        all_actions = []
        
        for trajectory in self.trajectories:
            # Skip the last observation (no corresponding action)
            for obs, action in zip(trajectory.observations[:-1], trajectory.actions):
                all_observations.append(obs)
                all_actions.append(action)
        
        observations = np.array(all_observations)
        actions = np.array(all_actions)
        
        # Normalize observations if requested
        if normalize_observations:
            observations = self._normalize_observations(observations)
        
        # Data augmentation if requested
        if augment_data:
            observations, actions = self._augment_data(observations, actions)
        
        return observations, actions
    
    def _normalize_observations(self, observations: np.ndarray) -> np.ndarray:
        """Normalize observations to [0, 1] or [-1, 1] range"""
        if observations.dtype == np.uint8:
            # Image observations (0-255) -> (0-1)
            return observations.astype(np.float32) / 255.0
        else:
            # Vector observations -> standardize
            mean = np.mean(observations, axis=0)
            std = np.std(observations, axis=0) + 1e-8
            return (observations - mean) / std
    
    def _augment_data(
        self, 
        observations: np.ndarray, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation techniques"""
        # For image observations, can apply:
        # - Random noise
        # - Slight rotations
        # - Brightness/contrast changes
        
        augmented_obs = [observations]
        augmented_actions = [actions]
        
        # Add noise to observations
        if len(observations.shape) > 2:  # Image observations
            noise_factor = 0.02
            noisy_obs = observations + np.random.normal(0, noise_factor, observations.shape)
            noisy_obs = np.clip(noisy_obs, 0, 1)
            augmented_obs.append(noisy_obs)
            augmented_actions.append(actions)
        
        return np.concatenate(augmented_obs), np.concatenate(augmented_actions)
    
    def create_torch_dataset(
        self,
        normalize_observations: bool = True,
        augment_data: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create PyTorch tensors from trajectories"""
        observations, actions = self.create_supervised_dataset(normalize_observations, augment_data)
        
        obs_tensor = torch.FloatTensor(observations)
        
        # Handle different action types
        if isinstance(actions[0], (int, np.integer)):
            action_tensor = torch.LongTensor(actions)
        else:
            action_tensor = torch.FloatTensor(actions)
        
        return obs_tensor, action_tensor
"""
Configuration file for autonomous driving RL/IL training
Contains shared environment configurations and hyperparameters
"""

import gymnasium as gym
from typing import Dict, Any

# Environment configurations for each scenario
ENVIRONMENT_CONFIGS = {
    "highway": {
        "env_id": "highway-v0",
        "config": {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "flatten": False,
                "normalize": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "duration": 40,  # [s]
            "initial_spacing": 2,
            "collision_reward": -1,
            "reward_speed_range": [20, 30],
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,
            "screen_height": 150,
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": False
        }
    },
    "intersection": {
        "env_id": "intersection-v0",
        "config": {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "flatten": False,
                "normalize": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "duration": 13,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5,
            "collision_reward": -5,
            "high_speed_reward": 1,
            "arrived_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False
        }
    },
    "roundabout": {
        "env_id": "roundabout-v0", 
        "config": {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "flatten": False,
                "normalize": True
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "incoming_vehicle_destination": None,
            "collision_reward": -1,
            "high_speed_reward": 0.2,
            "right_lane_reward": 0.1,
            "on_road_reward": 0.1,
            "duration": 11,
            "normalize_reward": False,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.5],
            "scaling": 5.5
        }
    },
    "parking": {
        "env_id": "parking-v0",
        "config": {
            "observation": {
                "type": "KinematicsGoal",
                "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],
                "scales": [100, 100, 5, 5, 1, 1],
                "normalize": True
            },
            "action": {
                "type": "ContinuousAction"
            },
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
            "steering_range": 0.7,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 100,
            "screen_width": 600,
            "screen_height": 300,
            "centering_position": [0.5, 0.5],
            "scaling": 7
        }
    }
}

# PPO Training hyperparameters
PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "use_sde": False,
    "sde_sample_freq": -1,
    "target_kl": None,
    "tensorboard_log": "./tensorboard_logs/",
    "policy_kwargs": dict(
        features_extractor_class=None,
        features_extractor_kwargs=None,
        normalize_images=True,
        optimizer_class=None,
        optimizer_kwargs=None,
    ),
    "verbose": 1,
    "seed": None,
    "device": "auto",
    "_init_setup_model": True
}

# Imitation Learning hyperparameters
IMITATION_HYPERPARAMS = {
    "bc": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "n_epochs": 100,
        "optimizer_class": "torch.optim.Adam",
        "optimizer_kwargs": {},
        "device": "auto"
    },
    "dagger": {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "n_epochs": 10,
        "beta_schedule": "linear",  # How to mix expert and learner actions
        "n_rounds": 20,  # Number of DAgger rounds
        "n_traj": 10,  # Number of trajectories per round
        "device": "auto"
    }
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps": 100000,
    "eval_freq": 10000,
    "save_freq": 10000,
    "n_eval_episodes": 10,
    "model_save_dir": "./models/",
    "log_dir": "./logs/"
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "n_episodes": 1,
    "render": True,
    "deterministic": True,
    "render_mode": "human"
}

def get_env_config(scenario: str) -> Dict[str, Any]:
    """Get environment configuration for a specific scenario."""
    if scenario not in ENVIRONMENT_CONFIGS:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(ENVIRONMENT_CONFIGS.keys())}")
    return ENVIRONMENT_CONFIGS[scenario]

def create_environment(scenario: str, render_mode: str = None) -> gym.Env:
    """Create and configure environment for a specific scenario."""
    env_config = get_env_config(scenario)
    env = gym.make(env_config["env_id"], render_mode=render_mode)
    env.configure(env_config["config"])
    return env

def get_model_filename(scenario: str, mode: str, algorithm: str = "ppo") -> str:
    """Generate model filename based on scenario, mode, and algorithm."""
    if mode == "rl":
        return f"{algorithm}_{scenario}_agent.zip"
    elif mode == "il":
        return f"bc_{scenario}_agent.zip"  # Default to BC for IL
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rl' or 'il'")
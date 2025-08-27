import os
import argparse
from typing import Tuple, List

import gymnasium as gym
import numpy as np

import highway_env  # noqa: F401


def make_env(env_id: str, use_cnn: bool) -> gym.Env:
    config = {}
    if use_cnn:
        config["observation"] = {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],
            "scaling": 1.75,
        }
    env = gym.make(env_id, config=config)
    env.reset()
    return env


def select_expert_action(env: gym.Env) -> int | np.ndarray:
    """
    Use the built-in high-level heuristic from ControlledVehicle: keep target lane and track speed.
    We let the environment default controller act by returning None for meta-actions.
    For discrete meta-actions envs, we choose IDLE.
    For continuous envs (e.g., parking), return zeros which delegates to low-level control.
    """
    action_space = env.action_space
    if hasattr(action_space, "n"):
        # Discrete: assume index 1 == IDLE per Intersection and Highway DiscreteMetaAction
        return 1
    # Continuous: zero accel/steer
    if hasattr(action_space, "shape"):
        return np.zeros(action_space.shape, dtype=np.float32)
    raise RuntimeError("Unsupported action space")


def run_rollouts(env_id: str, use_cnn: bool, num_episodes: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    env = make_env(env_id, use_cnn)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            act = select_expert_action(env)
            next_obs, reward, done, truncated, info = env.step(act)
            observations.append(obs)
            actions.append(np.array(act))
            obs = next_obs

    obs_arr = np.array(observations)
    act_arr = np.array(actions)
    np.savez_compressed(
        os.path.join(out_dir, f"{env_id.replace('-', '_')}_expert_{'cnn' if use_cnn else 'mlp'}.npz"),
        observations=obs_arr,
        actions=act_arr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True, help="Gym env id (e.g., highway-fast-v0)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--cnn", action="store_true", help="Use GrayscaleObservation for CNN")
    parser.add_argument("--out", default="/workspace/datasets")
    args = parser.parse_args()
    run_rollouts(args.env_id, args.cnn, args.episodes, args.out)


if __name__ == "__main__":
    main()


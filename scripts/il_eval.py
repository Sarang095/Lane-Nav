import os
import argparse

import gymnasium as gym
from stable_baselines3 import DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import highway_env  # noqa: F401


def make_cnn_env(env_id: str):
    def _fn():
        env = gym.make(
            env_id,
            config={
                "observation": {
                    "type": "GrayscaleObservation",
                    "observation_shape": (128, 64),
                    "stack_size": 4,
                    "weights": [0.2989, 0.5870, 0.1140],
                    "scaling": 1.75,
                }
            },
        )
        env.reset()
        return env

    return _fn


def evaluate(env_id: str, algo: str, model_path: str, episodes: int, video_dir: str | None):
    vec_env = DummyVecEnv([make_cnn_env(env_id)])
    if algo.lower() == "dqn":
        model = DQN.load(model_path, env=vec_env)
    elif algo.lower() == "sac":
        model = SAC.load(model_path, env=vec_env)
    else:
        raise ValueError("algo must be 'dqn' or 'sac'")

    total_rewards = []
    for ep in range(episodes):
        obs, info = vec_env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = vec_env.step(action)
            ep_rew += float(reward)
        total_rewards.append(ep_rew)
        print(f"Episode {ep+1}: reward={ep_rew:.2f}")

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        video_length = 2 * vec_env.envs[0].config.get("duration", 30)
        rec = VecVideoRecorder(
            vec_env,
            video_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=f"{algo}-eval",
        )
        obs, info = rec.reset()
        for _ in range(video_length + 1):
            action, _ = model.predict(obs)
            obs, _, _, _, _ = rec.step(action)
        rec.close()

    mean_reward = sum(total_rewards) / max(len(total_rewards), 1)
    print(f"Mean reward over {episodes} episodes: {mean_reward:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", choices=["dqn", "sac"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video-dir", default="", help="Directory to save evaluation video")
    args = parser.parse_args()
    evaluate(args.env_id, args.algo, args.model, args.episodes, args.video_dir or None)


if __name__ == "__main__":
    main()


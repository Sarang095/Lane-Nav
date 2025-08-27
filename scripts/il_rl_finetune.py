import os
import argparse

import gymnasium as gym
import torch

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


def load_fe_weights_into_dqn(model: DQN, fe_path: str) -> None:
    state = torch.load(fe_path, map_location="cpu")
    model.policy.features_extractor.load_state_dict(state)


def load_fe_weights_into_sac(model: SAC, fe_path: str) -> None:
    state = torch.load(fe_path, map_location="cpu")
    model.policy.actor.features_extractor.load_state_dict(state)
    model.policy.critic.features_extractor.load_state_dict(state)


def finetune(env_id: str, algo: str, fe_path: str, total_timesteps: int, logdir: str, video: bool):
    os.makedirs(logdir, exist_ok=True)
    vec_env = DummyVecEnv([make_cnn_env(env_id)])
    if algo.lower() == "dqn":
        model = DQN(
            "CnnPolicy",
            vec_env,
            learning_rate=5e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.2,
            verbose=1,
            tensorboard_log=logdir,
        )
        if fe_path:
            load_fe_weights_into_dqn(model, fe_path)
    elif algo.lower() == "sac":
        model = SAC(
            "CnnPolicy",
            vec_env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128,
            gamma=0.99,
            tau=0.02,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=logdir,
        )
        if fe_path:
            load_fe_weights_into_sac(model, fe_path)
    else:
        raise ValueError("algo must be 'dqn' or 'sac'")

    model.learn(total_timesteps=total_timesteps)
    save_path = os.path.join(logdir, f"{env_id.replace('-', '_')}_{algo}_finetuned")
    model.save(save_path)
    print(f"Saved finetuned model to {save_path}")

    if video:
        rec_env = DummyVecEnv([make_cnn_env(env_id)])
        video_length = 2 * rec_env.envs[0].config.get("duration", 30)
        rec = VecVideoRecorder(
            rec_env,
            os.path.join(logdir, "videos"),
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix=f"{algo}-finetuned",
        )
        obs, info = rec.reset()
        for _ in range(video_length + 1):
            action, _ = model.predict(obs)
            obs, _, _, _, _ = rec.step(action)
        rec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--algo", choices=["dqn", "sac"], required=True)
    parser.add_argument("--fe", default="", help="Path to IL feature extractor .pt")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--logdir", default="/workspace/rl_runs")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    finetune(args.env_id, args.algo, args.fe, args.steps, args.logdir, args.video)


if __name__ == "__main__":
    main()


import os
import argparse
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

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


def train_bc_discrete(env_id: str, dataset_path: str, epochs: int, batch_size: int, lr: float, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    # Build SB3 DQN CnnPolicy to reuse NatureCNN feature extractor
    vec_env = DummyVecEnv([make_cnn_env(env_id)])
    model = DQN(
        "CnnPolicy",
        vec_env,
        learning_rate=1e-4,
        buffer_size=1,
        learning_starts=1,
        batch_size=batch_size,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
    )
    policy = model.policy
    policy.train()

    # Load dataset
    data = np.load(dataset_path)
    obs_np = data["observations"]  # (N, C, W, H)
    acts_np = data["actions"].astype(np.int64).reshape(-1)

    # Torch dataset
    obs_t = torch.from_numpy(obs_np).float() / 255.0
    act_t = torch.from_numpy(acts_np)
    ds = TensorDataset(obs_t, act_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for batch_obs, batch_act in dl:
            optimizer.zero_grad()
            features = policy.extract_features(batch_obs)
            q_values = policy.q_net(features)
            loss = F.cross_entropy(q_values, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)
            total += batch_obs.size(0)
        avg_loss = total_loss / max(total, 1)
        print(f"[BC] Epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

    # Save feature extractor weights for transfer
    fe_path = os.path.join(out_dir, f"{env_id.replace('-', '_')}_bc_cnn_features.pt")
    torch.save(policy.features_extractor.state_dict(), fe_path)
    # Save full policy too
    pol_path = os.path.join(out_dir, f"{env_id.replace('-', '_')}_bc_policy.pt")
    torch.save(policy.state_dict(), pol_path)
    print(f"Saved feature extractor to {fe_path}\nSaved policy to {pol_path}")
    return fe_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="/workspace/il_models")
    args = parser.parse_args()
    fe_path = train_bc_discrete(
        env_id=args.env_id,
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out,
    )
    print(f"Feature extractor saved at: {fe_path}")


if __name__ == "__main__":
    main()


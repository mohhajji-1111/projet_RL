"""Train PPO on SimpleNavEnv with 8 parallel environments."""
import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from .env_wrapper import make_vector_env


def train(total_timesteps: int = 200_000, num_envs: int = 8, seed: int | None = 0, use_subproc: bool = True):
    log_dir = Path("results/ppo_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path("results/ppo_models")
    save_dir.mkdir(parents=True, exist_ok=True)

    vec_env = make_vector_env(num_envs=num_envs, seed=seed, use_subproc=use_subproc)
    vec_env = VecMonitor(vec_env, str(log_dir))

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=128,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        n_epochs=10,
        seed=seed,
    )

    checkpoint_callback = CheckpointCallback(save_freq=50_000 // num_envs, save_path=str(save_dir), name_prefix="ppo_nav")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    final_path = save_dir / "ppo_nav_final"
    model.save(str(final_path))
    vec_env.close()
    print(f"✅ Training finished. Model saved to {final_path}")


if __name__ == "__main__":
    train()

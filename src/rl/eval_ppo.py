"""Evaluate a trained PPO policy on SimpleNavEnv."""
import argparse
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .env_wrapper import make_env


def evaluate(model_path: str, episodes: int = 5, seed: int | None = 42):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Single env for evaluation
    vec_env = DummyVecEnv([make_env(seed)])

    model = PPO.load(model_path, env=vec_env)

    returns = []
    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = vec_env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)
        print(f"Episode {ep+1}: return={ep_ret:.2f}")

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print(f"\n✅ Eval complete over {episodes} episodes: mean={mean_ret:.2f} ± {std_ret:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/ppo_models/ppo_nav_final.zip", help="Path to PPO model zip")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Seed for evaluation env")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.seed)

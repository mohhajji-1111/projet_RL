import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .env import SimpleNavEnv


def make_env(seed: int | None = None):
    def _init():
        env = SimpleNavEnv(seed=seed)
        return env
    return _init


def make_vector_env(num_envs: int = 8, seed: int | None = None, use_subproc: bool = True):
    """Create a vectorized environment (Subproc or Dummy)."""
    env_fns = [make_env(None if seed is None else seed + i) for i in range(num_envs)]
    if use_subproc:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def make_single_env(seed: int | None = None):
    return DummyVecEnv([make_env(seed)])

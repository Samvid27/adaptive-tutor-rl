"""
Unified training script for comparing RL algorithms on the AdaptiveTutorEnv.

Usage:
    python -m training.train_algo ppo
    python -m training.train_algo dqn
    python -m training.train_algo a2c

All three use the same environment, same total timesteps, and the same
evaluation protocol later, so the comparison between them is fair --
differences in final performance reflect the algorithm, not the setup.
"""
import sys
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env

from env.tutor_env import AdaptiveTutorEnv

N_TOPICS = 5
SESSION_LENGTH = 30
TOTAL_TIMESTEPS = 400_000

ALGO_CLASSES = {"ppo": PPO, "dqn": DQN, "a2c": A2C}


def make_env():
    return AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ALGO_CLASSES:
        print(f"Usage: python -m training.train_algo <{'|'.join(ALGO_CLASSES)}>")
        sys.exit(1)

    algo_name = sys.argv[1]
    AlgoClass = ALGO_CLASSES[algo_name]

    if algo_name == "dqn":
        # DQN is off-policy -- a single environment is standard (it learns
        # from a replay buffer of past experience rather than fresh
        # on-policy rollouts, so it doesn't need parallel envs to be
        # sample-efficient the way PPO/A2C do).
        vec_env = make_vec_env(make_env, n_envs=1)
        model = AlgoClass(
            "MlpPolicy", vec_env, verbose=1,
            learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
            batch_size=128, gamma=0.99, train_freq=4, target_update_interval=500,
            seed=0,
        )
    else:
        # PPO and A2C are on-policy -- parallel envs speed up data
        # collection since each update needs fresh rollouts.
        vec_env = make_vec_env(make_env, n_envs=4)
        common_kwargs = dict(verbose=1, learning_rate=3e-4, gamma=0.99, seed=0)
        if algo_name == "ppo":
            model = AlgoClass("MlpPolicy", vec_env, n_steps=256, batch_size=256, **common_kwargs)
        else:  # a2c
            model = AlgoClass("MlpPolicy", vec_env, n_steps=8, **common_kwargs)

    print(f"Training {algo_name.upper()} for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    save_path = f"training/{algo_name}_tutor_agent"
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}.zip")

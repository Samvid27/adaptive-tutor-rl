"""
Train a PPO agent on the AdaptiveTutorEnv.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env.tutor_env import AdaptiveTutorEnv

N_TOPICS = 5
SESSION_LENGTH = 30
TOTAL_TIMESTEPS = 150_000


if __name__ == "__main__":
    # 4 parallel environments speeds up data collection on CPU without
    # needing a GPU -- PPO collects a batch of experience across all of
    # them before each update.
    vec_env = make_vec_env(lambda: AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH), n_envs=4)

    model = PPO(
        "MlpPolicy",       # small feedforward network -- plenty for a 5-dim state
        vec_env,
        verbose=1,
        n_steps=256,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        seed=0,
    )

    print("Starting training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model.save("training/ppo_tutor_agent")
    print("Training complete. Model saved to training/ppo_tutor_agent.zip")

"""
Fair comparison of PPO vs DQN vs A2C, plus the random/heuristic/oracle
baselines, using the same 20 seeds for everyone.
"""
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from env.tutor_env import AdaptiveTutorEnv, N_DIFFICULTY_LEVELS
from env.student import SyntheticStudent

N_EPISODES = 20
SESSION_LENGTH = 30
N_TOPICS = 5


def collect_model(model, seed_offset=0):
    env = AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)
    all_traj = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep + seed_offset)
        traj = [obs.mean()]
        for _ in range(SESSION_LENGTH):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            traj.append(obs.mean())
        all_traj.append(traj)
    return np.array(all_traj)


if __name__ == "__main__":
    results = {}
    for algo_name, AlgoClass in [("ppo", PPO), ("dqn", DQN), ("a2c", A2C)]:
        model = AlgoClass.load(f"training/{algo_name}_tutor_agent")
        traj = collect_model(model)
        results[algo_name] = traj
        np.save(f"training/traj_{algo_name}.npy", traj)
        final = traj[:, -1]
        print(f"{algo_name.upper():5s} -> avg final mastery: {final.mean():.3f} (+/- {final.std():.3f})")

    print("\n--- Compared to existing baselines ---")
    for name in ["random", "heuristic", "oracle"]:
        traj = np.load(f"training/traj_{name}.npy")
        final = traj[:, -1]
        print(f"{name:10s} -> avg final mastery: {final.mean():.3f} (+/- {final.std():.3f})")

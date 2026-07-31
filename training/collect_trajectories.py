"""
Collect mastery-over-time trajectories for all four policies
(random, heuristic, PPO, oracle) using the same seeds, so we can
plot a fair side-by-side comparison of learning curves.
"""
import numpy as np
from stable_baselines3 import PPO
from env.tutor_env import AdaptiveTutorEnv, N_DIFFICULTY_LEVELS
from env.student import SyntheticStudent

N_EPISODES = 20
SESSION_LENGTH = 30
N_TOPICS = 5


def collect_random(seed_offset=0):
    env = AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)
    rng = np.random.default_rng(0)
    all_traj = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep + seed_offset)
        traj = [obs.mean()]
        for _ in range(SESSION_LENGTH):
            action = rng.integers(0, env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(obs.mean())
        all_traj.append(traj)
    return np.array(all_traj)


def collect_heuristic(seed_offset=0):
    env = AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)
    all_traj = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep + seed_offset)
        traj = [obs.mean()]
        for _ in range(SESSION_LENGTH):
            topic = int(np.argmin(obs))
            difficulty_idx = N_DIFFICULTY_LEVELS // 2
            action = topic * N_DIFFICULTY_LEVELS + difficulty_idx
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(obs.mean())
        all_traj.append(traj)
    return np.array(all_traj)


def collect_ppo(seed_offset=0):
    env = AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)
    model = PPO.load("training/ppo_tutor_agent")
    all_traj = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep + seed_offset)
        traj = [obs.mean()]
        for _ in range(SESSION_LENGTH):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(obs.mean())
        all_traj.append(traj)
    return np.array(all_traj)


def collect_oracle(seed_offset=0):
    all_traj = []
    for ep in range(N_EPISODES):
        student = SyntheticStudent(N_TOPICS, seed=ep + seed_offset)
        traj = [student.est_mastery.mean()]
        for _ in range(SESSION_LENGTH):
            topic = int(np.argmin(student.true_ability))
            difficulty = student.true_ability[topic]
            student.attempt(topic, difficulty)
            traj.append(student.est_mastery.mean())
        all_traj.append(traj)
    return np.array(all_traj)


if __name__ == "__main__":
    print("Collecting trajectories for all policies...")
    results = {
        "random": collect_random(),
        "heuristic": collect_heuristic(),
        "ppo": collect_ppo(),
        "oracle": collect_oracle(),
    }
    for name, traj in results.items():
        np.save(f"training/traj_{name}.npy", traj)
        print(f"  {name}: final mastery = {traj[:, -1].mean():.3f}")
    print("Saved all trajectories to training/traj_*.npy")

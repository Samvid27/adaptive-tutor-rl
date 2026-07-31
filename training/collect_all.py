"""
Collect trajectories for ALL policies (random, heuristic, ppo, dqn, a2c,
oracle) on the corrected environment, tracking TRUE ability as the
ground-truth learning metric (not the observable estimate, which is
display-only now -- see student.py for why).
"""
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
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
        traj = [env.student.true_ability.mean()]
        for _ in range(SESSION_LENGTH):
            action = rng.integers(0, env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(env.student.true_ability.mean())
        all_traj.append(traj)
    return np.array(all_traj)


def collect_heuristic(seed_offset=0):
    env = AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)
    all_traj = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep + seed_offset)
        traj = [env.student.true_ability.mean()]
        for _ in range(SESSION_LENGTH):
            topic = int(np.argmin(obs))
            difficulty_idx = N_DIFFICULTY_LEVELS // 2
            action = topic * N_DIFFICULTY_LEVELS + difficulty_idx
            obs, reward, terminated, truncated, info = env.step(action)
            traj.append(env.student.true_ability.mean())
        all_traj.append(traj)
    return np.array(all_traj)


def collect_model(model, seed_offset=0):
    env = AdaptiveTutorEnv(n_topics=N_TOPICS, session_length=SESSION_LENGTH)
    all_traj = []
    for ep in range(N_EPISODES):
        obs, info = env.reset(seed=ep + seed_offset)
        traj = [env.student.true_ability.mean()]
        for _ in range(SESSION_LENGTH):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            traj.append(env.student.true_ability.mean())
        all_traj.append(traj)
    return np.array(all_traj)


def collect_oracle(seed_offset=0):
    all_traj = []
    for ep in range(N_EPISODES):
        student = SyntheticStudent(N_TOPICS, seed=ep + seed_offset)
        traj = [student.true_ability.mean()]
        for _ in range(SESSION_LENGTH):
            topic = int(np.argmin(student.true_ability))
            difficulty = float(student.true_ability[topic])
            student.attempt(topic, difficulty)
            traj.append(student.true_ability.mean())
        all_traj.append(traj)
    return np.array(all_traj)


if __name__ == "__main__":
    results = {
        "random": collect_random(),
        "heuristic": collect_heuristic(),
        "oracle": collect_oracle(),
    }
    for algo_name, AlgoClass in [("ppo", PPO), ("dqn", DQN), ("a2c", A2C)]:
        model = AlgoClass.load(f"training/{algo_name}_tutor_agent")
        results[algo_name] = collect_model(model)

    print(f"{'Policy':12s} {'Final true ability (mean +/- std)'}")
    print("-" * 50)
    for name, traj in results.items():
        np.save(f"training/traj_{name}.npy", traj)
        final = traj[:, -1]
        print(f"{name:12s} {final.mean():.3f} (+/- {final.std():.3f})")

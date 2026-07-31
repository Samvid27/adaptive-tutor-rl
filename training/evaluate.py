"""
Evaluate the trained PPO agent using the same protocol as the baselines
(20 episodes, same seeds) so the comparison is apples-to-apples.
"""
import numpy as np
from stable_baselines3 import PPO
from env.tutor_env import AdaptiveTutorEnv

env = AdaptiveTutorEnv(n_topics=5, session_length=30, seed=0)
model = PPO.load("training/ppo_tutor_agent")

final_masteries = []
mastery_trajectories = []  # track mastery over time within episodes, for plotting later

for ep in range(20):
    obs, info = env.reset(seed=ep)
    traj = [obs.mean()]
    for i in range(30):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        traj.append(obs.mean())
    final_masteries.append(obs.mean())
    mastery_trajectories.append(traj)

print(f"Trained PPO agent -> avg final mastery over 20 episodes: "
      f"{np.mean(final_masteries):.3f} (+/- {np.std(final_masteries):.3f})")

print("\n--- Comparison Summary ---")
print(f"Random policy:                      0.306")
print(f"Weakest-topic + medium difficulty:  0.301")
print(f"Trained PPO agent:                  {np.mean(final_masteries):.3f}")

# Save trajectories for the comparison plot in the next step
np.save("training/ppo_trajectories.npy", np.array(mastery_trajectories))

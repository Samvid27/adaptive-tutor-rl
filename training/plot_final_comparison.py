import numpy as np
import matplotlib.pyplot as plt

policies = {
    "random": ("Random", "#8a8f98", "--"),
    "heuristic": ("Weakest-topic, fixed difficulty", "#e08214", "-."),
    "a2c": ("A2C Agent", "#33a02c", "-"),
    "ppo": ("PPO Agent", "#2166ac", "-"),
    "dqn": ("DQN Agent", "#6a3d9a", "-"),
    "oracle": ("Oracle (upper bound)", "#1a9850", ":"),
}

fig, ax = plt.subplots(figsize=(9, 6))
for key, (label, color, style) in policies.items():
    traj = np.load(f"training/traj_{key}.npy")
    mean_traj = traj.mean(axis=0)
    std_traj = traj.std(axis=0)
    x = np.arange(len(mean_traj))
    lw = 3 if key == "dqn" else 2
    ax.plot(x, mean_traj, label=label, color=color, linestyle=style, linewidth=lw)
    ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj, color=color, alpha=0.08)

ax.set_xlabel("Question number in session", fontsize=12)
ax.set_ylabel("True average student ability (0-1)", fontsize=12)
ax.set_title("Adaptive Tutor: RL Algorithm Comparison\n(ground-truth ability, mean +/- std over 20 simulated students)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)
plt.tight_layout()
plt.savefig("training/final_comparison.png", dpi=150)
print("Saved to training/final_comparison.png")

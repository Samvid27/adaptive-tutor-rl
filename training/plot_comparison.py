import numpy as np
import matplotlib.pyplot as plt

policies = {
    "random": ("Random", "#999999", "--"),
    "heuristic": ("Weakest-topic, fixed difficulty", "#e08214", "-."),
    "ppo": ("Trained RL Agent (PPO)", "#2166ac", "-"),
    "oracle": ("Oracle (perfect info, upper bound)", "#1a9850", ":"),
}

fig, ax = plt.subplots(figsize=(9, 6))

for key, (label, color, style) in policies.items():
    traj = np.load(f"training/traj_{key}.npy")  # shape (n_episodes, session_length+1)
    mean_traj = traj.mean(axis=0)
    std_traj = traj.std(axis=0)
    x = np.arange(len(mean_traj))
    ax.plot(x, mean_traj, label=label, color=color, linestyle=style, linewidth=2.5)
    ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj, color=color, alpha=0.12)

ax.set_xlabel("Question number in session", fontsize=12)
ax.set_ylabel("Average student mastery (0-1)", fontsize=12)
ax.set_title("Adaptive Tutor: Mastery Growth per Teaching Strategy\n(mean +/- std over 20 simulated students)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0.25, 0.55)

plt.tight_layout()
plt.savefig("training/mastery_comparison.png", dpi=150)
print("Saved chart to training/mastery_comparison.png")

"""
Adaptive Tutor Environment
--------------------------
Gymnasium-compatible environment wrapping the SyntheticStudent.

The action/observation space is always structurally sized for
MAX_TOPICS slots (the network's output layer is fixed at training
time), but each episode only a subset are "active" -- the rest are
pinned to fully-mastered (1.0) so the agent has near-zero incentive to
pick them, without needing any special-casing in step().

During TRAINING, the number of active topics is randomized each
episode (1 to MAX_TOPICS) so the agent learns a policy that generalizes
across different real-world topic counts, rather than only ever having
seen exactly one fixed count. For the curated demo/baseline comparisons
(the "Compare All 6" story), n_active_topics is fixed at 5 so those
results stay tied to a specific, repeatable scenario.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.student import SyntheticStudent

MAX_TOPICS = 15
N_DIFFICULTY_LEVELS = 5
DIFFICULTY_VALUES = np.linspace(0.1, 0.9, N_DIFFICULTY_LEVELS)  # 0.1, 0.3, 0.5, 0.7, 0.9


def decode_action(action: int):
    """Turn a flat action index into (topic, difficulty_value). Standalone
    version of AdaptiveTutorEnv._decode_action, for use outside a live
    gym environment -- e.g. driving a trained model against a real
    student's actual answers instead of a simulated one."""
    topic = action // N_DIFFICULTY_LEVELS
    difficulty_idx = action % N_DIFFICULTY_LEVELS
    return int(topic), float(DIFFICULTY_VALUES[difficulty_idx])


class AdaptiveTutorEnv(gym.Env):
    def __init__(self, session_length: int = 30, seed: int = None, n_active_topics: int = None):
        """
        n_active_topics: if given, every episode uses exactly this many
        active topics (used for evaluation/demo consistency). If None,
        each episode randomizes the active count between 1 and
        MAX_TOPICS (used during training, so the policy generalizes).
        """
        super().__init__()
        self.session_length = session_length
        self._seed = seed
        self.n_active_topics_fixed = n_active_topics

        self.action_space = spaces.Discrete(MAX_TOPICS * N_DIFFICULTY_LEVELS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(MAX_TOPICS,), dtype=np.float32
        )

        self.student = None
        self.n_active = None
        self.step_count = 0
        self.step_cost = 0.005
        self.reward_scale = 20.0

    def _decode_action(self, action: int):
        topic = action // N_DIFFICULTY_LEVELS
        difficulty_idx = action % N_DIFFICULTY_LEVELS
        return topic, DIFFICULTY_VALUES[difficulty_idx]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        student_seed = seed if seed is not None else self._seed
        rng = np.random.default_rng(student_seed)

        if self.n_active_topics_fixed is not None:
            self.n_active = self.n_active_topics_fixed
        else:
            self.n_active = int(rng.integers(1, MAX_TOPICS + 1))

        self.student = SyntheticStudent(MAX_TOPICS, seed=student_seed)
        for idx in range(self.n_active, MAX_TOPICS):
            self.student.true_ability[idx] = 1.0
            self.student.est_mastery[idx] = 1.0

        self.step_count = 0
        obs = self.student.get_state().astype(np.float32)
        info = {"n_active": self.n_active}
        return obs, info

    def step(self, action: int):
        topic, difficulty = self._decode_action(action)
        correct, true_gain, est_gain = self.student.attempt(topic, difficulty)

        reward = (true_gain - self.step_cost) * self.reward_scale
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.session_length

        obs = self.student.get_state().astype(np.float32)
        info = {
            "topic": topic,
            "difficulty": difficulty,
            "correct": correct,
            "true_gain": true_gain,
            "est_gain": est_gain,
            "n_active": self.n_active,
        }
        return obs, reward, terminated, truncated, info
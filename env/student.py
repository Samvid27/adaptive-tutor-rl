"""
Synthetic Student Model
-----------------------
Simulates a student's knowledge across multiple topics using an
Item-Response-Theory (IRT) style model.

Each student has a hidden "true ability" per topic (never seen directly
by the agent -- only estimated). When presented with an item of a given
difficulty, the probability of a correct answer follows a logistic
function of (ability - difficulty). After each attempt, we nudge the
student's *true* ability slightly (real learning happens) and update
our *estimated* mastery (what the agent actually observes).
"""

import numpy as np


def estimate_update(current_est: float, difficulty: float, correct: bool) -> float:
    """
    Update a mastery ESTIMATE given an observed right/wrong answer,
    weighted by how informative the question was (how close its
    difficulty is to the current estimate). Standalone so it can be
    reused for a real student (who has no simulated true_ability to
    reference -- only observed answers exist).
    """
    est_gap = current_est - difficulty
    item_information = np.exp(-(est_gap ** 2) / (2 * 0.12 ** 2))
    item_information = max(item_information, 0.02)
    observed = 1.0 if correct else 0.0
    update_rate = 0.3 * item_information
    new_est = current_est + update_rate * (observed - current_est)
    return float(np.clip(new_est, 0.0, 1.0))


class SyntheticStudent:
    def __init__(self, n_topics: int, seed: int = None):
        self.n_topics = n_topics
        self.rng = np.random.default_rng(seed)

        # TRUE ability per topic -- hidden from the agent.
        # Start most students fairly weak (0.1 - 0.4 range), like a
        # student beginning a course.
        self.true_ability = self.rng.uniform(0.1, 0.4, size=n_topics)

        # ESTIMATED mastery per topic -- this is what the agent observes.
        # We start it at a neutral guess since the agent doesn't know
        # the true ability either.
        self.est_mastery = np.full(n_topics, 0.3)

        # How much a single correct/incorrect answer nudges true ability.
        # Small, so learning is gradual and realistic (no one masters a
        # topic in one question).
        self.learning_rate = 0.05

    def answer_probability(self, topic: int, difficulty: float) -> float:
        """P(correct) via logistic function of ability - difficulty."""
        gap = self.true_ability[topic] - difficulty
        return 1 / (1 + np.exp(-6 * gap))  # steepness=6 gives a realistic S-curve

    def attempt(self, topic: int, difficulty: float):
        """
        Student attempts one item of given difficulty on given topic.
        Returns: (correct: bool, true_gain: float, est_gain: float)

        true_gain reflects genuine learning (ground truth, only available
        because this is a simulation) and should be used for the reward
        signal an RL agent trains on. est_gain reflects the change in our
        *observable* estimate and is for display/UI purposes only -- it
        must never be used as the reward, since it is calculated from
        limited information and can be nudged upward by repeatedly asking
        easy, guaranteed-correct questions without real learning taking
        place. Using an observable-but-gameable quantity as the reward
        was tried and confirmed exploitable during development (a trained
        DQN agent learned to farm this estimate with easy questions
        instead of teaching) -- true_gain closes that loophole entirely
        since it is tied to actual simulated ability, not a proxy of it.
        """
        p_correct = self.answer_probability(topic, difficulty)
        correct = self.rng.random() < p_correct

        gap = self.true_ability[topic] - difficulty
        challenge_factor = np.exp(-(gap ** 2) / (2 * 0.12 ** 2))
        challenge_factor = max(challenge_factor, 0.02)

        if correct:
            true_gain = self.learning_rate * challenge_factor
        else:
            true_gain = self.learning_rate * challenge_factor * 0.3

        old_ability = self.true_ability[topic]
        self.true_ability[topic] = min(1.0, self.true_ability[topic] + true_gain)
        true_gain = self.true_ability[topic] - old_ability

        old_est = self.est_mastery[topic]
        self.est_mastery[topic] = estimate_update(self.est_mastery[topic], difficulty, correct)
        est_gain = self.est_mastery[topic] - old_est

        return correct, true_gain, est_gain

    def get_state(self) -> np.ndarray:
        """What the RL agent actually sees: estimated mastery per topic."""
        return self.est_mastery.copy()
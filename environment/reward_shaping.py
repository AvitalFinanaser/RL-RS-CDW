"""
Reward shaping module for collaborative document writing RL environment.
Implements multi-component reward functions.
"""

import numpy as np
from typing import List, Any, Dict
from environment.topic_model import agent_reward_cosine, agent_reward_jsd, compute_community_profile


class RewardShaper:
    """Handles all reward calculations for collaborative document writing environment."""

    def __init__(self, reward_params: Dict[str, Any] = None):
        """Initialize reward shaper with configurable parameters."""

        if reward_params is None:
            self.params = {
                "w1": 0.4,  # coverage weight
                "w2": 0.3,  # completion weight
                "w3": 0.3,  # content weight
                "use_jsd": True  # Use Jensen-Shannon divergence for topic alignment
            }
        else:
            self.params = reward_params

    def calculate_reward(self, agents, stance, doc_topic_matrix) -> float:
        """Compute total reward combining coverage, completion, and content metrics."""

        # Component 1: Coverage - distance between agent and community topic distributions
        coverage_reward = self._coverage_reward(agents, doc_topic_matrix)

        # Component 2: Completion - percentage of filled preferences
        completion_reward = self._completion_reward(stance)

        # Component 3: Content - percentage of up-votes from all paragraphs
        content_reward = self._content_reward(stance)

        # Weighted combination
        total_reward = (
                self.params["w1"] * coverage_reward +
                self.params["w2"] * completion_reward +
                self.params["w3"] * content_reward
        )

        return float(total_reward)

    def _coverage_reward(self, agents, doc_topic_matrix: np.ndarray) -> float:
        """
        Coverage reward: mean topic alignment across all active agents relative to the community topic distribution.
        Uses JSD or cosine similarity between agent profile and community profile.
        """
        # Community profile is mean of all document topic distributions
        community_profile = compute_community_profile(doc_topic_matrix)

        coverage_scores = []
        for agent in agents:
            if agent.topic_profile_vector is None:
                continue
            if self.params["use_jsd"]:
                score = agent_reward_jsd(agent.topic_profile_vector, community_profile)
            else:
                score = agent_reward_cosine(agent.topic_profile_vector, community_profile)
            coverage_scores.append(score)

        return float(np.mean(coverage_scores)) if coverage_scores else 0.0

    def _completion_reward(self, stance) -> float:
        """
        Completion reward: percentage of filled preferences in stance matrix.
        Higher reward as more preferences are elicited.
        """
        known_votes = np.sum(stance.matrix.values != '?')
        total_votes = stance.matrix.size
        return min(float(known_votes / total_votes), 1.0)  # Cap at 1.0

    def _content_reward(self, stance) -> float:
        """
        Content reward: percentage of upvotes (+1) from all votes.
        Encourages positive engagement with content.
        """
        positive_votes = np.sum(stance.matrix.values == '1')
        total_votes = stance.matrix.size
        return min(float(positive_votes / total_votes), 1.0)  # Cap at 1.0

"""
Baseline Models for Collaborative Writing Recommender System.
Implementation of three key baseline models:
1. Random Policy (sanity check)
2. Popularity Policy (with different engagement metrics)
3. Collaborative Filtering Policy (agent-based matrix factorization)

Design choices:
- Collaborative writing requires models that handle sparse preference matrices
- Engagement (even negative) is valuable for group decision-making
- Agent similarity is more meaningful than paragraph similarity in this context
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum

import pandas as pd
from scipy.spatial.distance import cosine
import pickle
np.seterr(divide='ignore', invalid='ignore')


class BaselinePolicy(ABC):
    """Abstract base class for baseline policies"""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def select_action(self, observation: Dict[str, Any], valid_actions: List[int]) -> int:
        """Select action given observation and valid actions"""
        pass

    @abstractmethod
    def reset(self):
        """Reset policy state"""
        pass

    def update(self, observation: Dict[str, Any], action: int, reward: float, next_observation: Dict[str, Any], info: Dict[str, Any] = None):
        """Update policy with interaction data (optional for online baselines)"""
        pass

    def end_of_episode(self):
        """
        Called at end of episode.
        This function allows a policy to perform any batch operations or updates that should only happen after a full episode
        """
        pass


class PopularityMetric(Enum):
    """Different ways to measure paragraph popularity in collaborative writing"""
    POSITIVE_ONLY = "positive_only"  # Count of +1 votes only
    ENGAGEMENT = "engagement"  # Count of any non-"?" votes
    NET_SENTIMENT = "net_sentiment"  # (+1 votes) - (-1 votes)


class RandomPolicy(BaselinePolicy):
    """
    Random baseline - uniformly selects from valid actions.
    Conceptual Role:
    - Sanity check: any learned policy should outperform random
    - Provides lower bound on performance
    - Tests environment implementation correctness
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="Random", seed=seed)

    def select_action(self, observation: Dict[str, Any], valid_actions: List[int]) -> int:
        """Randomly select from valid actions"""
        return self.rng.choice(valid_actions)

    def reset(self):
        """Reset random state"""
        self.rng = np.random.RandomState(self.seed)


class PopularityPolicy(BaselinePolicy):
    """
    Popularity-based baseline - recommends most popular paragraphs.
    Conceptual Design:
    In collaborative writing, "popularity" can be defined several ways:

    1. POSITIVE_ONLY: Only count +1 votes
       - Favors universally liked content
       - May miss engaging but controversial paragraphs

    2. ENGAGEMENT: Count any non-"?" votes (default)
       - Values discussion and attention
       - Controversial content that sparks debate is valuable
       - Matches collaborative writing goals

    3. NET_SENTIMENT: (+1 votes) - (-1 votes)
       - Balances positive and negative feedback
       - May penalize constructive controversy

    4. WEIGHTED_ENGAGEMENT: Weighted sum with different vote values
       - +1 = 1.0, 0 = 0.5, -1 = 0.3 (engagement still valuable)
       - Most nuanced approach
    """

    def __init__(self,
                 metric: PopularityMetric = PopularityMetric.ENGAGEMENT,
                 seed: int = 42):
        super().__init__("Popularity", seed)
        self.metric = metric
        # Track popularity using different metrics
        self.paragraph_votes = defaultdict(lambda: defaultdict(int))  # paragraph_id -> {vote: count}
        self.total_interactions = 0

    def _calculate_popularity_score(self, paragraph_id: int) -> float:
        """Calculate popularity score based on selected metric"""
        votes = self.paragraph_votes[paragraph_id]

        if self.metric == PopularityMetric.POSITIVE_ONLY:
            return votes[1]  # Only +1 votes

        elif self.metric == PopularityMetric.ENGAGEMENT:
            return votes[-1] + votes[0] + votes[1]  # Any engagement

        elif self.metric == PopularityMetric.NET_SENTIMENT:
            return votes[1] - votes[-1]  # Net positive sentiment

        return 0.0

    def select_action(self,
                      observation: Dict[str, Any],
                      valid_actions: List[int]) -> int:
        """Select most popular paragraph from valid actions"""
        if not valid_actions:
            return 0

        # If no popularity data, fall back to random
        if self.total_interactions == 0:
            return self.rng.choice(valid_actions)

        # Calculate popularity scores for valid actions
        action_scores = []
        for action in valid_actions:
            score = self._calculate_popularity_score(action)
            action_scores.append((action, score))

        # Sort by popularity score (descending)
        action_scores.sort(key=lambda x: x[1], reverse=True)

        # Handle ties randomly
        max_score = action_scores[0][1]
        best_actions = [action for action, score in action_scores if score == max_score]

        return self.rng.choice(best_actions)

    def update(self,
               observation: Dict[str, Any],
               action: int, reward: float,
               next_observation: Dict[str, Any],
               info: Dict[str, Any] = None):
        """Update popularity counts based on agent vote from info"""
        if info and "vote" in info:
            vote = int(info["vote"])
            self.paragraph_votes[action][vote] += 1
            self.total_interactions += 1

    def reset(self):
        """Reset popularity counts"""
        self.paragraph_votes = defaultdict(lambda: defaultdict(int))
        self.total_interactions = 0

    def get_popularity_stats(self) -> Dict[str, Any]:
        """Get detailed popularity statistics for analysis"""
        stats = {}
        for paragraph_id, votes in self.paragraph_votes.items():
            stats[paragraph_id] = {
                'votes': dict(votes),
                'total_engagement': votes[-1] + votes[0] + votes[1],
                'net_sentiment': votes[1] - votes[-1],
                'current_score': self._calculate_popularity_score(paragraph_id)
            }
        return stats


class CollaborativeFilteringPolicy(BaselinePolicy):
    """
    Collaborative Filtering baseline using agent-based matrix factorization.
    CF (Collaborative Filtering) exploits the idea that “agents with similar preferences will like similar paragraphs.”
    In collaborative writing, agent-based similarity is particularly meaningful: “people like you approved these paragraphs.”

    Conceptual Design:
    In collaborative writing, we use Agent-based CF because:

    1. Agent Similarity is Meaningful:
       - Writers with similar tastes/styles want similar content
       - More interpretable than paragraph similarity
       - "Writers like you found this paragraph valuable"

    2. Matrix Factorization Approach:
       - Stance matrix: Agents × Paragraphs with values {-1, 0, 1, "?"}
       - Learn latent factors representing writing preferences/themes
       - Handle sparse "?" values by excluding from loss
       - Predict missing preferences using learned factors

    3. Recommendation Strategy:
       - Find agents similar to current agent in latent space
       - Predict current agent's preferences for unrated paragraphs
       - Recommend paragraphs with highest predicted preference

    4. Cold Start Handling:
       - New agents: Use average preferences until enough data
       - New paragraphs: Use global statistics
    """

    def __init__(self, n_factors: int = 10, learning_rate: float = 0.01,
                 regularization: float = 0.1, similarity_threshold: float = 0.1,
                 min_interactions: int = 5, seed: int = 42):
        super().__init__("CollaborativeFiltering", seed)

        # Model hyperparameters
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.similarity_threshold = similarity_threshold
        self.min_interactions = min_interactions

        # Model parameters (initialized when dimensions known)
        self.agent_factors = None  # Agent latent factors
        self.paragraph_factors = None  # Paragraph latent factors
        self.agent_biases = None  # Agent-specific biases
        self.paragraph_biases = None  # Paragraph-specific biases
        self.global_mean = 0.0  # Global average preference

        # Data tracking
        self.stance_matrix = None  # Current stance matrix
        self.interactions = []  # List of (agent_id, paragraph_id, vote)
        self.agent_similarities = {}  # Cache of agent similarities
        self.n_agents = 0
        self.n_paragraphs = 0

        # Training control
        self.last_training_size = 0
        self.training_frequency = 10

    def _initialize_model(self, n_agents: int, n_paragraphs: int):
        """Initialize model parameters"""
        self.n_agents = n_agents
        self.n_paragraphs = n_paragraphs

        # Initialize latent factors with small random values
        self.agent_factors = self.rng.normal(0, 0.1, (n_agents, self.n_factors))
        self.paragraph_factors = self.rng.normal(0, 0.1, (n_paragraphs, self.n_factors))

        # Initialize biases
        self.agent_biases = np.zeros(n_agents)
        self.paragraph_biases = np.zeros(n_paragraphs)

        # Initialize stance matrix
        self.stance_matrix = np.full((n_agents, n_paragraphs), np.nan)

    def _extract_stance_data(self, observation: Dict[str, Any]):
        """Extract current stance matrix from observation"""
        stance_matrix = observation.get('stance_matrix', None)
        if stance_matrix is not None:
            # Convert -2 (unknown) to NaN for our processing
            stance_matrix = stance_matrix.copy()
            stance_matrix[stance_matrix == -2] = np.nan
            self.stance_matrix = stance_matrix

    def _predict_preference(self, agent_id: int, paragraph_id: int) -> float:
        """Predict agent's preference for paragraph using matrix factorization"""
        if self.agent_factors is None:
            return 0.0

        prediction = (self.global_mean +
                      self.agent_biases[agent_id] +
                      self.paragraph_biases[paragraph_id] +
                      np.dot(self.agent_factors[agent_id], self.paragraph_factors[paragraph_id]))

        # Clip to valid preference range
        return np.clip(prediction, -1, 1)

    def _calculate_agent_similarity(self, agent1_id: int, agent2_id: int) -> float:
        """Calculate similarity between two agents based on common votes"""
        if self.stance_matrix is None:
            return 0.0

        agent1_votes = self.stance_matrix[agent1_id]
        agent2_votes = self.stance_matrix[agent2_id]

        # Find common voted paragraphs (non-NaN values)
        mask = ~(np.isnan(agent1_votes) | np.isnan(agent2_votes))

        if np.sum(mask) < 2:  # Need at least 2 common votes
            return 0.0

        common_votes1 = agent1_votes[mask]
        common_votes2 = agent2_votes[mask]

        # Calculate cosine similarity
        try:
            similarity = 1 - cosine(common_votes1, common_votes2)
            return max(0, similarity)  # Only positive similarities
        except:
            return 0.0

    def _find_similar_agents(self, agent_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar agents to given agent"""
        if self.stance_matrix is None:
            return []

        similarities = []
        for other_agent_id in range(self.n_agents):
            if other_agent_id != agent_id:
                sim = self._calculate_agent_similarity(agent_id, other_agent_id)
                if sim > self.similarity_threshold:
                    similarities.append((other_agent_id, sim))

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _fit_matrix_factorization(self, n_epochs: int = 20):
        """Train matrix factorization model using SGD"""
        if len(self.interactions) < self.min_interactions:
            return

        # Update global mean
        votes = [vote for _, _, vote in self.interactions]
        self.global_mean = np.mean(votes)

        # SGD training
        for epoch in range(n_epochs):
            self.rng.shuffle(self.interactions)

            for agent_id, paragraph_id, vote in self.interactions:
                # Predict and calculate error
                pred = self._predict_preference(agent_id, paragraph_id)
                error = vote - pred

                # Store old factors for update
                agent_factors_old = self.agent_factors[agent_id].copy()

                # Update agent factors
                self.agent_factors[agent_id] += self.learning_rate * (
                        error * self.paragraph_factors[paragraph_id] -
                        self.regularization * self.agent_factors[agent_id]
                )

                # Update paragraph factors
                self.paragraph_factors[paragraph_id] += self.learning_rate * (
                        error * agent_factors_old -
                        self.regularization * self.paragraph_factors[paragraph_id]
                )

                # Update biases
                self.agent_biases[agent_id] += self.learning_rate * (
                        error - self.regularization * self.agent_biases[agent_id]
                )

                self.paragraph_biases[paragraph_id] += self.learning_rate * (
                        error - self.regularization * self.paragraph_biases[paragraph_id]
                )

    def select_action(self, observation: Dict[str, Any], valid_actions: List[int]) -> int:
        """Select action based on predicted preferences and agent similarity"""
        if not valid_actions:
            return 0

        current_agent_id = observation.get('current_agent_id', 0)
        self._extract_stance_data(observation)

        # Initialize model if first time
        if self.agent_factors is None:
            stance_matrix = observation.get('stance_matrix', np.array([[0]]))
            n_agents, n_paragraphs = stance_matrix.shape
            self._initialize_model(n_agents, n_paragraphs)

        # If insufficient data, fall back to random
        if len(self.interactions) < self.min_interactions:
            return self.rng.choice(valid_actions)

        # Method 1: Matrix Factorization Predictions
        mf_predictions = []
        for action in valid_actions:
            pred = self._predict_preference(current_agent_id, action)
            mf_predictions.append((action, pred))

        # Method 2: Collaborative Filtering with Similar Agents
        similar_agents = self._find_similar_agents(current_agent_id)
        cf_predictions = []

        if similar_agents and self.stance_matrix is not None:
            for action in valid_actions:
                # Calculate weighted average preference from similar agents
                weighted_sum = 0.0
                total_weight = 0.0

                for similar_agent_id, similarity in similar_agents:
                    if not np.isnan(self.stance_matrix[similar_agent_id, action]):
                        vote = self.stance_matrix[similar_agent_id, action]
                        weighted_sum += similarity * vote
                        total_weight += similarity

                if total_weight > 0:
                    avg_preference = weighted_sum / total_weight
                    cf_predictions.append((action, avg_preference))
                else:
                    cf_predictions.append((action, 0.0))
        else:
            # No similar agents, use MF predictions
            cf_predictions = mf_predictions

        # Combine predictions (average of MF and CF)
        final_predictions = []
        for i, action in enumerate(valid_actions):
            mf_score = mf_predictions[i][1]
            cf_score = cf_predictions[i][1] if cf_predictions else mf_score
            combined_score = (mf_score + cf_score) / 2
            final_predictions.append((action, combined_score))

        # Select action with highest predicted preference
        final_predictions.sort(key=lambda x: x[1], reverse=True)

        # Handle ties randomly
        max_score = final_predictions[0][1]
        best_actions = [action for action, score in final_predictions if abs(score - max_score) < 1e-6]

        return self.rng.choice(best_actions)

    def update(self,
               observation: Dict[str, Any],
               action: int, reward: float,
               next_observation: Dict[str, Any],
               info: Dict[str, Any] = None):
        """Update model with new interaction"""
        if not info or "vote" not in info:
            return

        current_agent_id = observation.get('current_agent_id', 0)
        vote = int(info["vote"])

        # Store interaction
        self.interactions.append((current_agent_id, action, vote))

        # Update stance matrix
        self._extract_stance_data(next_observation)

        # Retrain model periodically
        if len(self.interactions) - self.last_training_size >= self.training_frequency:
            self._fit_matrix_factorization()
            self.last_training_size = len(self.interactions)

    def reset(self):
        """Reset model state"""
        self.agent_factors = None
        self.paragraph_factors = None
        self.agent_biases = None
        self.paragraph_biases = None
        self.global_mean = 0.0
        self.stance_matrix = None
        self.interactions = []
        self.agent_similarities = {}
        self.last_training_size = 0

    def end_of_episode(self):
        """Final model training at episode end"""
        if len(self.interactions) > self.last_training_size:
            self._fit_matrix_factorization(n_epochs=50)  # More thorough training

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics for analysis"""
        if self.agent_factors is None:
            return {}

        return {
            'n_interactions': len(self.interactions),
            'global_mean': self.global_mean,
            'agent_factor_norm': np.linalg.norm(self.agent_factors),
            'paragraph_factor_norm': np.linalg.norm(self.paragraph_factors),
            'model_initialized': self.agent_factors is not None
        }


class BaselineRunner:
    """Runner for evaluating baseline policies with detailed metrics"""
    def __init__(self, env, policies: List[BaselinePolicy]):
        self.env = env
        self.policies = policies
        self.results = {}

    def run_episode(self,
                    policy: BaselinePolicy,
                    max_steps: int = 1000,
                    verbose: bool = False) -> Dict[str, Any]:
        """Run single episode with given policy"""
        # Reset environment and policy
        obs, _ = self.env.reset()
        policy.reset()

        # Reset episode variables
        total_reward = 0.0
        step_count = 0
        episode_rewards = []
        engagement_count = 0  # Count of non-neutral votes

        if verbose:
            print(f"Starting episode with {policy.name} policy")

        # Step the env until max steps or termination
        while step_count < max_steps:

            # 1 - Asking the policy for an action (paragraph) for the current agent

            # 1.1 - Get valid actions from current agent
            current_agent_id = obs["current_agent_id"]
            if current_agent_id < 0:  # Episode terminated
                break

            # Convert index back to agent ID
            current_agent_id = self.env.agents[current_agent_id].agent_id
            valid_actions = self.env.get_valid_actions(current_agent_id)
            if not valid_actions:
                break

            # 1.2 - Select action from policy
            action = policy.select_action(observation=obs, valid_actions=valid_actions)
            assert action in valid_actions, f"Policy selected invalid action: {action} not in {valid_actions}"

            # 2 - The action is executed in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # 3 - The policy is updated with the interaction result
            policy.update(observation=obs, action=action, reward=reward, next_observation=next_obs, info=info)

            # 4 - Record metrics
            total_reward += reward
            episode_rewards.append(reward)
            if abs(reward) > 0.1:  # Non-neutral vote
                engagement_count += 1
            step_count += 1

            obs = next_obs

            # 5 - If the episode is done (terminated/truncated), the loop ends
            if terminated or truncated:
                break

        # End of episode processing
        policy.end_of_episode()

        # Episode metrics recorded
        return {
            'total_reward': total_reward,
            'episode_length': step_count,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'completion_rate': obs.get('stance_completion_rate', np.array([0.0]))[0],
            'engagement_rate': engagement_count / step_count if step_count > 0 else 0.0,
            'episode_rewards': episode_rewards,
            'final_active_agents': len(obs.get('active_agents', [])) if isinstance(obs.get('active_agents'),
                                                                                   list) else np.sum(
                obs.get('active_agents', []))
        }

    def evaluate_policies(self,
                          n_episodes: int = 10,
                          max_steps: int = 1000,
                          verbose: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all policies for multiple episodes.
        For each policy, multiple episodes are run and per-episode metrics are aggregated.
        """
        results = {}

        for policy in self.policies:
            print(f"Evaluating {policy.name} baseline...")

            episode_results = []
            for episode in range(n_episodes):
                episode_result = self.run_episode(policy, max_steps, verbose)
                episode_results.append(episode_result)

                if (episode + 1) % 5 == 0:
                    print(f"  Episode {episode + 1}/{n_episodes} completed")

            # Aggregate results
            results[policy.name] = {
                'mean_total_reward': np.mean([r['total_reward'] for r in episode_results]),
                'std_total_reward': np.std([r['total_reward'] for r in episode_results]),
                'mean_episode_length': np.mean([r['episode_length'] for r in episode_results]),
                'mean_completion_rate': np.mean([r['completion_rate'] for r in episode_results]),
                'mean_engagement_rate': np.mean([r['engagement_rate'] for r in episode_results]),
                'episodes': episode_results
            }

            print(f"{policy.name} Results:")
            print(
                f"Mean Reward: {results[policy.name]['mean_total_reward']:.2f} (±{results[policy.name]['std_total_reward']:.2f})")
            print(f"Mean Episode Length: {results[policy.name]['mean_episode_length']:.1f}")
            print(f"Completion Rate: {results[policy.name]['mean_completion_rate']:.2%}")
            print(f"Engagement Rate: {results[policy.name]['mean_engagement_rate']:.2%}")

        return results

    @staticmethod
    def save_results(results: Dict[str, Dict[str, float]],
                     filepath: str):
        """Save results to file"""
        # with open(filepath, 'wb') as f:
        #     pickle.dump(results, f)
        # print(f"Results saved to {filepath}")
        records = []
        for policy_name, policy_results in results.items():
            for idx, episode_result in enumerate(policy_results['episodes']):
                records.append({
                    'policy': policy_name,
                    'episode': idx,
                    **episode_result
                })
        df = pd.DataFrame(records)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Results saved to {filepath}")

    @staticmethod
    def create_comparison_report(results: Dict[str, Dict[str, float]]) -> str:
        """Create detailed comparison report"""
        report = "Collaborative Writing Baseline Comparison Report\n"
        report += "=" * 60 + "\n\n"

        # Summary table
        report += "Performance Summary:\n"
        headers = ['Policy', 'Mean Reward', 'Std Reward', 'Episode Length', 'Completion Rate', 'Engagement Rate']
        col_widths = [20, 12, 12, 15, 15, 15]

        header_line = ""
        for header, width in zip(headers, col_widths):
            header_line += f"{header:<{width}}"
        report += header_line + "\n"
        report += "-" * sum(col_widths) + "\n"

        for policy_name, policy_results in results.items():
            line = f"{policy_name:<20}"
            line += f"{policy_results['mean_total_reward']:<12.2f}"
            line += f"{policy_results['std_total_reward']:<12.2f}"
            line += f"{policy_results['mean_episode_length']:<15.1f}"
            line += f"{policy_results['mean_completion_rate']:<15.2%}"
            line += f"{policy_results['mean_engagement_rate']:<15.2%}"
            report += line + "\n"

        report += "\n"

        # Best performing analysis
        best_reward = max(results.items(), key=lambda x: x[1]['mean_total_reward'])
        best_completion = max(results.items(), key=lambda x: x[1]['mean_completion_rate'])
        best_engagement = max(results.items(), key=lambda x: x[1]['mean_engagement_rate'])

        report += "Best Performers:\n"
        report += f"  Highest Mean Reward: {best_reward[0]} ({best_reward[1]['mean_total_reward']:.2f})\n"
        report += f"  Highest Completion Rate: {best_completion[0]} ({best_completion[1]['mean_completion_rate']:.2%})\n"
        report += f"  Highest Engagement Rate: {best_engagement[0]} ({best_engagement[1]['mean_engagement_rate']:.2%})\n"

        return report

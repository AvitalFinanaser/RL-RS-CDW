import argparse
import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import itertools

# Import your environment and components
from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader
from rl_algorithms.dqn_train import MaskableDQNPolicy, CustomFeaturesExtractor, ActionMaskWrapper


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DQN on Collaborative Document Environment')
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved DQN model")
    parser.add_argument("--data-base-path", type=str, required=True,
                        help="Base path to dataset directories")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results/",
                        help="Directory to save evaluation results")
    parser.add_argument("--n-eval-episodes", type=int, default=20,
                        help="Number of evaluation episodes per configuration")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random seeds to test")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for model")
    parser.add_argument("--render", action="store_true", help="Enable rendering during evaluation")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    return parser.parse_args()


class EnvironmentConfigurator:
    """Manages different environment configurations for evaluation"""

    def __init__(self, base_path: str):
        self.base_path = base_path

        # Define configuration variations
        self.completion_rates = [0.3, 0.7]  # Low and high sparsity
        self.community_sizes = ["small", "medium"]  # Community size variations
        self.paragraph_counts = ["small", "medium"]  # Problem complexity

        # Generate all configuration combinations
        self.configurations = list(itertools.product(
            self.completion_rates,
            self.community_sizes,
            self.paragraph_counts
        ))

    def get_env_config_path(self, completion_rate: float, community_size: str, paragraph_count: str) -> str:
        """Map configuration parameters to actual dataset paths"""
        # Adjust this mapping based on your dataset structure
        config_name = f"completion_{completion_rate}_community_{community_size}_paragraphs_{paragraph_count}"
        return os.path.join(self.base_path, config_name)

    def create_environment(self, completion_rate: float, community_size: str,
                           paragraph_count: str, seed: int = 42, render_mode: str = None) -> CollaborativeDocRecEnv:
        """Create environment with specific configuration"""
        config_path = self.get_env_config_path(completion_rate, community_size, paragraph_count)

        # Load environment components
        paragraphs_loader = ParagraphsLoader(filepath=config_path)
        agents_loader = AgentsLoader(filepath=config_path)
        events_loader = EventsLoader(filepath=config_path)

        # Create environment
        env = CollaborativeDocRecEnv(
            paragraphs_loader=paragraphs_loader,
            agents_loader=agents_loader,
            events_loader=events_loader,
            render_mode=render_mode or 'human',
            render_csv_name=f"eval_{completion_rate}_{community_size}_{paragraph_count}_seed{seed}.csv",
            seed=seed
        )

        # Wrap environment
        env = ActionMaskWrapper(env)
        return env


class EvaluationMetrics:
    """Collects and analyzes evaluation metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.completion_rates = []
        self.final_stance_matrices = []

    def add_episode(self, reward: float, length: int, success: bool,
                    completion_rate: float, stance_matrix: np.ndarray = None):
        """Add metrics from single episode"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(success)
        self.completion_rates.append(completion_rate)
        if stance_matrix is not None:
            self.final_stance_matrices.append(stance_matrix.copy())

    def get_summary(self) -> Dict:
        """Calculate summary statistics"""
        return {
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'success_rate': np.mean(self.success_rates),
            'mean_completion_rate': np.mean(self.completion_rates),
            'std_completion_rate': np.std(self.completion_rates),
            'episodes_evaluated': len(self.episode_rewards)
        }


def evaluate_single_configuration(model: DQN, config: Tuple[float, str, str],
                                  configurator: EnvironmentConfigurator,
                                  n_episodes: int, n_seeds: int,
                                  render: bool = False) -> Dict:
    """Evaluate model on single environment configuration"""
    completion_rate, community_size, paragraph_count = config
    config_name = f"comp{completion_rate}_comm{community_size}_para{paragraph_count}"

    print(f"\nEvaluating configuration: {config_name}")

    all_metrics = EvaluationMetrics()
    seed_results = []

    for seed in range(n_seeds):
        print(f"  Seed {seed}/{n_seeds}")

        # Create environment for this seed
        env = configurator.create_environment(
            completion_rate, community_size, paragraph_count,
            seed=seed, render_mode='csv' if render else None
        )
        env = Monitor(env)

        # Run evaluation episodes
        episode_rewards, episode_lengths = evaluate_policy(
            model, env, n_eval_episodes=n_episodes,
            render=render, return_episode_rewards=True,
            deterministic=True
        )

        # Collect additional metrics
        seed_metrics = EvaluationMetrics()
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
            # Reset environment to get final state
            obs, _ = env.reset(seed=seed + i * 100)

            # Run single episode to completion to get final metrics
            done = False
            total_reward = 0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward_step, terminated, truncated, info = env.step(action)
                total_reward += reward_step
                steps += 1
                done = terminated or truncated

            # Extract final metrics
            final_completion = env.env._get_stance_completion_rate()
            success = final_completion > 0.95  # Consider >95% completion as success

            seed_metrics.add_episode(total_reward, steps, success, final_completion)
            all_metrics.add_episode(total_reward, steps, success, final_completion)

        seed_results.append({
            'seed': seed,
            'metrics': seed_metrics.get_summary()
        })

        env.close()

    return {
        'configuration': config_name,
        'config_params': {
            'completion_rate': completion_rate,
            'community_size': community_size,
            'paragraph_count': paragraph_count
        },
        'overall_metrics': all_metrics.get_summary(),
        'seed_results': seed_results
    }


def run_comprehensive_evaluation(args):
    """Run evaluation across all configurations"""

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load trained model
    print(f"Loading model from: {args.model_path}")
    model = DQN.load(args.model_path, device=args.device)

    # Initialize configurator
    configurator = EnvironmentConfigurator(args.data_base_path)

    # Run evaluation for each configuration
    all_results = []

    for i, config in enumerate(configurator.configurations):
        print(f"\n{'=' * 60}")
        print(f"Configuration {i + 1}/{len(configurator.configurations)}")
        print(f"{'=' * 60}")

        try:
            result = evaluate_single_configuration(
                model, config, configurator,
                args.n_eval_episodes, args.n_seeds, args.render
            )
            all_results.append(result)

            # Save intermediate results
            with open(os.path.join(args.output_dir, f"results_{i + 1}.json"), 'w') as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"Error evaluating configuration {config}: {e}")
            continue

    # Save complete results
    final_results = {
        'model_path': args.model_path,
        'evaluation_params': {
            'n_eval_episodes': args.n_eval_episodes,
            'n_seeds': args.n_seeds,
            'device': args.device
        },
        'configurations_tested': len(all_results),
        'results': all_results
    }

    with open(os.path.join(args.output_dir, "complete_evaluation.json"), 'w') as f:
        json.dump(final_results, f, indent=2)

    # Generate summary report
    generate_summary_report(final_results, args.output_dir)

    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")
    return final_results


def generate_summary_report(results: Dict, output_dir: str):
    """Generate summary report with key findings"""

    # Extract key metrics for comparison
    summary_data = []
    for result in results['results']:
        config = result['config_params']
        metrics = result['overall_metrics']

        summary_data.append({
            'Configuration': result['configuration'],
            'Completion Rate': config['completion_rate'],
            'Community Size': config['community_size'],
            'Paragraph Count': config['paragraph_count'],
            'Mean Reward': metrics['mean_reward'],
            'Std Reward': metrics['std_reward'],
            'Success Rate': metrics['success_rate'],
            'Mean Episode Length': metrics['mean_length'],
            'Final Completion Rate': metrics['mean_completion_rate'],
            'Episodes Evaluated': metrics['episodes_evaluated']
        })

    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, "evaluation_summary.csv"), index=False)

    # Generate insights
    insights = []

    # Best performing configuration
    best_config = df.loc[df['Mean Reward'].idxmax()]
    insights.append(f"Best performing configuration: {best_config['Configuration']}")
    insights.append(f"  - Mean Reward: {best_config['Mean Reward']:.3f}")
    insights.append(f"  - Success Rate: {best_config['Success Rate']:.3f}")

    # Effect of sparsity
    low_sparsity = df[df['Completion Rate'] == 0.2]['Mean Reward'].mean()
    high_sparsity = df[df['Completion Rate'] == 0.8]['Mean Reward'].mean()
    insights.append(f"\nSparsity Effect:")
    insights.append(f"  - Low sparsity (20%): {low_sparsity:.3f}")
    insights.append(f"  - High sparsity (80%): {high_sparsity:.3f}")

    # Effect of community size
    small_comm = df[df['Community Size'] == 'small']['Mean Reward'].mean()
    medium_comm = df[df['Community Size'] == 'medium']['Mean Reward'].mean()
    insights.append(f"\nCommunity Size Effect:")
    insights.append(f"  - Small community: {small_comm:.3f}")
    insights.append(f"  - Medium community: {medium_comm:.3f}")

    # Save insights
    with open(os.path.join(output_dir, "evaluation_insights.txt"), 'w') as f:
        f.write("Evaluation Insights\n")
        f.write("==================\n\n")
        f.write("\n".join(insights))

    print("\nKey Insights:")
    print("\n".join(insights))


if __name__ == "__main__":
    args = parse_args()
    results = run_comprehensive_evaluation(args)
    print("Evaluation pipeline completed successfully!")
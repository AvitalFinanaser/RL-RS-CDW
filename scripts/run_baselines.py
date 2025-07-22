import os
import sys
from typing import List, Optional
import pandas as pd
from pathlib import Path
import warnings

# Suppress cosine similarity RuntimeWarnings globally for this script
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")

sys.path.append(str(Path(__file__).parent.parent))

from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader
from baselines.baseline_models import BaselinePolicy, RandomPolicy, PopularityPolicy, CollaborativeFilteringPolicy, \
    BaselineRunner, PopularityMetric

# ========== CONFIGURATION ==========

DEFAULT_FILE_PATH = "C:/path/to/your/data"
DEFAULT_RENDER_MODE = 'csv'
DEFAULT_RENDER_PATH = './results'
DEFAULT_CSV_NAME = 'baseline_results.csv'
DEFAULT_DETAILED_CSV = 'detailed_baseline_results.csv'
DEFAULT_N_EPISODES = 20
DEFAULT_MAX_STEPS = None  # None means auto-compute as agents*paragraphs
DEFAULT_SEED = 42
DEFAULT_POLICIES = [
    # Instantiate with DEFAULT_SEED at runtime if you want to use the same seed everywhere
    lambda seed: RandomPolicy(seed=seed),
    lambda seed: PopularityPolicy(metric=PopularityMetric.ENGAGEMENT, seed=seed),
    lambda seed: CollaborativeFilteringPolicy(n_factors=10, learning_rate=0.01, regularization=0.1, seed=seed)
]
DEFAULT_POLICY_NAMES = ['random', 'popularity', 'cf']


def create_environment(
        file_path=DEFAULT_FILE_PATH,
        render_mode=DEFAULT_RENDER_MODE,
        render_path=DEFAULT_RENDER_PATH,
        csv_name=DEFAULT_CSV_NAME,
        seed=DEFAULT_SEED):
    # Loaders
    paragraphs_loader = ParagraphsLoader(file_path)
    agents_loader = AgentsLoader(file_path)
    events_loader = EventsLoader(file_path)
    # Env
    env = CollaborativeDocRecEnv(
        paragraphs_loader=paragraphs_loader,
        agents_loader=agents_loader,
        events_loader=events_loader,
        render_mode=render_mode,
        render_path=render_path,
        render_csv_name=csv_name,
        seed=seed
    )
    return env


def create_policies(seed=DEFAULT_SEED):
    return [policy_ctor(seed) for policy_ctor in DEFAULT_POLICIES]


def create_detailed_csv_report(
        results,
        output_path: str = os.path.join(DEFAULT_RENDER_PATH, DEFAULT_DETAILED_CSV)):
    records = []
    for policy_name, policy_results in results.items():
        for episode_idx, episode_result in enumerate(policy_results['episodes']):
            record = {
                'policy': policy_name,
                'episode': episode_idx,
                'total_reward': episode_result['total_reward'],
                'episode_length': episode_result['episode_length'],
                'mean_reward': episode_result['mean_reward'],
                'completion_rate': episode_result['completion_rate'],
                'engagement_rate': episode_result['engagement_rate']
            }
            records.append(record)
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Detailed results saved to {output_path}")


def analyze_results(
        detailed_csv: str = os.path.join(DEFAULT_RENDER_PATH, DEFAULT_DETAILED_CSV)):
    """
    Generate plots and save visualization for baseline results.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = pd.read_csv(detailed_csv)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        sns.boxplot(data=df, x='policy', y='total_reward', ax=axes[0, 0])
        axes[0, 0].set_title('Total Reward Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)

        sns.boxplot(data=df, x='policy', y='episode_length', ax=axes[0, 1])
        axes[0, 1].set_title('Episode Length Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)

        sns.boxplot(data=df, x='policy', y='completion_rate', ax=axes[1, 0])
        axes[1, 0].set_title('Completion Rate Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)

        sns.lineplot(data=df, x='episode', y='total_reward', hue='policy', ax=axes[1, 1])
        axes[1, 1].set_title('Total Reward Over Episodes')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig('results/baseline_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualization saved to results/baseline_analysis.png")
    except ImportError:
        print("Matplotlib/Seaborn not available. Skipping visualizations.")


def run_baseline_evaluation(
        file_path: str = DEFAULT_FILE_PATH,
        render_mode: str = DEFAULT_RENDER_MODE,
        render_path: str = DEFAULT_RENDER_PATH,
        csv_name: str = DEFAULT_CSV_NAME,
        n_episodes: int = DEFAULT_N_EPISODES,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        seed: int = DEFAULT_SEED,
        policies: Optional[List[BaselinePolicy]] = None,
        analyze: bool = False):
    """
    Run evaluation for all (or provided) baseline policies.
    """
    print("Collaborative Writing Recommender System - Baseline Evaluation")
    print("=" * 60)

    env = create_environment(file_path, render_mode, render_path, csv_name, seed)
    if policies is None:
        policies = create_policies(seed=seed)

    if max_steps is None:
        max_steps = env.num_agents * env.num_paragraphs

    print(f"Environment: {env.num_agents} agents, {env.num_paragraphs} paragraphs")
    print(f"Evaluating {len(policies)} baseline policies...\n")

    runner = BaselineRunner(env, policies)
    results = runner.evaluate_policies(n_episodes=n_episodes, max_steps=max_steps)

    # Save results
    os.makedirs(render_path, exist_ok=True)
    runner.save_results(results, os.path.join(render_path, csv_name))

    # Create and save report
    report = runner.create_comparison_report(results)
    with open(os.path.join(render_path, 'baseline_comparison_report.txt'), 'w') as f:
        f.write(report)

    print("\nFinal Results:")
    print(report)

    # Create detailed CSV report
    if analyze:
        create_detailed_csv_report(results, output_path=os.path.join(render_path, 'detailed_baseline_results.csv'))
        analyze_results(detailed_csv=os.path.join(render_path, 'detailed_baseline_results.csv'))

    return results


def run_single_baseline(
        file_path: str = DEFAULT_FILE_PATH,
        policy: BaselinePolicy = None,  # Must be instantiated
        n_episodes: int = 1,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        seed: int = DEFAULT_SEED):
    """
    Run a single policy for baseline evaluation.
    """
    # Env
    env = create_environment(file_path)
    # Max steps
    if max_steps is None:
        max_steps = env.num_agents * env.num_paragraphs

    # Runner
    runner = BaselineRunner(env, [policy])
    results = runner.evaluate_policies(n_episodes=n_episodes, max_steps=max_steps)

    return results


def main(
        file_path: str = DEFAULT_FILE_PATH,
        policy: str = 'all',
        n_episodes: int = DEFAULT_N_EPISODES,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        analyze: bool = False,
        seed: int = DEFAULT_SEED):
    """
    Main entry: select and run experiment with all or a single policy.
    """
    assert file_path is not None, "file_path is required."

    if policy == 'all':
        run_baseline_evaluation(
            file_path=file_path,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            analyze=analyze
        )
    else:
        # Instantiate correct policy by name
        if policy.lower() == 'random':
            pol = RandomPolicy(seed=seed)
        elif policy.lower() == 'popularity':
            pol = PopularityPolicy(metric=PopularityMetric.ENGAGEMENT, seed=seed)
        elif policy.lower() == 'cf':
            pol = CollaborativeFilteringPolicy(seed=seed)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        run_single_baseline(
            file_path=file_path,
            policy=pol,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed
        )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline evaluation')
    parser.add_argument('--policy', type=str, default='all',
                        choices=['all', 'random', 'popularity', 'cf'],
                        help='Policy to run (default: all)')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of episodes to run (default: 20)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max steps per episode (default: agents*paragraphs)')
    parser.add_argument('--analyze', action='store_true',
                        help='Create visualizations after running')
    parser.add_argument('--file_path', type=str, required=True,
                        help='Path to dataset (required)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    main(
        file_path=args.file_path,
        policy=args.policy,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        analyze=args.analyze,
        seed=args.seed
    )

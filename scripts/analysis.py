import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional

# A consistent color map for the policies
POLICY_COLORS = {
    'CollaborativeFiltering': '#4A90E2',  # Blue
    'DQN': '#E67E22',  # Orange
    'Popularity': '#d62728',  # Red
    'Random': '#3ebc73',  # Green
}


def calculate_rolling_stats(data, window_size=10):
    """
    Calculate rolling mean and standard deviation for smoothing plots.
    """
    data = np.array(data)
    rolling_means = []
    rolling_stds = []
    x_indices = []

    for i in range(window_size - 1, len(data)):
        window_data = data[i - window_size + 1:i + 1]
        rolling_means.append(np.mean(window_data))
        rolling_stds.append(np.std(window_data))
        x_indices.append(i)

    return np.array(x_indices), np.array(rolling_means), np.array(rolling_stds)


def plot_rewards(episode_rewards, title, window_size=10):
    """
    Plot episode rewards with rolling mean and standard deviation bands.
    """
    plt.figure(figsize=(10, 6))

    # Calculate rolling statistics
    x_indices, rolling_means, rolling_stds = calculate_rolling_stats(episode_rewards, window_size)

    # Plot the smoothed mean line
    plt.plot(x_indices, rolling_means, label=f'Mean Reward ({window_size}-episode average)',
             color='#3ebc73')

    # Fill between mean ± std in grey
    plt.fill_between(x_indices,
                     rolling_means - rolling_stds,
                     rolling_means + rolling_stds,
                     color='#D3D3D3', alpha=0.4, label='± Std. Dev.')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f"{title}", fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves_by_sparsity(training_data: pd.DataFrame,
                                     policy_name: str,
                                     window_size: int = 10,
                                     output_dir: str = 'plots'):
    """
    Plots the training curve (rolling mean and std) for a specific policy,
    with separate curves for each sparsity level.

    Args:
        training_data: A DataFrame containing the training step metrics, including 'Episode', 'Reward', and 'sparsity'.
        policy_name: The name of the policy to plot (e.g., 'DQN').
        window_size: The window size for the rolling average.
        output_dir: The directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Get unique sparsity levels for the given policy
    sparsities = sorted(training_data['sparsity'].unique())

    # Get the color for the policy from the predefined map
    policy_color = POLICY_COLORS.get(policy_name, '#000000')

    for sparsity in sparsities:
        # Filter data for the current sparsity level
        sparsity_df = training_data[training_data['sparsity'] == sparsity]

        # Calculate episode rewards by summing rewards for each episode
        episode_rewards = sparsity_df.groupby('Episode')['Reward'].sum()

        # Calculate rolling statistics for smoothing
        if len(episode_rewards) > window_size:
            x_indices, rolling_means, rolling_stds = calculate_rolling_stats(episode_rewards, window_size)

            # Plot the smoothed mean line with the consistent color
            plt.plot(x_indices, rolling_means,
                     label=f'Sparsity {sparsity} (Mean)',
                     color=policy_color)

            # Plot the standard deviation band
            plt.fill_between(x_indices,
                             rolling_means - rolling_stds,
                             rolling_means + rolling_stds,
                             color=policy_color, alpha=0.2)
        else:
            print(f"Not enough data for sparsity {sparsity} to calculate rolling stats.")

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'DQN Training Performance by Sparsity', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_filename = f"{policy_name}_training_by_sparsity.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison(episode_data: pd.DataFrame,
                    algorithms_to_plot: Optional[List[str]] = None,
                    output_dir: str = "plots",
                    base_title: str = 'Total Reward per Episode',
                    base_filename: str = 'total_reward'):
    """
    Generates line plots of total reward per episode for different policies and sparsity levels.

    Args:
        episode_data: A DataFrame containing aggregated episode data with 'policy', 'seed',
                      'sparsity', and 'total_reward' columns.
        algorithms_to_plot: A list of policy names to include in the plot. If None, all policies are plotted.
        output_dir: The directory to save the generated plots.
        base_title: The base title for the plots. The sparsity level will be appended.
        base_filename: The base filename for the plots. The sparsity level will be appended.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter the data if a subset of algorithms is specified
    if algorithms_to_plot:
        plot_df = episode_data[episode_data['policy'].isin(algorithms_to_plot)]
    else:
        plot_df = episode_data

    # Get unique sparsity levels to create a separate plot for each
    sparsities = sorted(plot_df['sparsity'].unique())

    for sparsity in sparsities:
        plt.figure(figsize=(10, 6))

        # Filter data for the current sparsity level
        sparsity_df = plot_df[plot_df['sparsity'] == sparsity]

        # Get unique policies to plot each as a separate line
        policies = sorted(sparsity_df['policy'].unique())

        for policy in policies:
            policy_df = sparsity_df[sparsity_df['policy'] == policy]

            # Get the color for the current policy from the predefined map
            # If the policy is not in the map, a default color will be used.
            color = POLICY_COLORS.get(policy, '#000000')

            # Plot the total reward for each episode
            plt.plot(policy_df['seed'], policy_df['total_reward'], label=policy, color=color)

        # Use the new base_title parameter
        plt.title(f'{base_title} (Sparsity: {sparsity})')
        plt.xlabel('Episode (Seed)')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Use the new base_filename parameter
        plot_filename = f"{base_filename}_sparsity_{sparsity}.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()


if __name__ == '__main__':
    # Load the eval_steps.csv file for the plot_comparison function
    file_path_comparison = 'results/analysis_eval_steps.csv'
    eval_steps_df = pd.read_csv(file_path_comparison)

    # Calculate the total reward for each episode (seed) and sparsity level
    episode_data = eval_steps_df.groupby(['policy', 'seed', 'sparsity'])['reward'].sum().reset_index()
    episode_data.rename(columns={'reward': 'total_reward'}, inplace=True)

    # Example 1: Plot all algorithms with custom title and filename
    print("Generating plots for all algorithms...")
    plot_comparison(episode_data,
                    base_title="All Algorithms Reward Performance",
                    base_filename="all_algorithms_reward")

    # Example 2: Plot a subset of algorithms (e.g., DQN and Random) with custom title and filename
    print("Generating plots for a subset of algorithms...")
    plot_comparison(episode_data,
                    algorithms_to_plot=['DQN', 'Random'],
                    base_title="DQN and Random Algorithms Reward Performance",
                    base_filename="dqn_random_reward")

    # New example to demonstrate the training curve plot by sparsity
    print("Generating DQN training curve plot by sparsity levels...")
    # Load the dqn_step_metrics.csv file for the training curve plot
    file_path_training = 'dqn_step_metrics.csv'
    dqn_metrics_df = pd.read_csv(file_path_training)

    # Plot the training curves for the DQN algorithm, separated by sparsity
    plot_training_curves_by_sparsity(dqn_metrics_df, policy_name='DQN')
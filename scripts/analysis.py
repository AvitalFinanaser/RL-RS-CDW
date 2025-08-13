import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

# A consistent color map for the policies and sparsity
POLICY_COLORS = {
    'CollaborativeFiltering': '#4A90E2',  # Blue
    'DQN': '#E67E22',  # Orange
    'Popularity': '#d62728',  # Red
    'Random': '#3ebc73',  # Green
}

SPARSITY_COLORS = {
    0.3: "#E67E22",
    0.5: "#3ebc73",
    0.7: "#4A90E2",
}

component_colors = {
    'Coverage': '#1f77b4',  # Blue
    'Completion': '#ff7f0e',  # Orange
    'Content': '#2ca02c',  # Green
}


# Data loading + enrichment
# -------------------------------

def load_training_data(
        file_path_training_steps='results/dqn_step_metrics.csv',
        file_path_training_episodes='results/dqn_episode_metrics.csv'
):
    dqn_episodes_df = pd.read_csv(file_path_training_episodes)
    dqn_steps_df = pd.read_csv(file_path_training_steps)

    sparsity_map = dqn_episodes_df.set_index('Episode')['Sparsity']
    dqn_steps_df['sparsity'] = dqn_steps_df['Episode'].map(sparsity_map)

    # normalize column names
    dqn_episodes_df.columns = [c.strip().replace(" ", "_") for c in dqn_episodes_df.columns]
    dqn_steps_df.columns = [c.strip().replace(" ", "_") for c in dqn_steps_df.columns]

    # Add a 'policy' column to the dqn_steps_df to allow for filtering
    dqn_steps_df['policy'] = 'DQN'

    return dqn_steps_df, dqn_episodes_df


def get_sparsity_color(sp: float) -> str:
    """Robust mapping for float keys (handles 0.30000004 etc.)."""
    sp_key = round(float(sp), 2)
    return SPARSITY_COLORS.get(sp_key, "#6a51a3")  # default to darkest


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

    # Plot the smoothed mean line (use DQN palette color)
    plt.plot(x_indices, rolling_means, label=f'Mean Reward ({window_size}-episode average)',
             color=POLICY_COLORS.get('DQN', '#E67E22'))

    # Fill between mean ± std in grey (unchanged)
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


# Training


def plot_training_curves_by_sparsity(training_data: pd.DataFrame,
                                     policy_name: str = 'DQN',
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
            c = get_sparsity_color(sparsity)

            plt.plot(x_indices, rolling_means,
                     label=f'Sparsity {sparsity}',
                     color=c)

            # Plot the standard deviation band
            plt.fill_between(x_indices,
                             rolling_means - rolling_stds,
                             rolling_means + rolling_stds,
                             color=c, alpha=0.2)
        else:
            print(f"Not enough data for sparsity {sparsity} to calculate rolling stats.")

    plt.xlabel('Episode', fontweight='bold')
    plt.ylabel('Total Reward', fontweight='bold')
    plt.title(f'DQN Training Performance by Sparsity', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"{policy_name}_training_by_sparsity.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curve_overall(training_data: pd.DataFrame,
                                window_size: int = 10,
                                output_dir: str = 'plots',
                                policy_name: str = 'DQN',
                                filename: str = 'dqn_training_overall.png'):
    """
    Overall DQN training curve (all sparsities together).
    Expects columns: ['Episode','Reward'] in training_data (your steps df).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate per episode and smooth
    episode_rewards = (
        training_data.groupby('Episode')['Reward']
        .sum()
        .sort_index()
    )
    if len(episode_rewards) <= window_size:
        print(f"[skip] not enough episodes for window={window_size}")
        return None

    x_indices, rolling_means, rolling_stds = calculate_rolling_stats(
        episode_rewards.values, window_size
    )

    plt.figure(figsize=(12, 8))
    color = POLICY_COLORS.get(policy_name)
    plt.plot(x_indices, rolling_means,
             label=f"Mean Reward ({window_size}-episode avg)", color=color)
    plt.fill_between(x_indices,
                     rolling_means - rolling_stds,
                     rolling_means + rolling_stds,
                     color=color, alpha=0.20, label="± Std. Dev.")

    plt.xlabel('Episode', fontweight='bold')
    plt.ylabel('Total Reward', fontweight='bold')
    plt.title(f'{policy_name} Training — Overall', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"[saved] {out_path}")
    plt.close()
    return out_path


def plot_reward_components(training_data: pd.DataFrame,
                           sparsity: float = 0.5,
                           window_size: int = 10,
                           output_dir: str = 'plots',
                           policy_name: str = 'DQN'):
    """
    Plots the evolution of individual reward components (coverage, completion, content)
    over training episodes for a specific sparsity level.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    # Filter data for the specified sparsity and policy
    filtered_df = training_data[
        (training_data['sparsity'] == sparsity) &
        (training_data['policy'] == policy_name)
        ]

    # Map for component names to their column names and colors
    components = {
        'Coverage': 'Coverage_Reward',
        'Completion': 'Completion_Reward',
        'Content': 'Content_Reward',
    }

    # Check if there is enough data to plot
    if filtered_df.empty:
        print(f"No data found for {policy_name} with sparsity {sparsity}.")
        return

    # Plot each reward component
    for component, col_name in components.items():
        if col_name in filtered_df.columns:
            # Aggregate per episode and smooth
            episode_rewards = filtered_df.groupby('Episode')[col_name].sum()

            # Check for sufficient episodes
            if len(episode_rewards) > window_size:
                x_indices, rolling_means, rolling_stds = calculate_rolling_stats(
                    episode_rewards.values, window_size
                )
                plt.plot(x_indices, rolling_means,
                         label=f'{component} Reward (Mean)',
                         color=component_colors[component])
                plt.fill_between(x_indices,
                                 rolling_means - rolling_stds,
                                 rolling_means + rolling_stds,
                                 color=component_colors[component],
                                 alpha=0.1)

    plt.xlabel('Episode', fontweight='bold')
    plt.ylabel('Total Reward Component Value', fontweight='bold')
    plt.title(f'DQN Reward Components Evolution (Sparsity: {sparsity})', fontweight='bold', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_filename = f"{policy_name}_reward_components_sparsity_{sparsity}.png"
    out_path = os.path.join(output_dir, plot_filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")
    return out_path


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
        plt.figure(figsize=(12, 8))

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
        plt.title(f'{base_title} (Sparsity: {sparsity})', fontweight='bold', fontsize=12)
        plt.xlabel('Episode', fontweight='bold')
        plt.ylabel('Total Reward', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Use the new base_filename parameter
        plot_filename = f"{base_filename}_sparsity_{sparsity}.png"
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()


def plot_mean_reward_by_sparsity(df: pd.DataFrame,
                                 policies_to_plot: Optional[List[str]] = None,
                                 output_dir: str = 'plots',
                                 title: str = 'Evaluation - Mean Reward by Policy and Sparsity',
                                 filename: str = 'mean_reward_by_sparsity.png'):
    """
    Plots the mean total reward by policy, with sparsity on the x-axis.
    Includes error bars for standard deviation.

    Args:
        df: DataFrame containing evaluation data with columns 'policy', 'sparsity', 'reward', and 'seed'.
        policies_to_plot: List of policy names to include in the plot. Plots all if None.
        output_dir: Directory to save the plots.
        title: Title of the plot.
        filename: Name of the file to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    # First, calculate total reward for each episode
    episode_rewards = df.groupby(['policy', 'sparsity', 'seed'])['reward'].sum().reset_index()

    # Filter data for specified policies
    if policies_to_plot:
        episode_rewards = episode_rewards[episode_rewards['policy'].isin(policies_to_plot)]

    # Calculate mean and standard deviation of total reward for each policy and sparsity
    summary_stats = episode_rewards.groupby(['policy', 'sparsity']).agg(
        mean_reward=('reward', 'mean'),
        std_reward=('reward', 'std')
    ).reset_index()

    # Get the sparsity levels and policies
    sparsities = sorted(summary_stats['sparsity'].unique())
    policies = sorted(summary_stats['policy'].unique())

    plt.figure(figsize=(10, 6))

    for policy in policies:
        policy_data = summary_stats[summary_stats['policy'] == policy]
        x = policy_data['sparsity']
        y = policy_data['mean_reward']
        yerr = policy_data['std_reward']

        # Get color from the predefined map
        color = POLICY_COLORS.get(policy, '#000000')

        plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, label=policy, color=color)

    plt.xlabel('Sparsity', fontweight='bold')
    plt.ylabel('Mean Total Reward ($\pm$ Std. Dev.)', fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=16)
    plt.xticks(sparsities)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
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
                    algorithms_to_plot=['DQN', 'Random', 'Popularity'],
                    base_title="Algorithms Comparison - Reward Performance",
                    base_filename="dqn_comparison_reward")

    # New example to demonstrate the training curve plot by sparsity
    print("Generating DQN training curve plot by sparsity levels...")
    # Load the dqn_step_metrics.csv file for the training curve plot
    file_path_training = 'dqn_step_metrics.csv'
    dqn_metrics_df = pd.read_csv(file_path_training)


    print("Generating mean reward by sparsity plot...")
    plot_mean_reward_by_sparsity(eval_steps_df,
                                 policies_to_plot=['DQN', 'Popularity', 'Random'])

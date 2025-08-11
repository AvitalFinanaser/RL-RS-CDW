"""
A script to load a trained model and run it in the environment for some episodes, collecting evaluation metrics.
"""

# evaluate.py - Run this after training completes
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
import gymnasium as gym
from environment.env import CollaborativeDocRecEnv
from baselines import *


def run_episode(env, policy, seed, fixed_sparsity=0.3):
    # Set fixed sparsity for eval
    env.stance_loader.sparsity = fixed_sparsity
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    steps = 0
    while True:
        if isinstance(policy, DQN):
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = policy.select_action(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        if done:
            break
    abandonment_rate = 1 - len(env.active_agents) / env.num_agents
    completion_rate = env._get_stance_completion_rate()
    return {
        "total_reward": total_reward,
        "steps": steps,
        "abandonment_rate": abandonment_rate,
        "completion_rate": completion_rate,
        "active_agents": len(env.active_agents),
        "seed": seed
    }

def evaluate_policies(env, policies, n_episodes=100, base_seed=42):
    results = {name: [] for name in policies}
    for episode in range(n_episodes):
        seed = base_seed + episode
        for name, policy in policies.items():
            result = run_episode(env, policy, seed)
            results[name].append(result)
    return results

def plot_results(results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    for policy_name, policy_results in results.items():
        for res in policy_results:
            res["policy"] = policy_name
            all_data.append(res)
    df = pd.DataFrame(all_data)
    df.to_csv(os.path.join(output_dir, "eval_results.csv"), index=False)

    # Plots for key metrics
    for metric in ["total_reward", "abandonment_rate", "completion_rate"]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="seed", y=metric, hue="policy")
        plt.title(f"{metric.capitalize()} per Episode")
        plt.xlabel("Episode (Seed)")
        plt.savefig(os.path.join(output_dir, f"{metric}_plot.png"))
        plt.close()

    print("Evaluation complete; results and plots saved in", output_dir)

if __name__ == "__main__":
    # Use same config as training (fixed sparsity for eval)
    config = {
        "file_path": "datasets/instances/instance2",
        "instance_path": "datasets/instances/instance2",
        "sparsity": 0.3,  # Fixed for eval
        "num_agents": 20,
        "num_paragraphs": 50,
        "seed": 42
    }
    env = CollaborativeDocRecEnv.from_config(config, render_mode='csv', render_csv_name="eval_render.csv")
    env = ActionMaskWrapper(env)  # Keep wrapper, but sparsity is fixed

    # Load trained model
    model = DQN.load("dqn_new_masked_cdw", env=env)

    # Policies to evaluate
    policies = {
        "DQN": model,
        "Random": RandomPolicy(),
        "Popularity": PopularityPolicy()
    }

    # Run evaluation
    eval_results = evaluate_policies(env, policies)
    plot_results(eval_results)
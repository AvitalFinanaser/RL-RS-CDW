import os
from typing import Dict, Any, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
# Env
from environment.env import CollaborativeDocRecEnv
from rl_algorithms.dqn_train import ActionMaskWrapper
from baselines import *


# ===== Utilities =====

def make_env(file_path: str,
             num_agents: int,
             num_paragraphs: int,
             sparsity: float,
             seed: int,
             render_csv_name: str = "eval_render.csv"):
    cfg = {
        "file_path": file_path,
        "instance_path": file_path,
        "num_agents": num_agents,
        "num_paragraphs": num_paragraphs,
        "sparsity": sparsity,
        "seed": seed,
    }
    env = CollaborativeDocRecEnv.from_config(cfg, render_mode="csv", render_csv_name=render_csv_name)
    env = ActionMaskWrapper(env)
    return env


def set_dqn_deterministic(model: DQN):
    if hasattr(model, "exploration_rate"):
        model.exploration_rate = 0.0
    if hasattr(model, "policy"):
        model.policy.set_training_mode(False)


def current_valid_actions(env, obs: Dict[str, Any]) -> List[int]:
    """ (Mapping index -> agent_id)."""
    if "current_agent_id" not in obs:
        return []
    current_idx = int(obs["current_agent_id"])
    if current_idx < 0 or current_idx >= env.num_agents:
        return []
    agent_id = env.agents[current_idx].agent_id
    return env.get_valid_actions(agent_id)


# ===== Core eval loop =====

def run_episode(env,
                policy_obj_or_model,
                seed: int,
                fixed_sparsity: Optional[float] = None,
                max_steps: int = 10_000,
                log_steps: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if fixed_sparsity is not None and hasattr(env, "stance_loader"):
        env.stance_loader.sparsity = float(fixed_sparsity)

    obs, info = env.reset(seed=seed)

    is_dqn = isinstance(policy_obj_or_model, DQN)
    if is_dqn:
        set_dqn_deterministic(policy_obj_or_model)
    else:
        assert isinstance(policy_obj_or_model, BaselinePolicy)
        policy_obj_or_model.reset()

    total_reward = 0.0
    steps = 0
    step_rows: List[Dict[str, Any]] = []

    while steps < max_steps:
        valid_actions = current_valid_actions(env, obs)
        if len(valid_actions) == 0:
            break

        if is_dqn:
            action, _ = policy_obj_or_model.predict(obs, deterministic=True)
            action = int(action)
            # ודא חוקיות — אם לא, בחר אקראי חוקי
            if action not in valid_actions:
                action = int(np.random.choice(valid_actions))
        else:
            action = int(policy_obj_or_model.select_action(observation=obs, valid_actions=valid_actions))

        next_obs, reward, terminated, truncated, info = env.step(action)

        if not is_dqn:
            policy_obj_or_model.update(observation=obs, action=action,
                                       reward=float(reward), next_observation=next_obs, info=info)

        total_reward += float(reward)
        steps += 1

        if log_steps:
            rc = info.get("reward_components", {})
            step_rows.append({
                "step": steps,
                "reward": float(reward),
                "coverage_r": rc.get("coverage", np.nan),
                "completion_r": rc.get("completion", np.nan),
                "content_r": rc.get("content", np.nan),
                "agent_id": info.get("agent_id", -1),
                "paragraph_id": info.get("paragraph_id", -1),
                "vote": info.get("vote", "?"),
                "continuation": info.get("continuation", -1),
                "sparsity": getattr(env.stance_loader, "sparsity", np.nan),
            })

        obs = next_obs
        if terminated or truncated:
            break

    if not is_dqn:
        policy_obj_or_model.end_of_episode()

    completion = env._get_stance_completion_rate()
    active_final = len(env.active_agents)
    abandonment_rate = 1.0 - (active_final / env.num_agents if env.num_agents else 0.0)

    ep_metrics = {
        "seed": seed,
        "sparsity": getattr(env.stance_loader, "sparsity", np.nan),
        "total_reward": total_reward,
        "total_steps": steps,
        "completion_rate": completion,
        "abandonment_rate": abandonment_rate,
        "active_agents_final": active_final,
    }
    return ep_metrics, step_rows


def eval_policy(name: str,
                policy_obj_or_model,
                base_env,
                n_episodes: int,
                base_seed: int,
                rotate_sparsities: Optional[List[float]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates one policy; returns (episodes_df, steps_df) with a 'policy' column."""
    ep_rows, st_rows = [], []
    if rotate_sparsities:
        chunks = np.array_split(list(range(n_episodes)), len(rotate_sparsities))
        for sp, idxs in zip(rotate_sparsities, chunks):
            for i in idxs:
                seed = base_seed + i
                ep, steps = run_episode(base_env, policy_obj_or_model, seed, fixed_sparsity=sp)
                ep_rows.append({**ep, "policy": name})
                for r in steps:
                    r2 = dict(r);
                    r2["policy"] = name;
                    r2["seed"] = seed
                    st_rows.append(r2)
    else:
        for i in range(n_episodes):
            seed = base_seed + i
            ep, steps = run_episode(base_env, policy_obj_or_model, seed)
            ep_rows.append({**ep, "policy": name})
            for r in steps:
                r2 = dict(r);
                r2["policy"] = name;
                r2["seed"] = seed
                st_rows.append(r2)

    return pd.DataFrame(ep_rows), pd.DataFrame(st_rows)


def plot_results(ep_df: pd.DataFrame, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Plots for key metrics
    for metric in ["total_reward", "abandonment_rate", "completion_rate"]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ep_df, x="seed", y=metric, hue="policy")
        plt.title(f"{metric.capitalize()} per Episode")
        plt.xlabel("Episode (Seed)")
        plt.savefig(os.path.join(output_dir, f"{metric}_plot.png"))
        plt.close()

    print("Evaluation complete; results and plots saved in", output_dir)


if __name__ == "__main__":
    out_dir = "results"
    seed = 42
    episodes = 100

    # Use same config as training (fixed sparsity for eval)
    config = {
        "file_path": "datasets/instances/instance2",
        "instance_path": "datasets/instances/instance2",
        "sparsity": 0.3,  # Fixed for eval
        "num_agents": 20,
        "num_paragraphs": 50,
        "seed": seed
    }
    env = CollaborativeDocRecEnv.from_config(config, render_mode='csv', render_csv_name="eval_render.csv")
    env = ActionMaskWrapper(env)

    # Load trained model
    model = DQN.load("dqn_cdw", env=env)

    # Policies to evaluate
    policies = {
        "DQN": model,
        "Random": RandomPolicy(),
        "Popularity": PopularityPolicy(),
        "CollaborativeFiltering": CollaborativeFilteringPolicy()
    }

    # === Evaluate ===
    ep_all, st_all = [], []
    for name, pol in policies.items():
        print(f"Evaluating {name} ...")
        ep_df, st_df = eval_policy(
            name=name,
            policy_obj_or_model=pol,
            base_env=env,
            n_episodes=episodes,
            base_seed=seed,
        )
        ep_all.append(ep_df)
        st_all.append(st_df)

    ep_all_df = pd.concat(ep_all, ignore_index=True)
    st_all_df = pd.concat(st_all, ignore_index=True)

    # Save
    ep_path = os.path.join(out_dir, "eval_episodes.csv")
    st_path = os.path.join(out_dir, "eval_steps.csv")
    ep_all_df.to_csv(ep_path, index=False)
    st_all_df.to_csv(st_path, index=False)

    # Plot
    plot_results(ep_all_df, out_dir)

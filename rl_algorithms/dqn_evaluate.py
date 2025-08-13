import os
from typing import Dict, Any, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from environment.env import CollaborativeDocRecEnv
from baselines import *
import gymnasium as gym
from rl_algorithms.dqn_train import MaskableDQNPolicy, CustomFeaturesExtractor


class EvalActionMaskWrapper(gym.Wrapper):
    """Wrapper that enforces the current action mask at step-time (no random sparsity here)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def step(self, action):
        # Enforce valid actions using env-side mask
        obs = self.env._get_obs()
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            obs, reward, done, truncated, info = self.env.step(0)
            # convert to SB3 vec-env-compatible tuple upstream; here we just pass through
            return obs, reward, True, truncated, info
        if action not in valid_actions:
            action = np.random.choice(valid_actions)
        return self.env.step(action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            seed = int(seed)
        return self.env.reset(seed=seed, options=options)

    def __getattr__(self, name):
        return getattr(self.env, name)


def unwrap_base_env(vec_env) -> Any:
    """Return the underlying gym env (beneath DummyVecEnv and our wrapper)."""
    base = vec_env.envs[0]
    while hasattr(base, "env") and getattr(base, "env") is not None:
        base = base.env
    return base


def to_single_env_obs(vec_obs: Any) -> Dict[str, Any]:
    """Convert DummyVecEnv observation (dict of arrays with leading batch dim) into single-env dict."""
    if isinstance(vec_obs, dict):
        single = {}
        for k, v in vec_obs.items():
            a = np.asarray(v)
            single[k] = a[0] if a.ndim > 0 else a
        return single
    return vec_obs


def vec_seed_and_reset(vec_env, seed: int):
    """Seed all sub-envs and reset (SB3 DummyVecEnv API: returns obs only)."""
    vec_env.seed(int(seed))
    return vec_env.reset()


def set_dqn_deterministic(model: DQN):
    # zero exploration, eval mode
    if hasattr(model, "exploration_rate"):
        model.exploration_rate = 0.0
    if hasattr(model, "policy"):
        model.policy.set_training_mode(False)


def current_valid_actions(env, obs: Any) -> List[int]:
    """Get valid actions for the current agent, handling vec/single obs."""
    base = unwrap_base_env(env)
    single_obs = to_single_env_obs(obs)

    if "current_agent_id" not in single_obs:
        return []

    # make sure this is a python int, not a 1-element array
    current_idx = int(np.asarray(single_obs["current_agent_id"]).item())
    if current_idx < 0 or current_idx >= base.num_agents:
        return []

    agent_id = base.agents[current_idx].agent_id
    return base.get_valid_actions(agent_id)


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
    env = EvalActionMaskWrapper(env)
    env = DummyVecEnv([lambda: env])  # SB3 expects a VecEnv
    return env


# ===== Core eval loop =====

def run_episode(env,
                policy_obj_or_model,
                seed: int,
                fixed_sparsity: Optional[float] = None,
                max_steps: int = 10_000,
                log_steps: bool = True,
                episode_num: int = 0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Set fixed sparsity per episode (if requested)
    base = unwrap_base_env(env)
    if fixed_sparsity is not None and hasattr(base, "stance_loader"):
        base.stance_loader.sparsity = float(fixed_sparsity)

    # Reset with seed (vec env returns obs only)
    obs = vec_seed_and_reset(env, seed)

    # Policy mode
    is_dqn = isinstance(policy_obj_or_model, DQN)
    if is_dqn:
        set_dqn_deterministic(policy_obj_or_model)
    else:
        assert isinstance(policy_obj_or_model, BaselinePolicy)
        policy_obj_or_model.reset()

    total_reward = 0.0
    steps = 0
    step_rows: List[Dict[str, Any]] = []

    policy_name = policy_obj_or_model.__class__.__name__

    while steps < max_steps:
        # valid actions from the (single) current agent
        valid_actions = current_valid_actions(env, obs)
        if not valid_actions:
            break

        if is_dqn:
            # DQN expects vec obs
            action, _ = policy_obj_or_model.predict(obs, deterministic=True)
            action = int(action[0])
            if action not in valid_actions:
                action = int(np.random.choice(valid_actions))
        else:
            # Baselines expect single-env obs
            single_obs = to_single_env_obs(obs)
            action = int(policy_obj_or_model.select_action(observation=single_obs, valid_actions=valid_actions))

        # SB3 DummyVecEnv.step returns (obs, rewards, dones, infos)
        next_obs, reward, done, info = env.step([action])

        if not is_dqn:
            single_prev = to_single_env_obs(obs)
            single_next = to_single_env_obs(next_obs)
            policy_obj_or_model.update(
                observation=single_prev, action=action,
                reward=float(reward[0]), next_observation=single_next, info=info[0]
            )

        total_reward += float(reward[0])
        steps += 1

        if log_steps:
            rc = info[0].get("reward_components", {})
            step_rows.append({
                "step": steps,
                "reward": float(reward[0]),
                "coverage_r": rc.get("coverage", np.nan),
                "completion_r": rc.get("completion", np.nan),
                "content_r": rc.get("content", np.nan),
                "agent_id": info[0].get("agent_id", -1),
                "paragraph_id": info[0].get("paragraph_id", -1),
                "vote": info[0].get("vote", "?"),
                "continuation": info[0].get("continuation", -1),
                "sparsity": getattr(base.stance_loader, "sparsity", np.nan),
            })

            # lightweight console trace (optional)
            print(
                f"Policy: {policy_name}, Ep: {episode_num}, Seed: {seed}, "
                f"Step: {steps}, Action: {action}, R: {float(reward[0]):.4f}, "
                f"Agent: {info[0].get('agent_id', -1)}, Para: {info[0].get('paragraph_id', -1)}, "
                f"Vote: {info[0].get('vote', '?')}"
            )

        obs = next_obs
        if done[0]:
            print(f"Episode {episode_num} terminated at step {steps}. Final reward: {total_reward:.4f}")
            break

    if not is_dqn:
        policy_obj_or_model.end_of_episode()

    completion = base._get_stance_completion_rate()
    active_final = len(base.active_agents)
    abandonment_rate = 1.0 - (active_final / base.num_agents if base.num_agents else 0.0)

    ep_metrics = {
        "seed": seed,
        "sparsity": getattr(base.stance_loader, "sparsity", np.nan),
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
    """Evaluate one policy; return (episodes_df, steps_df) with a 'policy' column."""
    ep_rows, st_rows = [], []

    if rotate_sparsities:
        chunks = np.array_split(list(range(n_episodes)), len(rotate_sparsities))
        for sp, idxs in zip(rotate_sparsities, chunks):
            for i in idxs:
                seed = base_seed + i
                ep, steps = run_episode(base_env, policy_obj_or_model, seed, fixed_sparsity=sp, episode_num=i)
                ep_rows.append({**ep, "policy": name})
                for r in steps:
                    r2 = dict(r)
                    r2["policy"] = name
                    r2["seed"] = seed
                    st_rows.append(r2)
    else:
        for i in range(n_episodes):
            seed = base_seed + i
            ep, steps = run_episode(base_env, policy_obj_or_model, seed, episode_num=i)
            ep_rows.append({**ep, "policy": name})
            for r in steps:
                r2 = dict(r)
                r2["policy"] = name
                r2["seed"] = seed
                st_rows.append(r2)

    return pd.DataFrame(ep_rows), pd.DataFrame(st_rows)


# ===== Plots =====


def plot_results(ep_df: pd.DataFrame, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    for metric in ["total_reward", "abandonment_rate", "completion_rate"]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ep_df, x="seed", y=metric, hue="policy")
        plt.title(f"{metric.replace('_', ' ').title()} per Episode")
        plt.xlabel("Episode (Seed)")
        plt.savefig(os.path.join(output_dir, f"{metric}_plot.png"))
        plt.close()
    print("Evaluation complete; results and plots saved in", output_dir)


def plot_results(ep_df: pd.DataFrame, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    for metric in ["total_reward", "abandonment_rate", "completion_rate"]:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=ep_df, x="seed", y=metric, hue="policy")
        plt.title(f"{metric.capitalize()} per Episode")
        plt.xlabel("Episode (Seed)")
        plt.savefig(os.path.join(output_dir, f"{metric}_plot.png"))
        plt.close()
    print("Evaluation complete; results and plots saved in", output_dir)


# ===== Tests =====

def test_initial_states(config, test_seeds=[42, 43], sparsities=[0.3, 0.5, 0.7]):
    """Test if initial states are identical for same seed/sparsity across resets (non-vec env)."""
    for sp in sparsities:
        print(f"Testing sparsity {sp}")
        env = CollaborativeDocRecEnv.from_config(config, render_mode='csv', render_csv_name="test_render.csv")
        env = EvalActionMaskWrapper(env)
        initial_obs_list = []
        for seed in test_seeds:
            env.stance_loader.sparsity = sp
            obs, _ = env.reset(seed=int(seed))
            stance_hash = hash(tuple(obs["stance_matrix"].flatten()))
            initial_obs_list.append((seed, stance_hash))
            print(f"Seed {seed}: Stance hash {stance_hash}")
        for seed in test_seeds:
            env.stance_loader.sparsity = sp
            obs, _ = env.reset(seed=int(seed))
            stance_hash = hash(tuple(obs["stance_matrix"].flatten()))
            assert (seed, stance_hash) in initial_obs_list, f"Inconsistent state for seed {seed} at sparsity {sp}"
        print(f"Sparsity {sp} consistent.")


def test_same_initial_states_across_algos(base_env, policies, test_seeds=[42, 43], sparsities=[0.3, 0.5, 0.7]):
    """Test if all algorithms face the same initial states for the same seed and sparsity (vec env)."""
    base = unwrap_base_env(base_env)
    for sp in sparsities:
        print(f"Testing sparsity {sp} across algorithms")
        for seed in test_seeds:
            policy_hashes = {}
            for name, _ in policies.items():
                base_env.seed(int(seed))
                base.stance_loader.sparsity = sp
                obs = base_env.reset()  # vec obs (dict of arrays)
                stance_hash = hash(tuple(to_single_env_obs(obs)["stance_matrix"].flatten()))
                policy_hashes[name] = stance_hash
                print(f"Policy {name}, Seed {seed}, Sparsity {sp}: Stance hash {stance_hash}")
            unique_hashes = set(policy_hashes.values())
            assert len(unique_hashes) == 1, f"Different initial states for seed {seed}, sparsity {sp}: {policy_hashes}"
        print(f"Sparsity {sp} consistent across algorithms.")


def get_underlying_env(env):
    """Navigate through wrappers to get the base environment."""
    while hasattr(env, 'env') and env.env is not None:
        env = env.env
    return env


def vec_seed_and_reset(vec_env, seed: int):
    s = int(seed)
    vec_env.seed(s)
    return vec_env.reset()


if __name__ == "__main__":
    out_dir = "results"
    seed = 42
    episodes = 300

    config = {
        "file_path": "datasets/instances/instance2",
        "instance_path": "datasets/instances/instance2",
        "sparsity": 0.3,
        "num_agents": 20,
        "num_paragraphs": 50,
        "seed": seed
    }
    env = make_env(
        file_path=config["file_path"],
        num_agents=config["num_agents"],
        num_paragraphs=config["num_paragraphs"],
        sparsity=config["sparsity"],
        seed=config["seed"],
        render_csv_name="eval_render.csv"
    )

    # Policies to evaluate
    policies = {
        "Random": RandomPolicy(),
        "Popularity": PopularityPolicy(),
        "CollaborativeFiltering": CollaborativeFilteringPolicy()
    }

    # Loading DQN model
    model = DQN.load(
        "dqn_cdw.zip",  # Ensure the .zip extension is correct
        env=env,
        custom_objects={
            "policy_class": MaskableDQNPolicy,
            "features_extractor_class": CustomFeaturesExtractor,
        },
        device="cpu",
        print_system_info=True,
    )
    policies["DQN"] = model

    # Run tests
    # test_initial_states(config=config, test_seeds=[42, 43], sparsities=[0.3, 0.5, 0.7])
    # test_same_initial_states_across_algos(base_env=env, policies=policies, test_seeds=[42, 43], sparsities=[0.3, 0.5, 0.7])

    # Evaluate
    ep_all, st_all = [], []
    for name, pol in policies.items():
        print(f"Evaluating {name} ...")
        ep_df, st_df = eval_policy(
            name=name,
            policy_obj_or_model=pol,
            base_env=env,
            n_episodes=episodes,
            base_seed=seed,
            rotate_sparsities=[0.3, 0.5, 0.7]
        )
        ep_all.append(ep_df)
        st_all.append(st_df)

    ep_all_df = pd.concat(ep_all, ignore_index=True)
    st_all_df = pd.concat(st_all, ignore_index=True)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    ep_path = os.path.join(out_dir, "eval_episodes.csv")
    st_path = os.path.join(out_dir, "eval_steps.csv")
    ep_all_df.to_csv(ep_path, index=False)
    st_all_df.to_csv(st_path, index=False)

    # Plot
    plot_results(ep_all_df, out_dir)

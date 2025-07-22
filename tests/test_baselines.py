"""
Test module for Baseline Models (Random, Popularity, Collaborative Filtering) for Collaborative Writing
"""
import os
import numpy as np
from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader
from baselines.baseline_models import RandomPolicy, PopularityPolicy, CollaborativeFilteringPolicy, BaselineRunner, \
    PopularityMetric

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


# Use the same file_path as your environment tests, or a smaller synthetic one for speed.
file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"


def setup_env():
    paragraphs_loader = ParagraphsLoader(file_path)
    agents_loader = AgentsLoader(file_path)
    events_loader = EventsLoader(file_path)
    return CollaborativeDocRecEnv(
        paragraphs_loader=paragraphs_loader,
        agents_loader=agents_loader,
        events_loader=events_loader,
        render_path="./render/",
        render_csv_name="baselines.csv",
        seed=123
    )


def test_random_policy_step():
    print("Testing RandomPolicy selection and update...")
    env = setup_env()
    policy = RandomPolicy(seed=123)
    obs, _ = env.reset()
    agent_id = obs["current_agent_id"]
    if agent_id < 0:
        print("No active agent at reset. Skipping.")
        return
    valid_actions = env.get_valid_actions(agent_id)
    assert valid_actions, "No valid actions for initial agent"
    action = policy.select_action(obs, valid_actions)
    assert action in valid_actions
    next_obs, reward, terminated, truncated, info = env.step(action)
    policy.update(obs, action, reward, next_obs, info)
    print("RandomPolicy test passed.\n")



def test_popularity_policy_step():
    print("Testing PopularityPolicy selection and update...")
    env = setup_env()
    policy = PopularityPolicy(metric=PopularityMetric.ENGAGEMENT, seed=123)
    obs, _ = env.reset()
    agent_id = obs["current_agent_id"]
    if agent_id < 0:
        print("No active agent at reset. Skipping.")
        return
    valid_actions = env.get_valid_actions(agent_id)
    action = policy.select_action(obs, valid_actions)
    assert action in valid_actions
    next_obs, reward, terminated, truncated, info = env.step(action)
    # Simulate info with vote if needed
    if "vote" not in info:
        info["vote"] = int(np.sign(reward))
    policy.update(obs, action, reward, next_obs, info)
    # Check popularity stats
    stats = policy.get_popularity_stats()
    assert isinstance(stats, dict)
    print("PopularityPolicy test passed.\n")


def test_cf_policy_step():
    print("Testing CollaborativeFilteringPolicy selection and update...")
    env = setup_env()
    policy = CollaborativeFilteringPolicy(n_factors=2, seed=123, min_interactions=1)
    obs, _ = env.reset()
    agent_id = obs["current_agent_id"]
    if agent_id < 0:
        print("No active agent at reset. Skipping.")
        return
    valid_actions = env.get_valid_actions(agent_id)
    action = policy.select_action(obs, valid_actions)
    assert action in valid_actions
    next_obs, reward, terminated, truncated, info = env.step(action)
    if "vote" not in info:
        info["vote"] = int(np.sign(reward))
    policy.update(obs, action, reward, next_obs, info)
    # After one interaction, model should be initialized
    stats = policy.get_model_stats()
    assert stats.get("model_initialized", False)
    print("CollaborativeFilteringPolicy test passed.\n")


def test_runner_integration():
    print("Testing BaselineRunner integration (one episode each)...")
    env = setup_env()
    policies = [
        RandomPolicy(seed=123),
        PopularityPolicy(metric=PopularityMetric.ENGAGEMENT, seed=123),
        CollaborativeFilteringPolicy(n_factors=2, seed=123, min_interactions=1)
    ]
    runner = BaselineRunner(env, policies)
    max_steps = env.num_agents * env.num_paragraphs
    results = runner.evaluate_policies(n_episodes=1, max_steps=max_steps)
    assert "Random" in results and "Popularity" in results and "CollaborativeFiltering" in results
    for policy, res in results.items():
        assert "mean_total_reward" in res
    print("BaselineRunner integration test passed.\n")


def run_all_baseline_tests():
    print("RUNNING BASELINE MODULE TEST SUITE")
    test_random_policy_step()
    test_popularity_policy_step()
    test_cf_policy_step()
    test_runner_integration()
    print("ALL BASELINE MODULE TESTS PASSED!")


if __name__ == "__main__":
    run_all_baseline_tests()

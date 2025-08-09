"""
Test module for CollaborativeDocRecEnv (environment integration and RL loop)
"""
import os

from environment.env import CollaborativeDocRecEnv  # adjust to your actual env module path
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader, StanceLoader
from environment.stance import StanceMatrix
from environment.reward_shaping import RewardShaper
import numpy as np

file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"


def setup_env():
    # Set up real or synthetic data using loaders
    paragraphs_loader = ParagraphsLoader(file_path)
    agents_loader = AgentsLoader(file_path)
    events_loader = EventsLoader(file_path)
    return CollaborativeDocRecEnv(
        paragraphs_loader=paragraphs_loader,
        agents_loader=agents_loader,
        events_loader=events_loader,
        render_path="./render/",
        seed=123
    )


def test_env_initialization():
    print("Testing environment initialization...")
    env = setup_env()
    assert env.num_agents > 0 and env.num_paragraphs > 0
    assert env.stance.matrix.shape == (env.num_agents, env.num_paragraphs)
    assert set(env.agent_ids) == set(a.agent_id for a in env.agents)
    print("Initialization test passed.\n")


def test_reset():
    print("Testing reset...")
    env = setup_env()
    obs, _ = env.reset()
    assert isinstance(obs, dict)
    assert "stance_matrix" in obs
    assert obs["stance_matrix"].shape == (env.num_agents, env.num_paragraphs)
    print("Reset test passed.\n")


def test_valid_actions_and_step():
    print("Testing valid actions and single step execution...")
    env = setup_env()
    env.reset()
    for _ in range(5):
        valid_actions = env._get_valid_actions(env.current_agent_id)
        assert isinstance(valid_actions, list) and all(isinstance(a, int) for a in valid_actions)
        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, dict)
        assert "stance_matrix" in obs
        assert isinstance(reward, float)
        if terminated:
            break
    print("Valid actions and step test passed.\n")


def test_episode_completion():
    print("Testing full episode until termination...")
    env = setup_env()
    obs, _ = env.reset()
    steps = 0
    while True:
        valid_actions = env._get_valid_actions(env.current_agent_id)
        if not valid_actions:
            break
        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated:
            break
    assert terminated or not env.active_agents or env.stance.is_complete()
    print(f"Episode completed in {steps} steps.\n")


def test_observation_space_consistency():
    print("Testing observation space output consistency...")
    env = setup_env()
    obs, _ = env.reset()
    assert set(obs.keys()) == set(env.observation_space.spaces.keys())
    print("Observation space consistency test passed.\n")


def test_agent_paragraph_mappings():
    print("Testing action-paragraph mappings...")
    env = setup_env()
    for action, pid in env.action_to_paragraph.items():
        assert env.paragraph_to_action[pid] == action
    print("Agent/paragraph mapping test passed.\n")


def test_render_modes():
    print("Testing render() with 'csv' output file...")
    env = setup_env()
    env.render_mode = 'csv'
    obs, _ = env.reset()

    # Take at least one step to generate a memory event
    valid_actions = env._get_valid_actions(env.current_agent_id)
    if valid_actions:
        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    expected_file = os.path.join(env.render_path, 'render.csv')
    assert os.path.exists(expected_file), "Render CSV file was not created."
    os.remove(expected_file)
    print("CSV render file test passed.\n")


def test_render_csv_output():
    env = setup_env()
    env.render_mode = 'csv'
    obs, _ = env.reset()
    env.render_csv_name = "test.csv"

    # Add an interaction to create events
    valid_actions = env._get_valid_actions(env.current_agent_id)
    env.step(valid_actions[0])

    env.render()
    expected_file = os.path.join(env.render_path, env.render_csv_name)
    assert os.path.exists(expected_file), "Render CSV file was not created."
    print(f"A csv file was created in {expected_file}")


def test_stance_loader_reproducibility():
    # Setup known agents and paragraphs
    agents_loader = AgentsLoader(file_path)
    paragraphs_loader = ParagraphsLoader(file_path)
    agents = agents_loader.load_all()
    paragraphs = paragraphs_loader.load_all()
    stance_loader = StanceLoader(agents=agents, paragraphs=paragraphs, sparsity=0.5, seed=123)
    env = CollaborativeDocRecEnv(
        paragraphs_loader=paragraphs_loader,
        agents_loader=agents_loader,
        stance_loader=stance_loader,
        seed=123
    )
    obs1, _ = env.reset()
    matrix1 = env.stance.matrix.copy(deep=True)
    obs2, _ = env.reset()
    matrix2 = env.stance.matrix.copy(deep=True)
    assert matrix1.equals(matrix2), "Stance matrix is not reproducible with same seed."


def run_all_env_tests():
    print("RUNNING ENVIRONMENT MODULE TEST SUITE")
    test_env_initialization()
    test_reset()
    test_valid_actions_and_step()
    test_episode_completion()
    test_observation_space_consistency()
    test_agent_paragraph_mappings()
    test_render_csv_output()
    test_render_modes()
    print("ALL ENVIRONMENT MODULE TESTS PASSED!")


if __name__ == "__main__":
    #run_all_env_tests()
    test_stance_loader_reproducibility()
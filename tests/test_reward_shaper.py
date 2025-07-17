"""
Test module for RewardShaper class
"""
from environment.reward_shaping import RewardShaper
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader
from environment.stance import StanceMatrix
from environment.topic_model import compute_paragraphs_topic_matrix
from environment.collaborators import LLMAgentWithTopics


def setup_test_data(file_path):
    """Load real data using loaders."""

    # Load data
    paragraphs_loader = ParagraphsLoader(filepath=file_path)
    agents_loader = AgentsLoader(filepath=file_path)
    events_loader = EventsLoader(filepath=file_path)

    paragraphs = paragraphs_loader.load_all()
    agents = agents_loader.load_all()
    events_df = events_loader.load_all(agents=agents, paragraphs=paragraphs)

    # Loading stance matrix from previous events
    stance = StanceMatrix.from_existing(agents=agents, paragraphs=paragraphs, matrix=events_df)

    # Edited stance with manually adding some test votes
    # stance = StanceMatrix(agents=agents, paragraphs=paragraphs)
    # stance.set_vote(agents[0].agent_id, paragraphs[0].paragraph_id, "1")
    # stance.set_vote(agents[0].agent_id, paragraphs[1].paragraph_id, "-1")
    # stance.set_vote(agents[1].agent_id, paragraphs[0].paragraph_id, "1")
    # stance.set_vote(agents[1].agent_id, paragraphs[2].paragraph_id, "0")

    # Topic modeling
    texts = [p.text for p in paragraphs]
    doc_topic_matrix, best_k, topic_keywords = compute_paragraphs_topic_matrix(texts, k_range=range(2,3))

    # Attach topics to paragraphs
    paragraphs = paragraphs_loader.attach_topics(paragraphs, doc_topic_matrix)

    # Attach topics to agents
    LLMAgentWithTopics.update_all_topic_vectors(
        agents=agents,
        doc_topic_matrix=doc_topic_matrix,
        stance_matrix=stance
    )

    return agents, paragraphs, stance, doc_topic_matrix


def test_reward_shaper_initialization():
    """Test RewardShaper initialization with default and custom parameters."""
    print("Testing RewardShaper initialization...")

    # Test default parameters
    print("Testing default parameters initialization")
    shaper = RewardShaper()
    assert shaper.params["w1"] == 0.4
    assert shaper.params["w2"] == 0.3
    assert shaper.params["w3"] == 0.3
    assert shaper.params["use_jsd"] == True
    print("Default parameters correct")

    # Test custom parameters
    print("Testing custom parameters initialization")
    custom_params = {"w1": 0.5, "w2": 0.25, "w3": 0.25, "use_jsd": False}
    shaper_custom = RewardShaper(custom_params)
    assert shaper_custom.params["w1"] == 0.5
    assert shaper_custom.params["w2"] == 0.25
    assert shaper_custom.params["w3"] == 0.25
    assert shaper_custom.params["use_jsd"] == False
    print("Custom parameters correct")

    print("RewardShaper initialization test passed!\n")


def test_coverage_reward(file_path):
    """Test coverage reward calculation."""
    print("Testing coverage reward...")

    agents, paragraphs, stance, doc_topic_matrix = setup_test_data(file_path)
    shaper = RewardShaper()

    # Test with JSD
    print("Testing coverage reward with JSD")
    coverage_reward_jsd = shaper._coverage_reward(agents=agents, doc_topic_matrix=doc_topic_matrix)
    assert isinstance(coverage_reward_jsd, float)
    assert 0 <= coverage_reward_jsd <= 1
    print(f"JSD coverage reward: {coverage_reward_jsd:.3f}")

    # Test with cosine similarity
    print("Testing coverage reward with cosine")
    shaper.params["use_jsd"] = False
    coverage_reward_cosine = shaper._coverage_reward(agents=agents, doc_topic_matrix=doc_topic_matrix)
    assert isinstance(coverage_reward_cosine, float)
    assert -1 <= coverage_reward_cosine <= 1
    print(f"Cosine coverage reward: {coverage_reward_cosine:.3f}")

    print("Coverage reward test passed!\n")


def test_completion_reward(file_path):
    """Test completion reward calculation."""
    print("Testing completion reward...")

    agents, paragraphs, stance, doc_topic_matrix = setup_test_data(file_path)
    shaper = RewardShaper()

    # Test with partial completion
    print("Testing completion reward with partial completion")
    completion_reward = shaper._completion_reward(stance)
    assert isinstance(completion_reward, float)
    assert 0 <= completion_reward <= 1
    print(f"Partial completion reward: {completion_reward:.3f}")

    # Test with empty stance
    print("Testing completion reward with empty stance")
    empty_stance = StanceMatrix(agents=agents, paragraphs=paragraphs)
    empty_completion = shaper._completion_reward(empty_stance)
    print(f"Empty stance completion: {empty_completion}")
    assert empty_completion == 0.0

    # Test with full completion
    print("Testing completion reward with full completion")
    full_stance = StanceMatrix(agents=agents, paragraphs=paragraphs)
    for agent in agents:
        for paragraph in paragraphs:
            full_stance.set_vote(agent.agent_id, paragraph.paragraph_id, "1")
    full_completion = shaper._completion_reward(full_stance)
    print(f"Full completion reward: {full_completion}")
    assert full_completion == 1.0
    print("Completion reward test passed!\n")


def test_content_reward(file_path):
    """Test content reward calculation."""
    print("Testing content reward...")

    agents, paragraphs, stance, doc_topic_matrix = setup_test_data(file_path)
    shaper = RewardShaper()

    # Test with mixed votes
    print("Testing content reward with mixed votes")
    content_reward = shaper._content_reward(stance)
    print(f"Mixed votes content reward: {content_reward:.3f}")
    assert isinstance(content_reward, float)
    assert 0 <= content_reward <= 1

    # Test with all positive votes
    print("Testing content reward with all positive votes")
    positive_stance = StanceMatrix(agents=agents, paragraphs=paragraphs)
    positive_stance.set_vote(agents[0].agent_id, paragraphs[0].paragraph_id, "1")
    positive_stance.set_vote(agents[1].agent_id, paragraphs[1].paragraph_id, "1")
    positive_content = shaper._content_reward(positive_stance)
    expected_positive = 2.0 / (len(agents) * len(paragraphs))  # 2 positive votes out of total cells
    print(f"All positive votes content reward: {positive_content:.3f}")
    assert abs(positive_content - expected_positive) < 0.01

    # Test with no votes
    print("Testing content reward with no votes")
    empty_stance = StanceMatrix(agents=agents, paragraphs=paragraphs)
    empty_content = shaper._content_reward(empty_stance)
    print(f"No votes content reward: {empty_content}")
    assert empty_content == 0.0

    print("Content reward test passed!\n")


def test_calculate_reward(file_path):
    """Test overall reward calculation."""
    print("Testing overall reward calculation...")

    agents, paragraphs, stance, doc_topic_matrix = setup_test_data(file_path)
    shaper = RewardShaper()

    # Test reward calculation
    print("Testing total reward")
    total_reward = shaper.calculate_reward(agents, stance, doc_topic_matrix)
    print(f"Total reward: {total_reward:.3f}")
    assert isinstance(total_reward, float)

    # Test individual components
    print("Test individual components")
    coverage = shaper._coverage_reward(agents, doc_topic_matrix)
    completion = shaper._completion_reward(stance)
    content = shaper._content_reward(stance)

    expected_total = (shaper.params["w1"] * coverage +
                      shaper.params["w2"] * completion +
                      shaper.params["w3"] * content)
    print(f"Component verification: {coverage:.3f}, {completion:.3f}, {content:.3f}")
    assert abs(total_reward - expected_total) < 0.001

    # Test with different weights
    print("Test with different weights")
    custom_params = {"w1": 0.6, "w2": 0.2, "w3": 0.2, "use_jsd": True}
    shaper_custom = RewardShaper(custom_params)
    custom_reward = shaper_custom.calculate_reward(agents, stance, doc_topic_matrix)
    print(f"Custom weights reward: {custom_reward:.3f}")
    assert custom_reward != total_reward  # Should be different with different weights

    print("Overall reward calculation test passed!\n")



def run_all_tests(file_path):
    """Run all RewardShaper tests."""
    print("RUNNING REWARD SHAPER TEST SUITE")
    test_reward_shaper_initialization()
    test_coverage_reward(file_path)
    test_completion_reward(file_path)
    test_content_reward(file_path)
    test_calculate_reward(file_path)
    print("ALL REWARD SHAPER TESTS PASSED!")


if __name__ == "__main__":
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    run_all_tests(file_path)

"""
Test module for Memory class and InteractionEvent class
"""

from environment.memory import Memory, InteractionEvent
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader


def setup_test_data(file_path):
    """Load real data using loaders."""
    # Load data
    paragraphs_loader = ParagraphsLoader(filepath=file_path)
    agents_loader = AgentsLoader(filepath=file_path)

    paragraphs = paragraphs_loader.load_all()
    agents = agents_loader.load_all()

    return agents, paragraphs


def test_interaction_event_creation():
    """Test InteractionEvent creation and methods."""
    print("Testing InteractionEvent creation...")

    # Test with all parameters
    print("Testing InteractionEvent with all parameters")
    event = InteractionEvent(
        agent_id=1,
        paragraph_id=5,
        vote=1,
        step=10,
        reward=0.8,
        timestamp=123456789.0
    )

    assert event.agent_id == 1
    assert event.paragraph_id == 5
    assert event.vote == "1"  # Should be converted to string
    assert event.step == 10
    assert event.reward == 0.8
    assert event.timestamp == 123456789.0
    print("✓ All parameters set correctly")

    # Test with minimal parameters
    print("Testing InteractionEvent with minimal parameters")
    event_minimal = InteractionEvent(agent_id=2, paragraph_id=3, vote=-1)
    assert event_minimal.agent_id == 2
    assert event_minimal.paragraph_id == 3
    assert event_minimal.vote == "-1"
    assert event_minimal.step is None
    assert event_minimal.reward is None
    assert isinstance(event_minimal.timestamp, float)  # Should be set to current time
    print("✓ Minimal parameters with auto timestamp")

    print("InteractionEvent creation test passed!\n")


def test_interaction_event_methods():
    """Test InteractionEvent methods."""
    print("Testing InteractionEvent methods...")

    event = InteractionEvent(
        agent_id=1,
        paragraph_id=5,
        vote="1",
        step=10,
        reward=0.8
    )

    # Test as_dict
    print("Testing as_dict method")
    event_dict = event.as_dict()
    expected_dict = {
        "agent_id": 1,
        "paragraph_id": 5,
        "vote": "1",
        "step": 10,
        "reward": 0.8
    }
    assert event_dict == expected_dict
    print("✓ as_dict method works correctly")

    # Test __repr__
    print("Testing __repr__ method")
    repr_str = repr(event)
    assert "Interaction" in repr_str
    assert "agent=1" in repr_str
    assert "para=5" in repr_str
    print(f"✓ __repr__ output: {repr_str}")

    # Test __str__
    print("Testing __str__ method")
    str_repr = str(event)
    assert "Agent 1" in str_repr
    assert "paragraph 5" in str_repr
    assert "step=10" in str_repr
    print(f"✓ __str__ output: {str_repr}")

    print("InteractionEvent methods test passed!\n")


def test_memory_initialization():
    """Test Memory initialization."""
    print("Testing Memory initialization...")

    memory = Memory()
    assert isinstance(memory.events, list)
    assert len(memory.events) == 0
    print("✓ Memory initialized with empty events list")

    print("Memory initialization test passed!\n")


def test_memory_log_event(file_path):
    """Test Memory log_event functionality."""
    print("Testing Memory log_event...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Test logging single event
    print("Testing single event logging")
    memory.log_event(
        agent_id=agents[0].agent_id,
        paragraph_id=paragraphs[0].paragraph_id,
        vote="1",
        step=1,
        reward=0.7
    )

    assert len(memory.events) == 1
    assert memory.events[0].agent_id == agents[0].agent_id
    assert memory.events[0].paragraph_id == paragraphs[0].paragraph_id
    assert memory.events[0].vote == "1"
    print("✓ Single event logged correctly")

    # Test logging multiple events
    print("Testing multiple event logging")
    memory.log_event(agents[1].agent_id, paragraphs[1].paragraph_id, "-1", 2, 0.3)
    memory.log_event(agents[0].agent_id, paragraphs[2].paragraph_id, "0", 3, 0.5)

    assert len(memory.events) == 3
    assert memory.events[1].vote == "-1"
    assert memory.events[2].vote == "0"
    print("✓ Multiple events logged correctly")

    print("Memory log_event test passed!\n")


def test_memory_get_events(file_path):
    """Test Memory get_events functionality."""
    print("Testing Memory get_events...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Add some events
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, "1", 1, 0.8)
    memory.log_event(agents[1].agent_id, paragraphs[1].paragraph_id, "-1", 2, 0.2)

    events = memory.get_events()
    assert len(events) == 2
    assert isinstance(events, list)
    assert all(isinstance(event, InteractionEvent) for event in events)
    print("✓ get_events returns correct list of InteractionEvent objects")

    print("Memory get_events test passed!\n")


def test_memory_agent_history(file_path):
    """Test Memory get_agent_history functionality."""
    print("Testing Memory get_agent_history...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Add events for different agents
    agent_1_id = agents[0].agent_id
    agent_2_id = agents[1].agent_id

    memory.log_event(agent_1_id, paragraphs[0].paragraph_id, "1", 1, 0.8)
    memory.log_event(agent_2_id, paragraphs[1].paragraph_id, "-1", 2, 0.2)
    memory.log_event(agent_1_id, paragraphs[2].paragraph_id, "0", 3, 0.5)
    memory.log_event(agent_2_id, paragraphs[0].paragraph_id, "1", 4, 0.9)

    # Test agent 1 history
    agent_1_history = memory.get_agent_history(agent_1_id)
    assert len(agent_1_history) == 2
    assert all(event.agent_id == agent_1_id for event in agent_1_history)
    print(f"✓ Agent {agent_1_id} has {len(agent_1_history)} events")

    # Test agent 2 history
    agent_2_history = memory.get_agent_history(agent_2_id)
    assert len(agent_2_history) == 2
    assert all(event.agent_id == agent_2_id for event in agent_2_history)
    print(f"✓ Agent {agent_2_id} has {len(agent_2_history)} events")

    # Test non-existent agent
    empty_history = memory.get_agent_history(999)
    assert len(empty_history) == 0
    print("✓ Non-existent agent returns empty history")

    print("Memory get_agent_history test passed!\n")


def test_memory_paragraph_history(file_path):
    """Test Memory get_paragraph_history functionality."""
    print("Testing Memory get_paragraph_history...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Add events for different paragraphs
    para_1_id = paragraphs[0].paragraph_id
    para_2_id = paragraphs[1].paragraph_id

    memory.log_event(agents[0].agent_id, para_1_id, "1", 1, 0.8)
    memory.log_event(agents[1].agent_id, para_2_id, "-1", 2, 0.2)
    memory.log_event(agents[0].agent_id, para_1_id, "0", 3, 0.5)  # Same paragraph, different vote
    memory.log_event(agents[1].agent_id, para_1_id, "1", 4, 0.9)  # Different agent, same paragraph

    # Test paragraph 1 history
    para_1_history = memory.get_paragraph_history(para_1_id)
    assert len(para_1_history) == 3
    assert all(event.paragraph_id == para_1_id for event in para_1_history)
    print(f"✓ Paragraph {para_1_id} has {len(para_1_history)} events")

    # Test paragraph 2 history
    para_2_history = memory.get_paragraph_history(para_2_id)
    assert len(para_2_history) == 1
    assert all(event.paragraph_id == para_2_id for event in para_2_history)
    print(f"✓ Paragraph {para_2_id} has {len(para_2_history)} events")

    # Test non-existent paragraph
    empty_history = memory.get_paragraph_history(999)
    assert len(empty_history) == 0
    print("✓ Non-existent paragraph returns empty history")

    print("Memory get_paragraph_history test passed!\n")


def test_memory_last_votes_stance_matrix(file_path):
    """Test Memory last_votes_stance_matrix functionality."""
    print("Testing Memory last_votes_stance_matrix...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Add some events, including vote changes
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, "1", 1, 0.8)
    memory.log_event(agents[1].agent_id, paragraphs[0].paragraph_id, "-1", 2, 0.2)
    memory.log_event(agents[0].agent_id, paragraphs[1].paragraph_id, "0", 3, 0.5)
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, "-1", 4, 0.3)  # Vote change

    # Build stance matrix
    stance_df = memory.last_votes_stance_matrix(agents, paragraphs)

    # Test structure
    expected_index = [f"a{a.agent_id}" for a in agents]
    expected_columns = [f"p{p.paragraph_id}" for p in paragraphs]
    assert list(stance_df.index) == expected_index
    assert list(stance_df.columns) == expected_columns
    print("✓ Stance matrix has correct structure")

    # Test last votes (should show most recent votes)
    assert stance_df.loc[f"a{agents[0].agent_id}", f"p{paragraphs[0].paragraph_id}"] == "-1"  # Last vote
    assert stance_df.loc[f"a{agents[1].agent_id}", f"p{paragraphs[0].paragraph_id}"] == "-1"
    assert stance_df.loc[f"a{agents[0].agent_id}", f"p{paragraphs[1].paragraph_id}"] == "0"
    print("✓ Stance matrix shows correct last votes")

    # Test unknown votes
    assert stance_df.loc[f"a{agents[1].agent_id}", f"p{paragraphs[1].paragraph_id}"] == "?"
    print("✓ Unknown votes marked with '?'")

    print("Memory last_votes_stance_matrix test passed!\n")


def test_memory_reset(file_path):
    """Test Memory reset functionality."""
    print("Testing Memory reset...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Add some events
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, "1", 1, 0.8)
    memory.log_event(agents[1].agent_id, paragraphs[1].paragraph_id, "-1", 2, 0.2)

    assert len(memory.events) == 2
    print("✓ Memory has events before reset")

    # Reset memory
    memory.reset()
    assert len(memory.events) == 0
    assert len(memory) == 0
    print("✓ Memory cleared after reset")

    print("Memory reset test passed!\n")


def test_memory_len(file_path):
    """Test Memory __len__ functionality."""
    print("Testing Memory __len__...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Test empty memory
    assert len(memory) == 0
    print("✓ Empty memory length is 0")

    # Add events and test length
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, "1", 1, 0.8)
    assert len(memory) == 1

    memory.log_event(agents[1].agent_id, paragraphs[1].paragraph_id, "-1", 2, 0.2)
    assert len(memory) == 2
    print("✓ Memory length updates correctly")

    print("Memory __len__ test passed!\n")


def test_memory_edge_cases(file_path):
    """Test Memory edge cases and data types."""
    print("Testing Memory edge cases...")

    agents, paragraphs = setup_test_data(file_path)
    memory = Memory()

    # Test different vote types
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, 1, 1, 0.8)  # int
    memory.log_event(agents[0].agent_id, paragraphs[1].paragraph_id, "1", 2, 0.8)  # string
    memory.log_event(agents[0].agent_id, paragraphs[2].paragraph_id, -1, 3, 0.8)  # negative int

    # All should be converted to strings
    assert memory.events[0].vote == "1"
    assert memory.events[1].vote == "1"
    assert memory.events[2].vote == "-1"
    print("✓ Different vote types handled correctly")

    # Test missing optional parameters
    memory.log_event(agents[0].agent_id, paragraphs[0].paragraph_id, "0")
    last_event = memory.events[-1]
    assert last_event.step is None
    assert last_event.reward is None
    assert isinstance(last_event.timestamp, float)
    print("✓ Missing optional parameters handled correctly")

    print("Memory edge cases test passed!\n")


def run_all_tests(file_path):
    """Run all Memory tests."""
    print("RUNNING MEMORY TEST SUITE")

    test_interaction_event_creation()
    test_interaction_event_methods()
    test_memory_initialization()
    test_memory_log_event(file_path)
    test_memory_get_events(file_path)
    test_memory_agent_history(file_path)
    test_memory_paragraph_history(file_path)
    test_memory_last_votes_stance_matrix(file_path)
    test_memory_reset(file_path)
    test_memory_len(file_path)
    test_memory_edge_cases(file_path)

    print("ALL MEMORY TESTS PASSED!")


if __name__ == "__main__":
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    run_all_tests(file_path)
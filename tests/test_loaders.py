from environment.collaborators import Agent, LLMAgentWithTopics
from environment.paragraphs import Paragraph, ParagraphWithTopics
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader, StanceLoader
from environment.topic_model import compute_paragraphs_topic_matrix


def test_paragraphs_loader():
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    # Directory containing events.json, agents.json, paragraphs.json
    loader = ParagraphsLoader(filepath=file_path)
    paragraphs = loader.load_all()
    assert all(isinstance(p, Paragraph) for p in paragraphs), "Not all paragraphs are converted."
    print("Loaded:", len(paragraphs), "paragraphs.")

    # Topic distribution

    texts = [p.text for p in paragraphs]
    doc_topic_matrix, best_k, topic_keywords = compute_paragraphs_topic_matrix(texts, k_range=range(2, 5))
    paragraphs_topical = loader.attach_topics(paragraphs=paragraphs, topic_matrix=doc_topic_matrix)
    assert all(
        isinstance(p, ParagraphWithTopics) for p in paragraphs_topical), "Not all paragraphs with topic are converted."
    for p in paragraphs_topical:
        print(p)
    print(f"Topic modeling pipeline completed and topics attached to all paragraphs with {best_k} topics")


def test_agents_loader():
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    # Directory containing events.json, agents.json, paragraphs.json
    agents_loader = AgentsLoader(filepath=file_path)
    agents = agents_loader.load_all()
    assert all(isinstance(a, LLMAgentWithTopics) for a in agents), "Not all agents loaded as LLMAgentWithTopics."
    print("Loaded:", len(agents), "agents:")
    for a in agents:
        print(a)


def test_events_loader():
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    # Directory containing events.json, agents.json, paragraphs.json

    # Load paragraphs and agents
    paragraphs = ParagraphsLoader(filepath=file_path).load_all()
    agents = AgentsLoader(filepath=file_path).load_all()

    # Load stance matrix from events
    events_loader = EventsLoader(filepath=file_path)
    stance_df = events_loader.load_all(agents, paragraphs)
    print(stance_df)

    # Check initiation of stance matrix instance from events df:
    from environment.stance import StanceMatrix
    stance = StanceMatrix.from_existing(agents, paragraphs, stance_df)
    print("StanceMatrix:\n", stance.matrix)


def test_env_total_loading():
    from environment.stance import StanceMatrix

    # 2. Loaders - Load all paragraphs (items), agents (users) and events (initial stance)
    # Load paragraphs and agents
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    # Directory containing events.json, agents.json, paragraphs.json    
    paragraphs_loader = ParagraphsLoader(filepath=file_path)
    agents_loader = AgentsLoader(filepath=file_path)
    events_loader = EventsLoader(filepath=file_path)

    ## 2.1 Paragraphs loading
    assert paragraphs_loader is not None, 'A ParagraphsLoader is required.'
    paragraphs_loader = paragraphs_loader
    paragraphs = paragraphs_loader.load_all()  # List[Paragraph]
    num_paragraphs = len(paragraphs)  # m - number of total paragraphs
    paragraph_ids = [p.paragraph_id for p in paragraphs]

    ## 2.2 Agents loading
    assert agents_loader is not None, 'An AgentsLoader is required.'
    agents_loader = agents_loader
    agents = agents_loader.load_all()  # List[Agent]
    num_agents = len(agents)  # n - number of total agents
    agent_ids = [a.agent_id for a in agents]

    ## 2.3 Initial stance loading (default: all "?")
    if events_loader is not None:
        # Option 1 - An EventsLoader is given for initiating a stance from previous event list.
        events_loader = events_loader
        events_df = events_loader.load_all(agents=agents, paragraphs=paragraphs)
        stance = StanceMatrix.from_existing(agents=agents, paragraphs=paragraphs, matrix=events_df)
    else:
        # Option 3 - No loader and not a starting stance matrix, then all preferences are unknown
        stance = StanceMatrix(agents=agents, paragraphs=paragraphs)  # all unknown

    print("Stance matrix created successfully")

    # 2.4 Attach topic vectors
    ## Paragraphs
    print("Extracting texts...")
    texts = [p.text for p in paragraphs]
    print("Starting topic modeling...")
    try:
        doc_topic_matrix, best_k, topic_keywords = compute_paragraphs_topic_matrix(texts, k_range=range(3, 21))
        print(f"Topic modeling completed with k={best_k}")
    except Exception as e:
        print(f"Error in topic modeling: {e}")
        return
    print("Attaching topics to paragraphs...")
    paragraphs = paragraphs_loader.attach_topics(paragraphs, doc_topic_matrix)

    print("Updating agent topic vectors...")
    LLMAgentWithTopics.update_all_topic_vectors(agents=agents, doc_topic_matrix=doc_topic_matrix, stance_matrix=stance)

    print("\nSample paragraph:", paragraphs[0])
    print("Sample agent topical profile:", agents[0].topic_profile_vector)

    print("\nEnvironment loading pipeline successful:")
    print("Paragraphs:\n")
    for p in paragraphs:
        print(p)
    print("Agents:\n")
    for a in agents:
        print(a)
    print("Stance Matrix:\n")
    print(stance)


def test_stance_loader():
    file_path = r"C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"

    # Load paragraphs and agents
    paragraphs = ParagraphsLoader(filepath=file_path).load_all()
    agents = AgentsLoader(filepath=file_path).load_all()

    # Load stance matrix from events
    stance_loader = StanceLoader(agents=agents, paragraphs=paragraphs, sparsity=0.5, seed=42)
    # Empty
    stance_df = stance_loader.load_empty_matrix()
    print(stance_df)

    # Empty
    stance_df = stance_loader.load_random()
    print(stance_df)


def run_all_tests():
    # print("Testing paragraphs loader...")
    # test_paragraphs_loader()
    # print("Testing agents loader...")
    # test_agents_loader()
    # print("Testing events loader...")
    # test_events_loader()
    # print("Testing environment loading pipeline ...")
    # test_env_total_loading()
    print("Testing stance loading pipeline ...")
    test_stance_loader()


if __name__ == "__main__":
    run_all_tests()

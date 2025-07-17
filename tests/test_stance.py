import pandas as pd
from environment.collaborators import Agent
from environment.paragraphs import Paragraph
from environment.stance import StanceMatrix


def setup_agents_paragraphs():
    agents = [
        Agent(agent_id=1),
        Agent(agent_id=2)
    ]
    paragraphs = [
        Paragraph(text="Para1", paragraph_id=1, name="P1"),
        Paragraph(text="Para2", paragraph_id=2, name="P2"),
        Paragraph(text="Para3", paragraph_id=3, name="P3"),
    ]
    return agents, paragraphs


def test_initialization():
    agents, paragraphs = setup_agents_paragraphs()
    stance = StanceMatrix(agents, paragraphs)
    print("Initial stance matrix:\n", stance)
    assert stance.matrix.shape == (2, 3), "The matrix is not in shape 2 X 3"
    assert all((stance.matrix == "?").all()), "The matrix is not full with ?"


def test_set_matrix():
    agents, paragraphs = setup_agents_paragraphs()
    stance = StanceMatrix(agents, paragraphs)
    # External stance matrix
    data = {'p1': ["1", "?"], 'p2': ["-1", "1"], 'p3': ["0", "?"]}
    df = pd.DataFrame(data, index=["a1", "a2"])
    stance.set_matrix(df)
    assert stance.get_vote(1, 1) == "1"
    assert stance.get_vote(2, 2) == "1"
    assert stance.get_vote(2, 3) == "?"
    print("Stance set_matrix works.")


def test_from_existing():
    agents, paragraphs = setup_agents_paragraphs()
    data = {'p1': ["1", "?"], 'p2': ["-1", "1"], 'p3': ["0", "?"]}
    df = pd.DataFrame(data, index=["a1", "a2"])
    stance = StanceMatrix.from_existing(agents, paragraphs, df)
    assert stance.get_vote(1, 1) == "1"
    assert stance.get_vote(2, 2) == "1"
    assert stance.get_vote(2, 3) == "?"
    print("Stance from_existing works.")


def test_set_and_get_vote():
    agents, paragraphs = setup_agents_paragraphs()
    stance = StanceMatrix(agents, paragraphs)
    stance.set_vote(agent_id=1, paragraph_id=2, vote=1)
    assert stance.get_vote(agent_id=1, paragraph_id=2) == "1", "The set/ get is not working"
    stance.set_vote(agent_id=2, paragraph_id=1, vote="-1")
    assert stance.get_vote(2, 1) == "-1", "The set/ get is not working with set of str"
    print("Updated stance matrix:\n", stance)


def test_get_unknown_and_known_paragraphs():
    agents, paragraphs = setup_agents_paragraphs()
    stance = StanceMatrix(agents, paragraphs)
    stance.set_vote(1, 2, "1")
    stance.set_vote(1, 3, "0")
    unknowns = stance.get_unknown_paragraphs(agent_id=1)
    known = stance.get_known_paragraphs(agent_id=1)
    unknown_ids = [p.paragraph_id for p in unknowns]
    known_ids = [p.paragraph_id for p in known]
    assert 1 in unknown_ids
    assert 2 in known_ids and 3 in known_ids
    assert len(unknowns) + len(known) == 3


def test_is_complete():
    agents, paragraphs = setup_agents_paragraphs()
    stance = StanceMatrix(agents, paragraphs)
    # Fill all votes
    for agent in agents:
        for para in paragraphs:
            stance.set_vote(agent.agent_id, para.paragraph_id, "1")
    assert stance.is_complete()
    # Remove one vote
    stance.matrix.loc["a1", "p2"] = "?"
    assert not stance.is_complete()


def run_all_tests():
    print("Testing initialization...")
    test_initialization()
    print("Testing set/get vote...")
    test_set_and_get_vote()
    print("Testing get_unknown/get_known_paragraphs...")
    test_get_unknown_and_known_paragraphs()
    print("Testing is_complete...")
    test_is_complete()
    print("Testing set_matrix...")
    test_set_matrix()
    print("Testing from_existing...")
    test_from_existing()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()

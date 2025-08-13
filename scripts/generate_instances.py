import json
from pathlib import Path
from random import random
import random
import os
import pandas as pd
import numpy as np
from contextlib import redirect_stdout

# RL-RS
from environment.collaborators import LLMAgent, LLMAgentWithTopics
from environment.stance import StanceMatrix
from environment.loaders import AgentsLoader, ParagraphsLoader


def sample_profiles(df_prepared: pd.DataFrame, num_samples: int):
    """
    Sample agent profiles from the demographic data with weighted random sampling.

    Args:
        df_prepared: DataFrame with normalized proportions.
        num_samples: Number of profiles to sample.

    Returns:
        List of profile dictionaries.
    """
    sampled_rows = df_prepared.sample(
        n=num_samples,
        weights='Proportion',
        replace=True  # Allows duplicates
    ).reset_index(drop=True)

    profiles = sampled_rows.apply(
        lambda row: {
            'Sex': row['Sex'],
            'Age': row['Age'],
            'Educational attainment': row['Educational attainment']
        }, axis=1
    ).tolist()

    return profiles


def create_community_agents(df_prepared, num_agents, output_file, seed=42):
    # Sample profiles
    sampled_profiles = sample_profiles(df_prepared=df_prepared, num_samples=num_agents)
    # Create community
    agents = LLMAgent.create_community(sampled_profiles, topic="climate change policy")
    # Saved as JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    agents_data = [
        {
            "agent_id": agent.agent_id,
            "profile": agent.profile,
            "topic": agent.topic,
            "topic_position": agent.topic_position
        } for agent in agents
    ]
    with open(output_file, 'w') as f:
        json.dump(agents_data, f, indent=2)

    print(f"Saved {num_agents} agents to {output_file}")


def create_suggested_paragraphs(input_file='datasets/processed/grouped_proposals_by_sentiment.json',
                                output_file='datasets/processed/selected_paragraphs.json',
                                num_paragraphs=25):
    # Load the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Select categories
    categories = ['minimal_acknowledgment', 'balanced_approach', 'supportive_measures']

    # Collect all paragraphs from selected categories
    all_paragraphs = []
    for category in categories:
        all_paragraphs.extend(data.get(category, []))

    # Randomly sample 25 paragraphs
    selected_paragraphs = random.sample(all_paragraphs, num_paragraphs)

    # Create output in the format of paragraphs.json
    output = [
        {
            "paragraph_id": i + 1,
            "text": para['text'],
            "name": f"p{i + 1}"
        }
        for i, para in enumerate(selected_paragraphs)
    ]

    # Save to output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {num_paragraphs} paragraphs to {output_file}")


def create_full_stance_matrix(
        agents_file: str,
        paragraphs_file: str,
        num_agents: int,
        num_paragraphs: int,
        output_file: str,
        seed: int = 42) -> StanceMatrix:
    """
    Generate a fully populated stance matrix for given agents and paragraphs.

    Args:
        agents_file: Path to the JSON file containing agent data.
        paragraphs_file: Path to the JSON file containing paragraph data.
        num_agents: Number of agents to load (e.g., 20).
        num_paragraphs: Number of paragraphs to load (e.g., 25).
        output_file: Path to save the generated stance matrix JSON.
        seed: Random seed for reproducibility.

    Returns:
        StanceMatrix: Fully populated stance matrix.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load agents and paragraphs
    agents_loader = AgentsLoader(filepath=agents_file, num_agents=num_agents)
    paragraphs_loader = ParagraphsLoader(filepath=paragraphs_file, num_paragraphs=num_paragraphs)
    agents = agents_loader.load_all()
    paragraphs = paragraphs_loader.load_all()

    # Initialize empty stance matrix
    stance_matrix = StanceMatrix(agents=agents, paragraphs=paragraphs)

    # Iterate over all agent-paragraph pairs
    for agent in sorted(agents, key=lambda x: x.agent_id):
        for paragraph in sorted(paragraphs, key=lambda x: x.paragraph_id):
            print(f"Voting process agent a{agent.agent_id} on paragraph {paragraph.paragraph_id}")
            # Get past votes for this agent
            past_votes = [
                (p.text, stance_matrix.get_vote(agent.agent_id, p.paragraph_id))
                for p in paragraphs
                if stance_matrix.get_vote(agent.agent_id, p.paragraph_id) != "?"
            ]
            # Get vote for the current paragraph
            vote = agent.get_vote_with_consistency_summary(
                current_paragraph=paragraph.text,
                past_votes=past_votes
            )
            # Set the vote in the stance matrix
            stance_matrix.set_vote(agent_id=agent.agent_id, paragraph_id=paragraph.paragraph_id, vote=vote)

    # Save the stance matrix to a JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    stance_matrix.save_to_json(output_file)

    return stance_matrix


def create_mock_stance_example(
        agents_file: str,
        paragraphs_file: str,
        output_file: str,
        log_file: str = "mock_stance_log.txt",
        num_agents: int = 5,
        num_paragraphs: int = 5,
        seed: int = 123):
    """
    Create a reduced-size stance matrix (e.g., 5x5) for documentation/demo purposes,
    with printed prompts and votes for each cell.
    All console output is also saved to a log file.
    """
    random.seed(seed)
    np.random.seed(seed)

    agents_loader = AgentsLoader(filepath=agents_file, num_agents=num_agents)
    paragraphs_loader = ParagraphsLoader(filepath=paragraphs_file, num_paragraphs=num_paragraphs)
    agents = agents_loader.load_all()
    paragraphs = paragraphs_loader.load_all()

    stance_matrix = StanceMatrix(agents=agents, paragraphs=paragraphs)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f):  # Redirect all prints inside to the log file
            for agent in agents:
                for paragraph in paragraphs:
                    past_votes = [
                        (p.text, stance_matrix.get_vote(agent.agent_id, p.paragraph_id))
                        for p in paragraphs
                        if stance_matrix.get_vote(agent.agent_id, p.paragraph_id) != "?"
                    ]
                    vote = agent.get_vote_with_consistency_summary(
                        current_paragraph=paragraph.text,
                        past_votes=past_votes,
                        log_prompts=True  # All prompts and votes printed here
                    )
                    stance_matrix.set_vote(agent.agent_id, paragraph.paragraph_id, vote)

    # Save stance matrix JSON
    stance_matrix.save_to_json(output_file)
    print(f"Saved stance matrix to {output_file}")
    print(f"Saved log to {log_file}")
    return stance_matrix


if __name__ == "__main__":
    base_path = Path("datasets/processed")
    file_path = base_path / "prepared_demographics.json"
    with open(file_path, "r") as f:
        df_prepared = json.load(f)
    df_prepared = pd.DataFrame(df_prepared)

    # Creating a pool of 1000 agents
    create_community_agents(df_prepared=df_prepared, num_agents=1000,
                            output_file="datasets/processed/agents_instances.json")

    # Creating an instance 1 = of 20 agents and 25 paragraphs
    create_community_agents(df_prepared=df_prepared, num_agents=20,
                            output_file="datasets/instances/instance1/agents.json")
    create_suggested_paragraphs(num_paragraphs=40,
                                output_file="datasets/instances/instance1/paragraphs.json")
    stance_matrix = create_full_stance_matrix(
        agents_file="datasets/instances/instance1",
        paragraphs_file="datasets/instances/instance1",
        num_agents=20,
        num_paragraphs=25,
        output_file="datasets/instances/instance1/stance.json",
        seed=42
    )
    print(stance_matrix)

    # Creating an instance 2 = of 20 agents and 50 paragraphs
    num_agents = 20
    num_paragraphs = 50
    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    create_community_agents(df_prepared=df_prepared, num_agents=num_agents,
                            output_file="datasets/instances/instance2/agents.json")
    create_suggested_paragraphs(num_paragraphs=num_paragraphs,
                                output_file="datasets/instances/instance2/paragraphs.json")
    stance_matrix = create_full_stance_matrix(
        agents_file="datasets/instances/instance2",
        paragraphs_file="datasets/instances/instance2",
        num_agents=num_agents,
        num_paragraphs=num_paragraphs,
        output_file="datasets/instances/instance2/stance.json",
        seed=seed
    )

    # Report Example
    create_mock_stance_example(
        agents_file="datasets/instances/instance2",
        paragraphs_file="datasets/instances/instance2",
        output_file="datasets/examples/mock_stance_5x5.json",
        log_file="datasets/examples/mock_stance_5x5_log.txt",
        num_agents=5,
        num_paragraphs=5
    )

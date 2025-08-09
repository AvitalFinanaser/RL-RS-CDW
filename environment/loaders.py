import os
import json
import pandas as pd
from typing import List, Optional
#
import numpy as np
#
from environment.paragraphs import Paragraph, ParagraphWithTopics
from environment.collaborators import Agent, LLMAgent, LLMAgentWithTopics


class ParagraphsLoader:
    """
    Loads Paragraph objects from a JSON file.
    """

    def __init__(self, filepath: str, num_paragraphs: int = 0):
        self.filepath = filepath
        self.num_paragraphs = num_paragraphs

    def load_all(self) -> List[Paragraph]:
        """
        Loads from a JSON file to create Paragraph objects (with paragraph_id, text, name)
        :return: List of paragraphs objects (List[Paragraph])
        """
        json_path = os.path.join(self.filepath, "paragraphs.json")
        with open(json_path, "r", encoding="utf-8") as f:
            paragraphs_data = json.load(f)
        # Each element is a dict with keys: paragraph_id, text, name
        paragraphs = [
            Paragraph(
                text=entry["text"],
                paragraph_id=entry["paragraph_id"],
                name=entry["name"]
            )
            for entry in paragraphs_data
        ]
        if self.num_paragraphs > 0:
            return paragraphs[:self.num_paragraphs]
        return paragraphs

    @staticmethod
    def attach_topics(paragraphs: List[Paragraph], topic_matrix: np.ndarray) -> List[ParagraphWithTopics]:
        """
        Given base Paragraphs and an n x k topic_matrix, returns new list of ParagraphWithTopics.
        """
        return [
            ParagraphWithTopics(
                text=p.text,
                paragraph_id=p.paragraph_id,
                name=p.name,
                topic_vector=topic_matrix[i]
            )
            for i, p in enumerate(paragraphs)
        ]


class AgentsLoader:
    def __init__(self, filepath: str, num_agents: int = 0):
        self.filepath = filepath
        self.num_agents = num_agents

    def load_all(self) -> List[LLMAgentWithTopics]:
        # Load Agents
        json_path = os.path.join(self.filepath, "agents.json")
        with open(json_path, "r", encoding="utf-8") as f:
            agents_data = json.load(f)

        # Each element is a dict with keys: agent_id, profile, topic, topic_position
        agents = [
            LLMAgentWithTopics(
                agent_id=entry["agent_id"],
                profile=entry["profile"],
                topic=entry["topic"],
                topic_position=entry["topic_position"],
                topic_profile_vector=None  # Always None on load
            )
            for entry in agents_data
        ]
        if self.num_agents > 0:
            return agents[:self.num_agents]
        return agents


class EventsLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_all(self, agents: List[Agent], paragraphs: List[Paragraph]) -> pd.DataFrame:
        """
        Loads the events file and returns a stance matrix DataFrame suitable for StanceMatrix.
        Returns:
            DataFrame: index = agent_ids (with 'a'), columns = paragraph_ids (with 'p'), values = last vote or "?".
        """
        # 1. Build empty DataFrame, all "?"
        index = [f"a{a.agent_id}" for a in agents]
        columns = [f"p{p.paragraph_id}" for p in paragraphs]
        stance_df = pd.DataFrame("?", index=index, columns=columns)

        # 2. Load all events and record last vote
        json_path = os.path.join(self.filepath, "events.json")
        with open(json_path, "r", encoding="utf-8") as f:
            events = json.load(f)

        # Build dict to store latest vote for (agent, paragraph)
        last_votes = {}
        for event in events:
            key = (event["agent_id"], event["paragraph_id"])
            last_votes[key] = str(event["vote"])

        # 3. Fill in the matrix
        for (agent_id, paragraph_id), vote in last_votes.items():
            agent_key = f"a{agent_id}"
            para_key = f"p{paragraph_id}"
            if agent_key in stance_df.index and para_key in stance_df.columns:
                stance_df.loc[agent_key, para_key] = vote

        return stance_df


class StanceLoader:
    """
    Initializes a stance matrix for a given set of agents and paragraphs.
    Can fill the matrix with a random values with a given completion rate (fraction of known votes).
    """

    def __init__(
            self,
            agents: List[Agent] = None,
            paragraphs: List[Paragraph] = None,
            sparsity: float = 0.3,
            seed: int = 42
    ):
        """
        Initialize StanceLoader with optional filepath for events, agents, paragraphs, sparsity, and seed.
        Args:
            agents: List of Agent objects (optional, required if no filepath).
            paragraphs: List of Paragraph objects (optional, required if no filepath).
            sparsity: Fraction of unknown entries (0 to 1, e.g., 0.7 for 30% completion).
                     If None, uses events or all unknown.
            seed: Random seed for generating random stance matrix.
        """
        self.agents = agents
        self.paragraphs = paragraphs
        self.sparsity = sparsity
        self.seed = seed

    def load_random(self):
        """
        Initializes the stance matrix with random votes (at completion_rate sparsity).
        Returns a StanceMatrix object.
        """
        from environment.stance import StanceMatrix

        rng = np.random.RandomState(self.seed)
        stance = StanceMatrix(self.agents, self.paragraphs)

        n_agents = len(self.agents)
        n_paragraphs = len(self.paragraphs)
        total = n_agents * n_paragraphs
        n_known = int(self.sparsity * total)

        # Select random entries to fill
        indices = rng.permutation(total)[:n_known]
        for idx in indices:
            agent_idx = idx // n_paragraphs
            para_idx = idx % n_paragraphs
            agent_id = self.agents[agent_idx].agent_id
            paragraph_id = self.paragraphs[para_idx].paragraph_id
            vote = str(rng.choice([-1, 0, 1]))
            stance.set_vote(agent_id, paragraph_id, vote)
        return stance

    def load_empty_matrix(self):
        """
        Returns an empty StanceMatrix object with all values marked as un-known.
        """
        from environment.stance import StanceMatrix
        return StanceMatrix(agents=self.agents, paragraphs=self.paragraphs)

import os
import json
import pandas as pd
from typing import List
#
import numpy as np
#
from environment.paragraphs import Paragraph, ParagraphWithTopics
from environment.collaborators import Agent, LLMAgent, LLMAgentWithTopics


class ParagraphsLoader:
    """
    Loads Paragraph objects from a JSON file.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_all(self) -> List[Paragraph]:
        """
        Loads from a JSON file to create Paragraph objects (with paragraph_id, text, name)
        :return: List of paragraphs objects (List[Paragraph])
        """
        json_path = os.path.join(self.filepath, "paragraphs.json")
        with open(json_path, "r", encoding="utf-8") as f:
            paragraphs_data = json.load(f)
        # Each element is a dict with keys: paragraph_id, text, name
        return [
            Paragraph(
                text=entry["text"],
                paragraph_id=entry["paragraph_id"],
                name=entry["name"]
            )
            for entry in paragraphs_data
        ]

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
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_all(self) -> List[LLMAgentWithTopics]:
        # Load Agents
        json_path = os.path.join(self.filepath, "agents.json")
        with open(json_path, "r", encoding="utf-8") as f:
            agents_data = json.load(f)

        # Each element is a dict with keys: agent_id, profile, topic, topic_position
        return [
            LLMAgentWithTopics(
                agent_id=entry["agent_id"],
                profile=entry["profile"],
                topic=entry["topic"],
                topic_position=entry["topic_position"],
                topic_profile_vector=None  # Always None on load
            )
            for entry in agents_data
        ]


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
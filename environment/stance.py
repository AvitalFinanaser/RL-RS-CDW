from typing import List
import numpy as np
import pandas as pd
from environment.collaborators import Agent
from environment.paragraphs import Paragraph


class StanceMatrix:
    """
    The stance matrix object records the last preference vote each agent has on all paragraphs.
    Methods support updating votes, checking completeness, extracting data subsets, and derived metrics.
    """

    def __init__(self, agents: List[Agent], paragraphs: List[Paragraph]):
        self.agents = agents
        self.agents_ids = [a.agent_id for a in self.agents]
        self.paragraphs = paragraphs
        self.paragraphs_ids = [p.paragraph_id for p in paragraphs]

        # Initialize the stance matrix with unknown values
        self.matrix = pd.DataFrame("?", index=[f"a{a.agent_id}" for a in agents],
                                   columns=[f"p{p.paragraph_id}" for p in paragraphs])

    @classmethod
    def from_existing(cls, agents: List[Agent], paragraphs: List[Paragraph], matrix: pd.DataFrame):
        """
        Alternative constructor to build a StanceMatrix from an existing instance.
        DataFrame index must be agent IDs (with 'a' prefix), columns are paragraph IDs (with 'p' prefix).
        """
        obj = cls(agents, paragraphs)
        assert set(matrix.index) == set(obj.matrix.index), "Row index mismatch"
        assert set(matrix.columns) == set(obj.matrix.columns), "Column index mismatch"
        obj.matrix = matrix.copy()
        return obj

    def set_vote(self, agent_id, paragraph_id, vote):
        """
        Record preference vote of agent on paragraph.
        :param agent_id: agent a's identifier.
        :param paragraph_id: paragraph p's identifier.
        :param vote: preference of agent a on p.
        """
        assert str(vote) in ["-1", "0", "1"], "Vote must be in-favor (1), against (-1) or neutral (0)."
        assert agent_id in self.agents_ids, "Agent id is unknown."
        assert paragraph_id in self.paragraphs_ids, "Paragraph id is unknown."

        self.matrix.loc[f"a{agent_id}", f"p{paragraph_id}"] = str(vote)

    def set_matrix(self, matrix: pd.DataFrame):
        """
        Set the stance matrix explicitly after initialization.
        Note that in ndex/columns must match.
        :param matrix: A different stance matrix for the existing (agents, paragraphs).
        """
        assert set(matrix.index) == set(self.matrix.index), "Row index mismatch"
        assert set(matrix.columns) == set(self.matrix.columns), "Column index mismatch"
        self.matrix = matrix.copy()

    def get_vote(self, agent_id, paragraph_id):
        """
        Retrieve preference vote of agent on paragraph.
        :param agent_id: agent a's identifier.
        :param paragraph_id: paragraph p's identifier.
        :return: Vote of the agent on the paragraph.
        """
        assert agent_id in self.agents_ids, "Agent id is unknown."
        assert paragraph_id in self.paragraphs_ids, "Paragraph id is unknown."

        return self.matrix.loc[f"a{agent_id}", f"p{paragraph_id}"]

    def get_unknown_paragraphs(self, agent_id):
        """
        Agent-specific subset of unknown stance paragraphs.
        :param agent_id: An identifier of an agent.
        :return: A list of paragraphs objects that the given agent has unknown preference (?).
        """
        assert agent_id in self.agents_ids, "Agent id is unknown."
        col_names = self.matrix.columns[self.matrix.loc[f"a{agent_id}"] == "?"].tolist()
        para_ids = [int(name[1:]) for name in col_names]  # remove 'p' prefix
        unknowns = [p for p in self.paragraphs if p.paragraph_id in para_ids]
        return unknowns

    def get_known_paragraphs(self, agent_id):
        """
        Agent-specific subset of known stance paragraphs.
        :param agent_id: An identifier of an agent.
        :return: A list of paragraphs objects that the given agent has known preference (1/-1/0).
        """
        assert agent_id in self.agents_ids, "Agent id is unknown."
        # Select those with votes not '?'
        agent_pref = self.matrix.loc[f"a{agent_id}"]
        col_names = agent_pref[agent_pref.isin(["-1", "0", "1"])].index.tolist()
        para_ids = [int(name[1:]) for name in col_names]
        known = [p for p in self.paragraphs if p.paragraph_id in para_ids]
        return known

    def is_complete(self):
        """
        Boolean indicator for completion of preferences of all agents on all paragraphs.
        :return: True if there are no '?' values left (matrix is fully voted).
        """
        return not (self.matrix == "?").any().any()

    def to_numerical(self) -> np.ndarray:
        """Convert stance matrix to numerical format for RL agent."""
        numerical = np.zeros((len(self.agents), len(self.paragraphs)))
        for i, agent in enumerate(self.agents):
            for j, paragraph in enumerate(self.paragraphs):
                vote = self.get_vote(agent.agent_id, paragraph.paragraph_id)
                numerical[i, j] = 0.0 if vote == "?" else float(vote)
        return numerical

    def __str__(self):
        return str(self.matrix)
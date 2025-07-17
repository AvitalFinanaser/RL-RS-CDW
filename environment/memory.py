"""
The history of agents interactions with the recommender.
"""

import time
import pandas as pd
from typing import List, Dict, Optional, Any, Union


class InteractionEvent:
    def __init__(self, agent_id: int, paragraph_id: int, vote: Union[int, str], step: Optional[int] = None,
                 reward: Optional[float] = None, timestamp: Optional[float] = None):
        self.agent_id = agent_id
        self.paragraph_id = paragraph_id
        self.vote = str(vote)
        self.step = step
        self.reward = reward
        self.timestamp = timestamp if timestamp is not None else time.time()

    def as_dict(self):
        return {
            "agent_id": self.agent_id,
            "paragraph_id": self.paragraph_id,
            "vote": self.vote,
            "step": self.step,
            "reward": self.reward
        }

    def __repr__(self):
        return f"Interaction(agent={self.agent_id}, para={self.paragraph_id}, vote={self.vote}, step={self.step}, reward={self.reward})"

    def __str__(self):
        return f"Agent {self.agent_id} voted {self.vote} on paragraph {self.paragraph_id} (step={self.step}, reward={self.reward})"


class Memory:
    """
    Stores all agent-paragraph interaction events, supports chronological logs
    and efficient stance reconstruction (last votes).
    """

    def __init__(self):
        self.events: List[InteractionEvent] = []

    def log_event(self, agent_id: int, paragraph_id: int, vote: Union[int, str], step: Optional[int] = None,
                  reward: Optional[float] = None):
        event = InteractionEvent(agent_id, paragraph_id, vote, step, reward)
        self.events.append(event)

    def get_events(self) -> List[InteractionEvent]:
        """Returns all logged events in order."""
        return self.events

    def last_votes_stance_matrix(self, agents: List[Any], paragraphs: List[Any]) -> pd.DataFrame:
        """
        Builds a stance matrix [agents × paragraphs] with last (most recent) vote for each agent-paragraph.
        If no vote, cell is '?'.
        """
        stance_df = pd.DataFrame("?", index=[f"a{a.agent_id}" for a in agents],
                                 columns=[f"p{p.paragraph_id}" for p in paragraphs])
        # Process in chronological order, overwrite as new events appear
        for e in self.events:
            stance_df.loc[f"a{e.agent_id}", f"p{e.paragraph_id}"] = str(e.vote)
        return stance_df

    def get_agent_history(self, agent_id: int) -> List[InteractionEvent]:
        """Returns all events for a given agent."""
        return [e for e in self.events if e.agent_id == agent_id]

    def get_paragraph_history(self, paragraph_id: int) -> List[InteractionEvent]:
        """Returns all events for a given paragraph."""
        return [e for e in self.events if e.paragraph_id == paragraph_id]

    def reset(self):
        """Clears all memory."""
        self.events = []

    def __len__(self):
        return len(self.events)
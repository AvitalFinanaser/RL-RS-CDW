"""
A straightforward abstract class to ensure consistency across all RL algorithms
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import gymnasium as gym


class BaseRLRunner(ABC):
    def __init__(self, env: gym.Env, seed: int = 42):
        self.env = env
        self.seed = seed
        self.env.reset(seed=seed)

    @abstractmethod
    def train(self, num_episodes: int, **kwargs) -> None:
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

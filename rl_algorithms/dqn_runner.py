# algorithms/dqn_runner.py
from stable_baselines3 import DQN
from rl_algorithms.base_runner import BaseRLRunner
from typing import Dict, Any
import numpy as np


#
import torch as th
from torch import nn


class DQNRunner(BaseRLRunner):
    def __init__(self, env, seed: int = 42, learning_rate: float = 1e-3):
        super().__init__(env, seed)
        self.model = DQN(
            "MultiInputPolicy",
            env=self.env,
            learning_rate=learning_rate,
            seed=seed,
            verbose=1
        )

    def train(self, num_episodes: int, **kwargs) -> None:
        total_timesteps = num_episodes * self.env.num_agents * self.env.num_paragraphs
        self.model.learn(total_timesteps=total_timesteps)

    def evaluate(self, num_episodes: int, **kwargs) -> Dict[str, Any]:
        total_rewards, completions = [], []

        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done, ep_reward = False, 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)
                ep_reward += reward

            total_rewards.append(ep_reward)
            completion = self.env._get_stance_completion_rate()
            completions.append(completion)

        return {
            "average_reward": np.mean(total_rewards),
            "average_completion_rate": np.mean(completions)
        }

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = DQN.load(path, env=self.env)

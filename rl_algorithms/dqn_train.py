import os
from typing import Optional, List, Dict, Type, Any
# stable baselines
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
# DQN
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Our env
from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader, StanceLoader


# class EpisodeTrackerCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)
#         self.episode_count = 0
#         self.episode_start_step = 0
#
#     def _on_step(self) -> bool:
#         if self.locals["dones"][0]:
#             self.episode_count += 1
#             print(f"Episode {self.episode_count} ended at step {self.num_timesteps}")
#             self.episode_start_step = self.num_timesteps
#         return True

class EpisodeTrackerCallback(BaseCallback):
    def __init__(self, verbose=0, log_path="dqn_episode_metrics.csv", step_log_path="dqn_step_metrics.csv"):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_abandonments = []
        self.episode_completions = []
        self.episode_active_agents = []
        self.episode_seeds = []  # Track seeds per episode
        self.episode_steps = []  # Track total steps per episode
        self.episode_sparsities = []
        self.log_path = log_path
        self.step_log_path = step_log_path
        self.episode_start_step = 0
        self.step_data = []
        self.current_episode_seed = None

        # Clear previous CSV files
        for path in [log_path, step_log_path]:
            if os.path.exists(path):
                os.remove(path)

    def _on_step(self) -> bool:
        step = self.num_timesteps
        reward = self.locals["rewards"][0]
        info = self.locals["infos"][0]
        components = info.get("reward_components", {"coverage": 0.0, "completion": 0.0, "content": 0.0})

        env = self.locals["env"].envs[0]
        while isinstance(env, gym.Wrapper):
            env = env.env

        # Get seed from info dict if available, otherwise from environment
        if self.current_episode_seed is None:
            self.current_episode_seed = info.get("seed", env.seed if hasattr(env, 'seed') else "unknown")

        # Get seed from environment
        if hasattr(env, 'seed') and self.current_episode_seed is None:
            self.current_episode_seed = env.seed

        # Extract step information
        agent_id = info.get("agent_id", "unknown")
        paragraph_id = info.get("paragraph_id", "unknown")
        vote = info.get("vote", "unknown")
        continuation = info.get("continuation", "unknown")

        self.step_data.append({
            "Episode": self.episode_count + 1,
            "Step": step - self.episode_start_step,
            "Reward": reward,
            "Coverage Reward": components.get("coverage", 0.0),
            "Completion Reward": components.get("completion", 0.0),
            "Content Reward": components.get("content", 0.0),
            "Seed": self.current_episode_seed,
            "Agent ID": agent_id,
            "Paragraph ID": paragraph_id,
            "Vote": vote,
            "Continuation": continuation
        })

        if self.locals["dones"][0]:
            self.episode_count += 1
            episode_steps = step - self.episode_start_step
            episode_reward = sum(d["Reward"] for d in self.step_data[-episode_steps:])

            episode_sparsity = env.stance_loader.sparsity
            self.episode_sparsities.append(episode_sparsity)

            # Access environment metrics
            try:
                abandonment_rate = 1 - len(env.active_agents) / env.num_agents
                completion_rate = env._get_stance_completion_rate()
                active_agents = len(env.active_agents)
            except AttributeError as e:
                print(f"Error accessing environment metrics: {e}")
                abandonment_rate = 0.0
                completion_rate = 0.0
                active_agents = 0

            self.episode_rewards.append(episode_reward)
            self.episode_abandonments.append(abandonment_rate)
            self.episode_completions.append(completion_rate)
            self.episode_active_agents.append(active_agents)
            self.episode_seeds.append(self.current_episode_seed)
            self.episode_steps.append(episode_steps)

            print(f"Episode {self.episode_count}: Steps {episode_steps}, Reward {episode_reward:.2f}, "
                  f"Abandonment {abandonment_rate:.2%}, Completion {completion_rate:.2%}, "
                  f"Active {active_agents}, Seed {self.current_episode_seed}")

            self._log_metrics()
            self._log_step_metrics()
            self.episode_start_step = step
            self.current_episode_seed = None  # Reset for next episode

        return True

    def _log_metrics(self):
        """Log episode-level metrics to CSV"""
        episode_data = {
            "Episode": list(range(1, self.episode_count + 1)),
            "Seed": self.episode_seeds,
            "Sparsity": self.episode_sparsities,
            "Total Steps": self.episode_steps,
            "Reward": self.episode_rewards,
            "Abandonment Rate": self.episode_abandonments,
            "Completion Rate": self.episode_completions,
            "Active Agents": self.episode_active_agents,
        }
        df = pd.DataFrame(episode_data)
        df.to_csv(self.log_path, index=False)

    def _log_step_metrics(self):
        """Log step-level metrics to CSV"""
        df = pd.DataFrame(self.step_data)
        df.to_csv(self.step_log_path, index=False)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # Calculate actual output dimension based on network outputs
        stance_out_dim = 256
        active_out_dim = 64
        completion_out_dim = 32
        mask_out_dim = 64
        features_dim = stance_out_dim + active_out_dim + completion_out_dim + mask_out_dim  # 416

        super().__init__(observation_space, features_dim)

        # Calculate input dimensions
        stance_input_dim = observation_space["stance_matrix"].shape[0] * observation_space["stance_matrix"].shape[1]
        active_input_dim = observation_space["active_agents"].shape[0]
        mask_input_dim = observation_space["action_mask"].shape[0]

        self.stance_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stance_input_dim, stance_out_dim),
            nn.ReLU()
        )
        self.active_agents_net = nn.Linear(active_input_dim, active_out_dim)
        self.stance_completion_net = nn.Linear(1, completion_out_dim)
        self.action_mask_net = nn.Linear(mask_input_dim, mask_out_dim)

    def forward(self, observations):
        stance = self.stance_net(observations["stance_matrix"])
        active = self.active_agents_net(observations["active_agents"].float())

        # Ensure completion rate is properly shaped
        completion_rate = observations["stance_completion_rate"]
        if completion_rate.dim() == 1:
            completion_rate = completion_rate.unsqueeze(-1)
        elif completion_rate.dim() == 0:
            completion_rate = completion_rate.unsqueeze(0).unsqueeze(-1)
        completion = self.stance_completion_net(completion_rate)

        mask = self.action_mask_net(observations["action_mask"].float())

        # Debug: print shapes if there's a mismatch
        if not all(t.dim() == 2 for t in [stance, active, completion, mask]):
            print(
                f"Shape mismatch - stance: {stance.shape}, active: {active.shape}, completion: {completion.shape}, mask: {mask.shape}")

        return torch.cat([stance, active, completion, mask], dim=1)


class QNetwork(BasePolicy):
    """Q-Network with action masking for collaborative document environment"""

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Discrete,
            features_extractor: BaseFeaturesExtractor,
            features_dim: int,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.net_arch = net_arch or [256, 128]

        # Create Q-network: features_dim -> action_space.n (number of paragraphs)
        q_net_layers = create_mlp(
            self.features_dim,
            action_space.n,
            self.net_arch,
            self.activation_fn
        )
        self.q_net = nn.Sequential(*q_net_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass with action masking"""
        features = self.extract_features(obs, self.features_extractor)
        q_values = self.q_net(features)

        # Apply action masking
        if "action_mask" in obs:
            action_mask = obs["action_mask"].bool()
            q_values = q_values.masked_fill(~action_mask, float('-inf'))

        return q_values

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self(obs)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class MaskableDQNPolicy(BasePolicy):
    """DQN Policy with action masking support"""

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Discrete,
            lr_schedule: Schedule,
            net_arch: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            features_extractor_class: Type[BaseFeaturesExtractor] = CustomFeaturesExtractor,
            features_extractor_kwargs: Optional[Dict] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict] = None,
            **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs or {},
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs or {},
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch or [256, 128]
        self.activation_fn = activation_fn
        self.lr_schedule = lr_schedule

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule):
        """Build Q-networks and optimizer"""
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def make_q_net(self) -> QNetwork:
        """Create Q-network with proper features extractor"""
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        # Pass the correct features_dim from features extractor
        net_args["features_dim"] = net_args["features_extractor"].features_dim
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.q_net.set_training_mode(mode)
        self.q_net_target.set_training_mode(False)
        self.training = mode

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data


class ActionMaskWrapper(gym.Wrapper):
    """Wrapper to handle invalid actions"""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space

    def step(self, action):
        obs = self.env._get_obs()
        action_mask = obs["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            # If no valid actions, return done
            obs, reward, done, truncated, info = self.env.step(0)
            return obs, reward, True, truncated, info

        if action not in valid_actions:
            # Choose random valid action
            action = np.random.choice(valid_actions)

        return self.env.step(action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        sparsity_levels = [0.3, 0.5, 0.7]
        sparsity = np.random.choice(sparsity_levels)
        self.env.stance_loader.sparsity = sparsity  # Set random sparsity
        return self.env.reset(seed=seed, options=options)


# class SparsityRandomizerWrapper(gym.Wrapper):
#     def __init__(self, env, sparsity_levels=None):
#         super().__init__(env)
#         if sparsity_levels is None:
#             sparsity_levels = [0.3, 0.5, 0.7]
#         self.sparsity_levels = sparsity_levels
#
#     def reset(self, seed=None, options=None):
#         sparsity = np.random.choice(self.sparsity_levels)
#         self.env.stance_loader.sparsity = sparsity  # Set random sparsity here
#         return self.env.reset(seed=seed, options=options)


# def evaluate_dqn(model, env, n_episodes=10, base_seed=42):
#     runner = BaselineRunner(env, [])  # Use runner's episode logic
#     results = []
#     for episode in range(n_episodes):
#         result = runner.run_episode(lambda obs: model.predict(obs, deterministic=True)[0], seed=base_seed + episode)
#         results.append(result)
#     return results
#
#
# def plot_results(dqn_csv="dqn_episode_metrics.csv", baseline_csv="baseline_results.csv", output_dir="results"):
#     os.makedirs(output_dir, exist_ok=True)
#     dqn_df = pd.read_csv(dqn_csv)
#     baseline_df = pd.read_csv(baseline_csv)
#     baseline_df['Policy'] = 'Baseline'  # For simplicity; adjust if multiple
#
#     # Reward plot
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=dqn_df, x='Episode', y='Reward', label='DQN')
#     sns.lineplot(data=baseline_df, x='episode', y='total_reward', label='Baseline')
#     plt.title("Reward per Episode")
#     plt.savefig(os.path.join(output_dir, "reward_plot.png"))
#     plt.close()
#
#     # Abandonment plot
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=dqn_df, x='Episode', y='Abandonment Rate', label='DQN')
#     sns.lineplot(data=baseline_df, x='episode', y='abandonment_rate', label='Baseline')  # Adjust column
#     plt.title("Abandonment Rate per Episode")
#     plt.savefig(os.path.join(output_dir, "abandonment_plot.png"))
#     plt.close()
#
#     # Completion plot
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=dqn_df, x='Episode', y='Completion Rate', label='DQN')
#     sns.lineplot(data=baseline_df, x='episode', y='completion_rate', label='Baseline')  # Adjust column
#     plt.title("Completion Rate per Episode")
#     plt.savefig(os.path.join(output_dir, "completion_plot.png"))
#     plt.close()
#
#     print("Plots saved in", output_dir)


if __name__ == "__main__":
    # Env initialization

    ## 1) using specific initial state
    file_path = "C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    paragraphs_loader = ParagraphsLoader(filepath=file_path)
    agents_loader = AgentsLoader(filepath=file_path)
    events_loader = EventsLoader(filepath=file_path)
    env = CollaborativeDocRecEnv(
        paragraphs_loader=paragraphs_loader,
        agents_loader=agents_loader,
        events_loader=events_loader,
        render_mode='csv',
        render_csv_name="dqn_try.csv",
        seed=42
    )

    # Check compatibility
    env = ActionMaskWrapper(env)
    check_env(env)

    ## 2) using config properties

    config = {
        "file_path": "datasets/instances/instance2",
        "instance_path": "datasets/instances/instance2",
        "sparsity": 0.3,
        "num_agents": 20,
        "num_paragraphs": 50,
        "seed": 42
    }

    env = CollaborativeDocRecEnv.from_config(config, render_mode='csv', render_csv_name="dqn_try.csv")
    env = ActionMaskWrapper(env)
    check_env(env)

    # OLD
    # config = {"sparsity": 0.3,
    #           "num_agents": 20,
    #           "num_paragraphs": 20,
    #           "file_path": "C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0",
    #           "instance_id": "instance_0"}
    # env = CollaborativeDocRecEnv.from_config(config)
    # env = ActionMaskWrapper(env)
    # check_env(env)

    # Initialize the DQN model
    model = DQN(
        policy=MaskableDQNPolicy,
        env=env,
        policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=100,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=50,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log="./dqn_tensorboard/",
        verbose=1
    )

    # Train the model
    model.learn(
        total_timesteps=100000,
        log_interval=1,
        callback=EpisodeTrackerCallback(),
        progress_bar=True
    )

    # Save trained model
    model.save("dqn_new_masked_cdw")
    print("DQN model trained and saved as 'dqn_new_masked_cdw'")

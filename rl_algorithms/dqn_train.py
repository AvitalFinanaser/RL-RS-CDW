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
# Our env
from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader, StanceLoader


class EpisodeTrackerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_start_step = 0

    def _on_step(self) -> bool:
        if self.locals["dones"][0]:
            self.episode_count += 1
            print(f"Episode {self.episode_count} ended at step {self.num_timesteps}")
            self.episode_start_step = self.num_timesteps
        return True


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

    # def reset(self, **kwargs):
    #     return self.env.reset(**kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed=seed, options=options)


if __name__ == "__main__":
    # Loaders initialization
    file_path = "C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
    paragraphs_loader = ParagraphsLoader(filepath=file_path)
    agents_loader = AgentsLoader(filepath=file_path)
    events_loader = EventsLoader(filepath=file_path)

    # Env initialization

    ## 1) using specific initial state
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
    config = {"sparsity": 0.3,
              "num_agents": 20,
              "num_paragraphs": 20,
              "file_path": "C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0",
              "instance_id": "instance_0"}
    env = CollaborativeDocRecEnv.from_config(config)
    env = ActionMaskWrapper(env)
    check_env(env)


    # Initialize the DQN model
    model = DQN(
        policy=MaskableDQNPolicy,
        env=env,
        policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
        learning_rate=1e-3,
        buffer_size=10000,
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
        total_timesteps=1000,
        log_interval=2,
        callback=EpisodeTrackerCallback(),
        progress_bar=True
    )

    # Save trained model
    model.save("dqn_masked_cdw")

# stable baselines
from typing import Type

from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.env_checker import check_env
# DWN
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
# Our env
from environment.env import CollaborativeDocRecEnv
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=128)
        self.stance_net = nn.Sequential(nn.Flatten(), nn.Linear(
            observation_space["stance_matrix"].shape[0] * observation_space["stance_matrix"].shape[1], 256), nn.ReLU())
        self.active_agents_net = nn.Linear(observation_space["active_agents"].shape[0], 64)
        self.stance_completion_net = nn.Linear(1, 32)
        self.action_mask_net = nn.Linear(observation_space["action_mask"].shape[0], 64)

    def forward(self, observations):
        stance = self.stance_net(observations["stance_matrix"])
        active = self.active_agents_net(observations["active_agents"].float())
        completion = self.stance_completion_net(observations["stance_completion_rate"])
        mask = self.action_mask_net(observations["action_mask"].float())
        return torch.cat([stance, active, completion, mask], dim=1)


class MaskableDQNPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                 activation_fn: Type[nn.Module] = nn.ReLU, **kwargs):
        super().__init__(observation_space, action_space,
                         optimizer_class=optimizer_class,
                         **kwargs)
        q_input_dim = 128
        self.q_net = nn.Linear(q_input_dim, action_space.n)
        self.q_net_target = nn.Linear(q_input_dim, action_space.n)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.activation_fn = activation_fn


    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        q_values = self.q_net(features)
        action_mask = obs["action_mask"].bool().to(self.device)
        q_values = q_values.masked_fill(~action_mask, float('-inf'))
        action = torch.argmax(q_values, dim=1) if deterministic else torch.distributions.Categorical(
            logits=q_values).sample()
        return action, q_values, None

    def _predict(self, observation, deterministic=False):
        action, _, _ = self.forward(observation, deterministic)
        return action

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        q_values = self.q_net(features)
        action_mask = obs["action_mask"].bool().to(self.device)
        q_values = q_values.masked_fill(~action_mask, float('-inf'))
        return q_values, torch.zeros_like(q_values), q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = env.action_space

    def step(self, action):
        action_mask = self.env._get_obs()["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]
        if action not in valid_actions:
            action = np.random.choice(valid_actions)
            print(f"Invalid action {action} chosen, selecting random valid action {action} from set {valid_actions}")
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Loaders initialization (replace with your data paths)
file_path = "C:/Users/avita/Desktop/לימודים/תוכנית מיתר/Consenz project/CDW/datasets/event_lists/config001_llm/(CSF=0_events,_APS,_threshold=0.5)/instance_0"
paragraphs_loader = ParagraphsLoader(filepath=file_path)
agents_loader = AgentsLoader(filepath=file_path)
events_loader = EventsLoader(filepath=file_path)

# Env initialization
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
# check_env(env)

# Initialize the DQN model
model = DQN(
    policy=MaskableDQNPolicy,
    env=env,
    policy_kwargs={"features_extractor_class": CustomFeaturesExtractor},
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=100,
    batch_size=32,
    tau=1.0,  # Soft update coefficient for target network
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,  # Number of gradient steps per update
    target_update_interval=50,
    exploration_initial_eps=1.0,  # Initial epsilon for exploration
    exploration_final_eps=0.05,  # Final epsilon for exploration
    max_grad_norm=10,  # Gradient clipping threshold
    tensorboard_log="./dqn_tensorboard/",
    verbose=1
)

# Train the model
model.learn(
    total_timesteps=10000,  # Total training steps
    log_interval=2,  # Log progress every 10 episodes
    progress_bar=True  # Show a progress bar during training
)

# Save trained model
model.save("dqn_masked_cdw")

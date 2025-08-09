"""
The Gymnasium environment class implementation for the collaborative writing scenario, tying everything together.
Initializes the collaborative writing environment.
Defines the observation space and action space.
Implements reset() and step(agent, action) (get feedback from a Collaborator, compute reward, update state).
Will use other modules and ensures compliance with Gym (return (obs, reward, done, info)).
"""
import os
from typing import Optional, Dict, Any, List
#
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
#
from environment.collaborators import LLMAgentWithTopics
from environment.stance import StanceMatrix
from environment.memory import Memory
from environment.topic_model import compute_paragraphs_topic_matrix
from environment.loaders import ParagraphsLoader, AgentsLoader, EventsLoader, StanceLoader
from environment.reward_shaping import RewardShaper


#


class CollaborativeDocRecEnv(gym.Env):
    """
    Collaborative Document Recommendation RL Environment.
    - Initializes paragraphs (items), agents (users), and stance (mappings of current votes).
    - Sets up MDP components: action/observation spaces, memory, and reward shaping.
    - Modular: adaptive to different agent/paragraph loaders, reward schemes.
    """

    def __init__(
            self,
            render_mode: str = 'human',
            stance_loader: Optional[StanceLoader] = None,
            paragraphs_loader: ParagraphsLoader = None,
            agents_loader: AgentsLoader = None,
            events_loader: EventsLoader = None,
            initial_stance_matrix: Optional[Any] = None,
            paragraphs_retrieval=None,
            reward_shaping_params: Optional[Dict[str, Any]] = None,
            render_path: str = "./render/",
            render_csv_name: str = "render.csv",
            seed=42):

        """
        Flow of environment construction:
          0. Gym Environment setup
          1. Render mode and metadata
          2. Loaders
          2.1 Paragraphs
          2.2 Agents
          2.3 Initial stance
          2.4 Attach topic vectors
          3. Observation space
          4. Action space
          5. Action/paragraph mappings
          5. Interaction (env) memory
          6. Reward shaping parameters (alpha, etc.)
          7. Retrieval and selection modules
          8. Reward Shaping
        """
        # 0. Gym Environment setup
        super().__init__()

        ## Store and set seed
        self._set_seed(seed=seed)

        # 1. Render Mode - Allows flexible output/logging (human-readable, CSV).
        self.render_mode = render_mode
        self.render_path = render_path
        self.render_csv_name = render_csv_name
        self.metadata = {"render_modes": ["human", "csv"]}

        # 2. Loaders - Load all paragraphs (items), agents (users) and events (initial stance)

        ## 2.1 Paragraphs loading
        assert paragraphs_loader is not None, 'A ParagraphsLoader is required.'
        self.paragraphs_loader = paragraphs_loader
        self.paragraphs = self.paragraphs_loader.load_all()  # List[Paragraph]
        self.num_paragraphs = len(self.paragraphs)  # m - number of total paragraphs
        self.paragraph_ids = [p.paragraph_id for p in self.paragraphs]

        ## 2.2 Agents loading
        assert agents_loader is not None, 'An AgentsLoader is required.'
        self.agents_loader = agents_loader
        self.agents = self.agents_loader.load_all()  # List[LLMAgentWithTopics]
        self.num_agents = len(self.agents)  # n - number of total agents
        self.agent_ids = [a.agent_id for a in self.agents]

        ## 2.3 Initial stance loading (default: all "?")

        ## StanceLoader
        self.stance_loader = stance_loader
        if self.stance_loader is not None:
            self.stance_loader.agents = self.agents
            self.stance_loader.paragraphs = self.paragraphs
        self.events_loader = events_loader
        self.initial_stance_matrix = initial_stance_matrix

        self._init_stance_matrix()

        ## 2.4 Attach topic vectors

        ### 2.4.1 Topic modeling
        # - Extract texts for topic modeling
        texts = [p.text for p in self.paragraphs]
        # - Perform topic modeling to get document-topic matrix
        doc_topic_matrix, best_k, topic_keywords = compute_paragraphs_topic_matrix(texts, k_range=range(3, 4))
        # doc_topic_matrix, best_k, topic_keywords = compute_paragraphs_topic_matrix(texts, k_range=range(3, 21))
        # - Store topic modeling results
        self.doc_topic_matrix = doc_topic_matrix
        self.best_k = best_k
        self.topic_keywords = topic_keywords

        ### 2.4.2 Attach topics to paragraphs
        self.paragraphs = self.paragraphs_loader.attach_topics(self.paragraphs, doc_topic_matrix)

        ### 2.4.3 Attach topics to agents
        LLMAgentWithTopics.update_all_topic_vectors(
            agents=self.agents,
            doc_topic_matrix=self.doc_topic_matrix,
            stance_matrix=self.stance
        )

        # 3. Observation Space
        ## observation_space represents the state information that is available to the RL agent at each decision step.

        ## 3.1 Defining state variables
        self.current_agent_id = None
        self.step_num = 0
        self.evaluation_count = 0
        self.evaluation_previous_user_id = None
        ### Active agents - agents with incomplete preferences hence participating in the RS
        self.active_agents = set(
            agent.agent_id for agent in self.agents
            if len(self.stance.get_unknown_paragraphs(agent.agent_id)) > 0
        )

        ## 3.2 Defining observation space
        self.observation_space = spaces.Dict({
            "stance_matrix": spaces.Box(
                low=-2, high=1,  # -2 stands for "?"
                shape=(self.num_agents, self.num_paragraphs),
                dtype=np.float32
            ),
            "current_agent_id": spaces.Discrete(self.num_agents),
            "active_agents": spaces.MultiBinary(self.num_agents),
            "stance_completion_rate": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.num_paragraphs)
        })

        # 4. Action Space
        self.action_space = spaces.Discrete(self.num_paragraphs)

        # 5. Action-Paragraph mappings
        self.action_to_paragraph = {i: pid for i, pid in enumerate(self.paragraph_ids)}
        self.paragraph_to_action = {pid: i for i, pid in enumerate(self.paragraph_ids)}

        # 6. Memory Initialization (interactions history)
        self.memory = Memory()

        # 7. Retrieval and Selection
        ## Additional modules for prompting with selecting a shortlist from the paragraphs pool before making the final recommendation.
        self.paragraphs_retrieval = paragraphs_retrieval

        # 8. Reward Shaping
        self.reward_shaper = RewardShaper(reward_shaping_params)

    def step(self, action: int):
        """
        The step takes an action, which correspond to a paragraph recommendation.
        We use the mapping paragraph_to_action to map an action to its corresponding paragraph.
        """
        if self._check_termination():
            print(f"Step {self.step_num}: Episode terminated before action.")
            return self._get_obs(), 0.0, True, False, {"info": "Episode terminated"}

        # 1 - Validate action
        valid_actions = self.get_valid_actions(self.current_agent_id)
        assert action in valid_actions, f"Invalid action: {action} not in valid actions: {valid_actions}"

        # 2 - Current interaction
        paragraph_id = self.action_to_paragraph[action]
        agent_id = self.current_agent_id
        self.step_num += 1

        # 3- Get vote & continuation from agent (feedback)
        vote, continuation = self._get_agent_feedback(agent_id, paragraph_id, self.stance)

        print(f"Step {self.step_num},"
              f" Current agent: a{self.current_agent_id},"
              f" Action: p{action},"
              f" Vote:{vote},"
              f" Continuation: {continuation}")

        # Updating state variables - state, topic vectors, active agents

        # 4 - Update stance matrix with new vote
        self.stance.set_vote(agent_id=self.current_agent_id, paragraph_id=paragraph_id, vote=vote)

        # 5 - Update agent topic vectors after new interaction
        LLMAgentWithTopics.update_all_topic_vectors(
            agents=self.agents,
            doc_topic_matrix=self.doc_topic_matrix,
            stance_matrix=self.stance
        )

        # 6 - Handle agent continuation
        if continuation == 0 or len(self.stance.get_unknown_paragraphs(agent_id)) == 0:  # Agent terminates
            self.active_agents.discard(agent_id)

        # 5 - Calculate reward
        agents_active = [a for a in self.agents if a.agent_id in self.active_agents]
        reward = self.reward_shaper.calculate_reward(agents=agents_active,
                                                     stance=self.stance,
                                                     doc_topic_matrix=self.doc_topic_matrix)

        # 6 - Update memory - Log interaction
        self.memory.log_event(
            agent_id=agent_id,
            paragraph_id=paragraph_id,
            vote=vote,
            step=self.step_num,
            reward=reward
        )

        # Moving onwards

        # 7 - Select next agent randomly from active agents
        if self.active_agents:
            self.current_agent_id = np.random.choice(list(self.active_agents))
        else:
            self.current_agent_id = None

        # 8 - Check termination
        terminated = self._check_termination()

        # 9 - Get observation
        observation = self._get_obs()

        info = {
            "vote": vote,
            "continuation": continuation,
            "agent_id": agent_id,
            "paragraph_id": paragraph_id
        }

        return observation, reward, terminated, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Clean up reset method"""
        # Reset provided seed or default (random)
        if seed is None:
            seed = np.random.randint(1000000)
        self._set_seed(seed=seed)
        super().reset(seed=seed)

        # Reset state variables
        self.step_num = 0
        self.active_agents = set(
            agent.agent_id for agent in self.agents
            if len(self.stance.get_unknown_paragraphs(agent.agent_id)) > 0
        )
        self.current_agent_id = np.random.choice(self.agent_ids)
        if self.active_agents:
            self.current_agent_id = np.random.choice(list(self.active_agents))
        else:
            self.current_agent_id = None

        # Reset memory
        self.memory.reset()

        # Reset stance matrix to initial state
        self._init_stance_matrix()

        # Reset initial observation
        observation = self._get_obs()

        return observation, {}

    def render(self):
        """Render environment state and preserve interaction metadata."""

        # Print to console
        if self.render_mode == "human":
            print(
                f"Step: {self.step_num}\n"
                f"Current agent: a{self.current_agent_id}\n"
                f"Active agents: {len(self.active_agents)}/{self.num_agents}\n"
                f"Stance completion: {self._get_stance_completion_rate():.2%}"
            )
            if len(self.memory.events) > 0:
                recent = self.memory.events[-1:]  # Last interaction
                print(f"Recent interaction: {recent}")
            print("-" * 50)

        # Print to csv
        if len(self.memory.events) > 0 and self.render_mode == "csv":
            # Convert memory events to DataFrame
            events_data = [event.as_dict() for event in self.memory.events]
            df = pd.DataFrame(events_data)

            # Add episode metadata
            df['episode_step'] = self.step_num
            df['active_agents_count'] = len(self.active_agents)
            df['completion_rate'] = self._get_stance_completion_rate()

            # Save to file in the render_path directory as 'render.csv'
            csv_path = os.path.join(self.render_path, self.render_csv_name)
            if not os.path.exists(self.render_path):
                os.makedirs(self.render_path, exist_ok=True)
            # if os.path.exists(csv_path):
            #     df.to_csv(csv_path, mode="a", index=False, header=False)
            # else:
            #     df.to_csv(csv_path, index=False)
            df.to_csv(csv_path, index=False)  # Overwrite

    def _init_stance_matrix(self):
        """Initialize stance matrix."""
        if self.stance_loader is not None:
            # Option 1 - Using the stance loader (for random)
            self.stance = self.stance_loader.load_random()
        elif self.initial_stance_matrix is not None:
            # Option 2 - No loader but yes for a starting stance matrix
            self.stance = self.initial_stance_matrix
        elif self.events_loader is not None:
            # Option 3 - An EventsLoader is given for initiating a stance from previous event list.
            events_df = self.events_loader.load_all(agents=self.agents, paragraphs=self.paragraphs)
            self.stance = StanceMatrix.from_existing(agents=self.agents, paragraphs=self.paragraphs, matrix=events_df)
        else:
            # Option 4 (Default) - No loader and not a starting stance matrix -> all preferences are unknown
            self.stance = StanceMatrix(agents=self.agents, paragraphs=self.paragraphs)

    def get_valid_actions(self, agent_id: int) -> List[int]:
        """
        Retrieve list of valid actions (paragraph indices) that the given agent has not voted on yet.
        Returns: a list of action indices of paragraphs with unknown votes.
        """
        unknown_paragraphs = self.stance.get_unknown_paragraphs(agent_id)  # A list of paragraphs objects
        return [self.paragraph_to_action[p.paragraph_id] for p in unknown_paragraphs]

    def _get_obs(self):
        """
        Returns the observation for the RL agent to decide which paragraph to recommend next.
        - stance_matrix: float matrix with votes {-1, 0, 1}, unknown as -2
        - current_agent_id: integer index
        - active_agents: binary mask over agents
        - stance_completion_rate: float in [0,1]
        """
        # Build stance_numerical and mask
        stance_numerical = np.full((self.num_agents, self.num_paragraphs), -2, dtype=np.float32)

        for i, agent in enumerate(self.agents):
            for j, paragraph in enumerate(self.paragraphs):
                vote = self.stance.get_vote(agent.agent_id, paragraph.paragraph_id)
                if vote == "?":
                    # Unknown vote: leave as -2, mask=0
                    continue
                else:
                    stance_numerical[i, j] = float(vote)  # -1, 0, or 1

        # Active agents mask (boolean array)
        active_mask = np.array([agent.agent_id in self.active_agents for agent in self.agents], dtype=np.int8)

        # Current agent index (for the RL agent to know who it's recommending to)
        if self.current_agent_id is not None:
            valid_actions = self.get_valid_actions(self.current_agent_id)
            current_agent_idx = next(i for i, a in enumerate(self.agents) if a.agent_id == self.current_agent_id)
        else:
            current_agent_idx = 0
            valid_actions = []

        # Action valid mask (boolean array)
        action_mask = np.zeros(self.num_paragraphs, dtype=np.int8)
        action_mask[valid_actions] = 1

        return {
            "stance_matrix": stance_numerical,  # shape [num_agents, num_paragraphs]
            "current_agent_id": current_agent_idx,
            "active_agents": active_mask,
            "stance_completion_rate": np.array([self._get_stance_completion_rate()], dtype=np.float32),
            "action_mask": action_mask
        }

    def _get_stance_completion_rate(self):
        """Calculate percentage of known votes."""
        known_entries = (self.stance.matrix.values != "?").sum()
        total_entries = self.stance.matrix.size
        return known_entries / total_entries if total_entries > 0 else 0.0

    def _check_termination(self) -> bool:
        """Check if episode should terminate in case of:
        - stance matrix completion (all preferences noted)
        - no active agents (all agents abandon the system)
        """

        # Stance matrix complete
        if self.stance.is_complete():
            print("Termination - stance is complete")
            return True

        # No active agents
        if len(self.active_agents) == 0:
            print("Termination - no active agents")
            return True

        return False

    def _get_agent_feedback(self, agent_id: int, paragraph_id: int, stance: StanceMatrix) -> tuple:
        """Get agent vote and continuation signal."""
        # For now, agent favors vote at random
        vote = np.random.choice([-1, 0, 1])  # Placeholder
        # For now, agent favors continuation when are missing preferences
        if len(self.stance.get_unknown_paragraphs(agent_id)) > 0:
            continuation = 1
        else:
            continuation = 0
        return str(vote), continuation

    def _set_seed(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    @staticmethod
    def from_config(config: dict, render_mode: str = 'human', reward_params: Optional[Dict[str, Any]] = None,
                    render_csv_name: str = "render.csv"):
        agents_loader = AgentsLoader(filepath=config["file_path"], num_agents=config["num_agents"])
        paragraphs_loader = ParagraphsLoader(filepath=config["file_path"], num_paragraphs=config["num_paragraphs"])
        stance_loader = StanceLoader(
            agents=agents_loader.load_all(),
            paragraphs=paragraphs_loader.load_all(),
            sparsity=config["sparsity"]
        )
        return CollaborativeDocRecEnv(
            paragraphs_loader=paragraphs_loader,
            agents_loader=agents_loader,
            stance_loader=stance_loader,
            reward_shaping_params=reward_params,
            render_mode=render_mode,
            render_csv_name=render_csv_name,
            seed=config.get("seed", 42)
        )

    # def step(self, action: int):
    #     """
    #     The step takes an action, which correspond to a paragraph recommendation.
    #     We use the mapping paragraph_to_action to map an action to its corresponding paragraph.
    #     """
    #     item_id = self.action_to_item[action]
    #
    #     """
    #     Use the MoviesLoader to load curr_item, which is a Movie object crresponding to the id item_id
    #     """
    #     curr_item = self.paragraphs_loader.load_items_from_ids(id_list=[item_id])
    #
    #     """
    #     We fetch from the memory all previous films seen by the user together with the
    #     interaction that the user had with the items. The interaction is represented via an interaction object
    #     that summarize all important informations.
    #     """
    #     past_items, past_interactions = self.memory.get_items_and_scores(self._user.id)
    #     num_interacted = self.memory.get_num_interaction(self._user.id, item_id)
    #
    #     """
    #     The next step is to retrieve from the list of all items seen a smaller list of relevant items. The relevance from the Movie
    #     depends on the retrieved mode.
    #     """
    #     retrieved_items, retrieved_interactions = self.paragraphs_retrieval.retrieve(
    #         curr_item[0], past_items, past_interactions
    #     )
    #
    #     """
    #     Given the user, the recommended item and the retieved item we construct a prompt for the LLM to predict the rating that
    #     the user would give to the recommended Movie.
    #     """
    #     with torch.random.fork_rng(["cuda:0"]):
    #         torch.manual_seed(self.llm_seed)
    #         rating, explanation, html_interaction = self.rating_prompt.query(
    #             self._user,
    #             curr_item[0],
    #             num_interacted,
    #             retrieved_interactions,
    #             retrieved_items,
    #         )
    #         self.llm_seed += 1
    #
    #     """
    #     After collecting the explanation and the rating from the LLM the next step is to select the item
    #     (but this only in the case more than one is recommended)
    #     """
    #     selected_items, selected_ratings = self.paragraphs_selector.select(
    #         curr_item, [rating]
    #     )
    #
    #     """
    #     Add a small perturbation to the rating.
    #     """
    #     selected_items, selected_ratings = self.reward_perturbator.perturb(
    #         curr_item, [rating]
    #     )
    #
    #     """
    #     The next step consists in updating the Memory by adding the recommended item to the list of film seen by the user.
    #     """
    #     selected_items_ids = []
    #
    #     for m in selected_items:
    #         selected_items_ids.append(m.id)
    #
    #     self.memory.update_memory(self._user.id, selected_items_ids, selected_ratings)
    #
    #     """
    #     We also update the state by adding the recommended item to the list of film seen
    #     """
    #     self._items_interact = self._items_interact + (
    #         np.array(
    #             [self.item_to_action[selected_items_ids[0]], selected_ratings[0]],
    #             dtype=np.int_,
    #         ),
    #     )
    #
    #     """
    #     Termination is modelled in a similar fashion to a geometric distribution: after every step the user with some small probability
    #     stops intercating with the environment
    #     """
    #     terminated = self.np_random.choice([True, False], p=[0.025, 0.975])
    #     terminated = bool(terminated)
    #     observation = self._get_obs()
    #     reward = reduce(lambda x, y: x + y, selected_ratings)
    #     info = {
    #         "LLM_explanation": explanation,
    #         "LLM_rating": rating,
    #         "LLM_interaction_HTML": html_interaction,
    #     }

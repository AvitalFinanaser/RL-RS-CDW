import os
import re
import random

import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Optional, Literal
from langchain_openai import ChatOpenAI

# Parser output
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Langchain
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain

# Env
from environment.topic_model import compute_agents_profiles

SENTIMENT_CATEGORIES = {
    (0.00, 0.20): "active_resistance",  # Complete opposition to in favor action
    (0.20, 0.40): "minimal_acknowledgment",  # Basic recognition without meaningful action
    (0.40, 0.60): "balanced_approach",  # Equal consideration of other factors
    (0.60, 0.80): "supportive_measures",  # Proactive support with practical constraints
    (0.80, 1.00): "proactive_action"  # Topic-first transformative approaches
}
CATEGORY_DESCRIPTIONS = {
    "active_resistance": "complete opposition to in-favor action.",
    "minimal_acknowledgment": "basic recognition without meaningful action.",
    "balanced_approach": "equal consideration of the topic and other factors.",
    "supportive_measures": "proactive support with practical constraints.",
    "proactive_action": "topic-first transformative approaches.",
    "undefined": "no valid position category assigned."
}


class Agent:

    def __init__(self, agent_id, name=""):
        """
        agent_id (integer): unique identifier for each agent
        name (string): name of the agent
        """
        default_name = f"a{agent_id}"
        self.agent_id = agent_id  # Agent's identifier
        self.name = name if name else default_name

    def __str__(self):
        return f"Agent: a{self.agent_id}, named: {self.name}"


class LLMAgent(Agent):
    """
    An agent powered by a large language model (LLM) through LangChain.
    This agent is defined by a demographic profile and topic-distribution vector.
    This agent can participate in CDW - RS system and given a system state and recommendation, make a decisions.
    """

    def __init__(self,
                 agent_id: int,
                 profile: Dict[str, str],
                 topic: str,
                 topic_position: float,
                 examples_topic: str = 'grouped_proposals_by_sentiment.json'):
        """
        Initialize a new CDWAgent.
        """
        super().__init__(agent_id)

        # Load environment variables from .env file
        load_dotenv()

        # Check for API key
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

        # Agent characteristics
        self._profile = profile
        self._topic = topic
        self._topic_position = topic_position
        self._position_category = self._get_position_category()
        self._grouped_examples = self.get_example_category_proposals(examples_topic)
        self._sample_examples = self.sample_diverse_proposals()

        # LangChain LLM
        self._llm = ChatOpenAI(
            model_name=os.environ.get("LLM_MODEL"),
            temperature=os.environ.get("LLM_TEMPERATURE")
        )

        # System Prompt
        self.system_prompt = self.construct_system_prompt()

        # Initialize memory for tracking history
        self.memory = []  # List of action dicts
        self.action_counter = 0  # Start counting actions from zero

    def reset(self):
        """
        Resets the memory tracking of the agent.
        This method clears the existing history, effectively restarting the agent actions.
        """
        # Clear conversation history
        self.memory = {"proposals": [], "votes": [], "reasoning": []}

    def _get_position_category(self) -> str:
        """Get the sentiment category based on the position value."""
        for (low, high), category in SENTIMENT_CATEGORIES.items():
            if low <= self._topic_position < high:
                return category
        return "undefined"

    def get_example_category_proposals(self, examples_topic: str = 'grouped_proposals_by_sentiment.json') -> dict:
        """
        Load the grouped by category proposals for few-shot prompting.

        Args:
            examples_topic (str): Filename of the JSON file containing categorized proposals.

        Returns:
            dict: Sentiment category mapped to proposal examples.
        """
        import json
        import os
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_dir, 'datasets', 'processed', examples_topic)
        with open(file_path, 'r', encoding='utf-8') as f:
            all_examples = json.load(f)
            # Return only examples relevant to the agent's sentiment category
        category = self._position_category
        if category == "active_resistance":
            category = "balanced_approach"
        if category == "proactive_action":
            category = "supportive_measures"
        return all_examples.get(category, [])

    def sample_diverse_proposals(self, num_samples=3):
        """
        Sample a specified number of diverse domain proposals from the agent's sentiment category.

        Parameters:
        -----------
        num_samples : int, optional (default=3)
            The number of diverse domain proposals to return.

        Returns:
        --------
        diverse_samples :
        List of proposal dicts:
                - text: str, the proposal text
                - domain: str, the domain category
                - rationale: str, the reasoning behind the proposal
        """
        proposals = self._grouped_examples
        random.shuffle(proposals)

        seen_domains = set()
        diverse_samples = []

        for proposal in proposals:
            if proposal['domain'] not in seen_domains:
                diverse_samples.append(proposal)
                seen_domains.add(proposal['domain'])
            if len(diverse_samples) == num_samples:
                break

        return diverse_samples

    def _format_profile(self) -> str:
        """
        Format the agent's profile into a natural language sentence.
        Example: "You are a 35 years old male with upper secondary education."
        """
        sex = self._profile.get("Sex", "unspecified").lower()
        age = self._profile.get("Age", "unspecified")
        education = self._profile.get("Educational attainment", "unspecified").lower()

        return f"You are a {age} years old {sex} with {education}"

    def _format_examples_for_prompt(self, num_samples=3) -> str:
        """
        Samples and formats few-shot examples for inclusion in the decision prompt.

        Returns:
            str: A string with formatted examples for the agent based on their sentiment category.
        """
        # Single-line formatting
        lines = ["EXAMPLES OF ACTIONS SIMILAR TO YOUR POSITION:"]
        for i, p in enumerate(self._sample_examples, 1):
            lines.append(f"{i}. [{p['domain']}] {p['text']} (Reasoning: {p['rationale']})")

        return "\n".join(lines)

    def construct_system_prompt(self) -> str:
        """
        Construct the system prompt that defines the agent's role and behavior.

        Returns:
            A string representing the system prompt for the LLM.
        """
        profile_text = self._format_profile()
        description = CATEGORY_DESCRIPTIONS.get(self._position_category, "undefined")

        system_prompt = f"""
        You are Agent a{self.agent_id}, a participant in a collaborative constitution writing system {self._topic}.
        You engage dynamically with an evolving draft document, consisting of community proposals.
        The document is a list of action proposals the community you are part of should determine policy.
        YOUR PROFILE:{profile_text}. 
        YOUR POSITION ON {self._topic.upper()}: {self._topic_position} ({self._position_category}). 
        Accordingly, you have the following orientation: {description}
        As a participant, you can decide on voting on existing proposal (upvote, down-vote, abstain an existing vote) or suggest a new proposal.
        Act consistently with your profile. Think step-by-step; align your reasoning explicitly with your stated orientation.
        """
        return system_prompt.strip()

    def get_memory_context(self, limit=2) -> str:
        """
        Retrieve recent memory of actions in string format for the agent's prompt.

        Args:
            limit (int): Number of recent actions to include.

        Returns:
            str: Formatted string of recent actions.
        """
        recent_actions = self.memory[-limit:]
        context_lines = []

        for action in recent_actions:
            context_lines.append(f"Action ID: {action['action_id']}")
            context_lines.append(f"Type: {action['type']}")
            context_lines.append(f"Content: {action['content']}")
            context_lines.append(f"Vote: {action['vote']}")
            context_lines.append(f"Reasoning: {action['reasoning']}")
            context_lines.append("")  # Blank line between actions

        return "\n".join(context_lines).strip() if context_lines else "No prior actions."

    def _update_memory(self, action: Dict[str, str]):
        """
        Update memory with the latest action.

        Args:
            action (Dict): Contains 'DECISION', 'TEXT', 'VOTE', 'REASONING'
        """
        self.memory.append(action)
        self.action_counter += 1
        self._sample_examples = self.sample_diverse_proposals()

    def construct_decision_prompt(self, current_state):
        """
           Construct the decision prompt for the agent based on the current system state and memory.

           Args:
               current_state (str): Description of the current state of the document/system.
           Returns:
               str: A decision prompt ready for LLM input.
           """
        if not current_state:
            system_state_section = "There are currently no proposals in the system."
            action_hint = "You are expected to propose the first action for the community."
        else:
            system_state_section = (f"The current system state is as follows:\n"
                                    f"Current own vote: (? - you did not express a position yet, +1 - favor, -1 - against)\n{current_state}\n")
            action_hint = "You can either PROPOSE a new action or VOTE on an existing proposal"
        prompt = f"""
{system_state_section}
TASK: Based on the system state and your profile, you must decide your next action through the following reasoning steps.
{action_hint}. Be very concise when addressing each step.

**Step 1: Situation Analysis**
- Analyze the current proposals in the system, sub-topics, their vote counts, and **your own voting stance**.
- Note which paragraphs are **included** in the current document and which are not.
- Identify which topics are over-represented (repeated ideas), under-represented (missing themes), or unbalanced (too extreme or vague).

**Step 2: Position Alignment**
- Evaluate if paragraphs included in the document reflect your profile and position.
- Identify proposals or included content that you support or oppose, and consider whether they align with your background and orientation.
- Is there a critical gap in themes, sub-topics, or emphasis that makes the document feel incomplete or unbalanced from your perspective?

**Step 3: Decision Evaluation**
- Evaluate whether proposing a new action or voting on an existing proposal would be the best way to influence the document at this point.
- If your position on a proposal has weakened from previous vote (*own vote is not ?*), explicitly choose ABSTAIN to reflect uncertainty or neutrality.
- Determine the optimal strategic action: propose a new paragraph / vote on an existing one, to best align the document with your values.

**Step 4: Action Formulation**

If PROPOSE: Formulate a new proposal that:
- Addresses a sub-topic important to your background
- Contains a practical, realistic, diverse action suggestion (max 20 words)
- Aligns with these examples of your position:
{self._format_examples_for_prompt()}

If VOTE: Choose a paragraph and indicate clearly if you UPVOTE (agree), DOWNVOTE (disagree), or ABSTAIN (change previous vote to undecided).
VOTING RULES (MANDATORY):  
- You can only ABSTAIN on a paragraph where your current 'own vote' is already +1 or -1
- If your current 'own vote' is '?', choose only UPVOTE or DOWNVOTE
- Do NOT repeat your current vote.

You must format your response exactly according to this structure (MANDATORY):
DECISION: PROPOSE / VOTE
PARAGRAPH ID: [If VOTE: Paragraph ID, If PROPOSE: The next available ID]
ACTION DETAILS: [If VOTE: The full text of the proposal you are voting on, If PROPOSE: New Proposal text]
VOTE: [If VOTE: UPVOTE / DOWNVOTE / ABSTAIN, If PROPOSE: UPVOTE (Only)]
REASONING: [Brief justification, max 20 words, aligned with position, profile, and current state].
"""
        return prompt.strip()

    def _generate_decision_chain(self, prompt_template) -> LLMChain:
        """
        Create a LangChain for processing a decision type of request.
        Returns:
            A LangChain for processing the request
        """

        # Use SystemMessage instead of SystemMessagePromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt_template)
        ])

        return chat_prompt | self._llm

    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the LLM response into structured action fields and assign action_id.
        """
        decision_match = re.search(r"DECISION:\s*(PROPOSE|VOTE)", response, re.IGNORECASE)
        paragraph_id_match = re.search(r"PARAGRAPH ID:\s*(\d+)", response, re.IGNORECASE)
        action_details_match = re.search(r"ACTION DETAILS:\s*(.*)", response, re.IGNORECASE)
        vote_match = re.search(r"VOTE:\s*(UPVOTE|DOWNVOTE|ABSTAIN)", response, re.IGNORECASE)
        reasoning_match = re.search(r"REASONING:\s*(.*)", response, re.IGNORECASE)

        parsed_action = {
            "action_id": self.action_counter,
            "type": decision_match.group(1).strip().upper() if decision_match else "",
            "paragraph_id": int(paragraph_id_match.group(1)) if paragraph_id_match else -1,
            "content": action_details_match.group(1).strip() if action_details_match else "",
            "vote": vote_match.group(1).strip().upper() if vote_match else "",
            "reasoning": reasoning_match.group(1).strip() if reasoning_match else ""
        }

        return parsed_action

    def decision_chat(self, current_state: str):
        """
        Generates a response from the agent based on the system state.

        This method constructs a prompt with the current system state and decision history.
        Then, sends it to the llm, updates history memory, and returns parse decision response.

        Parameters:
            - user_msg (str): The user's message to which the chatbot will respond.

        Returns:
            - str: The generated response from the chatbot.
        """

        # 1. Construct decision prompt
        prompt = self.construct_decision_prompt(current_state)
        print(prompt)
        # 2. Generate LLM response
        llm_chain = self._generate_decision_chain(prompt)
        llm_result = llm_chain.invoke({})
        llm_response_text = llm_result.content

        # 3. Parse response into action
        action = self._parse_response(llm_response_text)
        print(action)
        # 4. Update memory with new action
        self._update_memory(action)

        return action

    @staticmethod
    def create_community(sampled_profiles: List[Dict[str, str]], topic: str):
        """
        Static method to create a community of LLMAgents.

        Args:
            sampled_profiles: List of profile dictionaries (each with Sex, Age, Education).
            topic: The shared topic for all agents.

        Returns:
            List of LLMAgent instances.
        """
        agents = []

        for i, profile in enumerate(sampled_profiles, start=1):
            # topic_position = bounded_normal(mu=0.5, sigma=0.2)
            topic_position = round(random.uniform(0.0, 1.0), 2)
            agent = LLMAgent(
                agent_id=i,
                profile=profile,
                topic=topic,
                topic_position=topic_position
            )
            agents.append(agent)

        return agents

    @property
    def model(self) -> str:
        """Get the LLM model being used."""
        return self._llm.model_name

    @model.setter
    def model(self, value: str):
        """Set the LLM model to use."""
        self._llm.model_name = value

    @property
    def temperature(self) -> float:
        """Get the temperature parameter for the LLM."""
        return self._llm.temperature

    @temperature.setter
    def temperature(self, value: float):
        """Set the temperature parameter for the LLM."""
        if not 0 <= value <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        self._llm.temperature = value

    @property
    def profile(self) -> Dict[str, str]:
        return self._profile

    @profile.setter
    def profile(self, value: Dict[str, str]):
        if not isinstance(value, dict):
            raise ValueError("Profile must be a dictionary.")
        self._profile = value

    @property
    def topic(self) -> str:
        return self._topic

    @topic.setter
    def topic(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Topic must be a string.")
        self._topic = value

    @property
    def topic_position(self) -> float:
        return self._topic_position

    @topic_position.setter
    def topic_position(self, value: float):
        if not (0.0 <= value <= 1.0):
            raise ValueError("Topic position must be between 0 and 1.")
        self._topic_position = value
        self._position_category = self._get_position_category()  # Update category dynamically

    @property
    def position_category(self) -> str:
        return self._position_category

    @property
    def grouped_examples(self) -> str:
        return self._grouped_examples

    def __str__(self):
        return f"ID: {self.agent_id}, Topic: {self._topic}, Position: {self._topic_position} ({self._position_category}), Profile: {self._format_profile()}"


class LLMAgentWithTopics(LLMAgent):
    def __init__(self,
                 agent_id: int,
                 profile: Dict[str, str],
                 topic: str,
                 topic_position: float,
                 topic_profile_vector=None,
                 examples_topic='grouped_proposals_by_sentiment.json'):
        super().__init__(agent_id=agent_id, profile=profile, topic=topic, topic_position=topic_position,
                         examples_topic=examples_topic)
        self._topic_profile_vector = topic_profile_vector

    def get_response_on_recommendation(self, paragraph) -> int:
        # LLM-based voting logic
        # return self.llm_vote(paragraph.text, self.profile, self.topic_position)
        return

    @property
    def topic_profile_vector(self):
        return self._topic_profile_vector

    @topic_profile_vector.setter
    def topic_profile_vector(self, value):
        # Setter for agent's profile topic vector
        self._topic_profile_vector = value

    @staticmethod
    def update_all_topic_vectors(agents: List["LLMAgentWithTopics"],
                                 doc_topic_matrix: np.ndarray,
                                 stance_matrix: "StanceMatrix") -> List["LLMAgentWithTopics"]:
        from environment.stance import StanceMatrix

        """
        Given agents and doc-topic matrix, compute and update topic_profile_vector for each agent in the list based on their votes' stance.
        """
        # For each agent we get the known votes indices
        agent_doc_indices = {}
        for agent in agents:
            known_paragraphs = stance_matrix.get_known_paragraphs(agent_id=agent.agent_id)
            indices = [stance_matrix.paragraphs.index(p) for p in known_paragraphs]
            agent_doc_indices[agent.agent_id] = indices

        # For each agent we compute topical profile
        agent_profiles = compute_agents_profiles(doc_topic_matrix=doc_topic_matrix, agent_doc_indices=agent_doc_indices)

        # Attach topic-profile vector for all agents
        for a in agents:
            a.topic_profile_vector = agent_profiles[a.agent_id]

        return agents

    def __str__(self):
        base_str = super().__str__()
        return f"\n{base_str},\nSub-Topics: {self._topic_profile_vector}"


__all__ = ["Agent", "LLMAgent", "LLMAgentWithTopics"]

# Configs
configurations = [
    {"sparsity": 0.3, "num_agents": 20, "num_paragraphs": 20, "file_path": "...", "instance_id": "instance_0"},
    {"sparsity": 0.7, "num_agents": 20, "num_paragraphs": 20, "file_path": "...", "instance_id": "instance_1"},
    {"sparsity": 0.3, "num_agents": 40, "num_paragraphs": 20, "file_path": "...", "instance_id": "instance_2"},
    {"sparsity": 0.7, "num_agents": 40, "num_paragraphs": 20, "file_path": "...", "instance_id": "instance_3"}
]

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

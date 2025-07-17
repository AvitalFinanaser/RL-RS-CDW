"""
Feedback class to define an interface provide_feedback(paragraph, collaborator) -> FeedbackResult.
A FeedbackResult is a small dataclass containing fields like score (numeric reward contribution), text (optional textual feedback from an LLM, for logging).
It includes common helper methods like formatting a prompt for the LLM given a paragraph and stance.
"""
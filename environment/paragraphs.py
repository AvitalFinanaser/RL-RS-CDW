import numpy as np


class Paragraph:

    def __init__(self, text, paragraph_id, name):
        self._text = text
        self._paragraph_id = paragraph_id
        self._name = name

    def __eq__(self, other):
        """
        Defines equality between two paragraphs if they have the same id, name and text.
        """
        if isinstance(other, Paragraph):
            return self._paragraph_id == other._paragraph_id and self._text == other._text and self._name == other.name
        return False

    @property
    def paragraph_id(self):
        # Getter for paragraph's id
        return self._paragraph_id

    @paragraph_id.setter
    def paragraph_id(self, value):
        # Setter for paragraph's id
        self._paragraph_id = value

    @property
    def name(self):
        # Getter for paragraph's name
        return self._name

    @name.setter
    def name(self, value):
        # Setter for paragraph's name
        self._name = value

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        # Setter for paragraph's text
        self._text = value

    @staticmethod
    def get_paragraph_name_by_id(paragraph_id, paragraphs):
        # Search for the paragraph in the provided list
        for paragraph in paragraphs:
            if paragraph.paragraph_id == paragraph_id:
                return paragraph.name
        return None

    def __str__(self) -> str:
        """
        Returns the id, name and text of paragraph
        """
        return f"Paragraph: p{self._paragraph_id}, named: {self._name}, text: {self._text}"


class ParagraphWithScore(Paragraph):
    """
    Represents a paragraph with a score between 0 and 1.
    New Attributes:
        _score (float): The score of the paragraph.
    """

    def __init__(self, text, paragraph_id, name, score=0.0):
        """
        Initialize a ParagraphWithScore instance - inheritance  of Paragraph
        :param text: The text of the paragraph.
        :param score: The score of the paragraph in [0,1], default=0
        """
        super().__init__(text, paragraph_id, name)
        self._score = score

    def __eq__(self, other):
        """
        Equality check for ParagraphWithScore instances.
        :param other: Another ParagraphWithScore instance
        :return: True if equal, False otherwise
        """
        if isinstance(other, ParagraphWithScore):
            return super().__eq__(other) and self._score == other._score
        return False

    @property
    def score(self):
        """
        Getter for score.
        :return: The score of the paragraph
        """
        return self._score

    @score.setter
    def score(self, value):
        """
        Setter for score.
        :param value: The new value for the paragraph's score
        """
        self._score = value

    def __str__(self) -> str:
        """
        Returns the id, name and text of paragraph
        """
        return f"ID: {self._paragraph_id}, Name: {self._name}, Text: {self._text}, Score: {self._score}"


class ParagraphWithTopics(Paragraph):
    """
    Represents a paragraph with a topic vector.
    New Attributes:
        _topic_vector (np.ndarray): The topic distribution vector of the paragraph.
    """

    def __init__(self, text: str, paragraph_id: int, name: str, topic_vector: np.ndarray):
        """
        Initialize a ParagraphWithTopics instance
        :param topic_vector: A vector representing topic distribution for the paragraph.
        """
        super().__init__(text, paragraph_id, name)
        self._topic_vector = topic_vector  # Store in private attribute

    def __eq__(self, other):
        """
        Equality check for ParagraphWithTopics instances.
        :param other: Another ParagraphWithTopics instance
        :return: True if equal, False otherwise
        """
        if isinstance(other, ParagraphWithTopics):
            return super().__eq__(other) and np.array_equal(self._topic_vector, other._topic_vector)
        return False

    @property
    def topic_vector(self):
        return self._topic_vector  # Return the private attribute

    @topic_vector.setter
    def topic_vector(self, value):
        self._topic_vector = value  # Set the private attribute

    def __str__(self) -> str:
        """
        Returns the id, name and text of paragraph
        """
        return f"Paragraph: p{self._paragraph_id}, named: {self._name}, text: {self._text},\nSub-topics:{self._topic_vector}"
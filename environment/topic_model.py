"""
Handles topic model initialization and inference.

stage 1: Collect and Preprocess the Data
Gather all paragraphs and clean the text by preprocessing.

stage 2: Vectorize the Text
Convert the preprocessed text into a numerical representation using TF-IDF or a similar technique.

stage 3: Run Topic Modeling (NMF)
For a range of potential topics (k), apply Non-negative Matrix Factorization to get the topic distributions for each paragraph.

stage 4: Label the Topics
Use a large language model to generate meaningful labels for each topic based on the top keywords.

stage 5: Determine Agent Preferences
For each agent, take the subset of paragraphs they've interacted with and average their topic distributions to get a personalized topic preference vector.

Integrate into RL Environment: Use these agent-specific topic preferences to inform a coverage metric in your reinforcement learning reward function, encouraging a balanced representation of topics that align with each agent’s interests.

"""

from typing import List, Optional, Dict, Any, Callable, Tuple
import numpy as np
from numpy.linalg import norm
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim.models.phrases import Phrases, Phraser
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon


# 1. Pre-process

def preprocess_text(
        text: str,
        additional_stopwords: Optional[List[str]] = None) -> str:
    """
    Preprocesses text docu,ent: cleaning, tokenization, lemmatization and stop words removal.
    """
    # 1. Lowercase and remove digits/punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 2. Tokenization
    tokens = word_tokenize(text)

    # 3. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 4. Remove stopwords and short tokens (length shorter than 2)
    stopwords = set(nltk_stopwords.words("english"))
    if additional_stopwords:
        stopwords = set(stopwords)
        stopwords.update(additional_stopwords)
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2 and t.isalpha()]

    # 5. Join back for vectorizer
    cleaned_text = " ".join(tokens)
    return cleaned_text


def preprocess_paragraphs(paragraphs: List[str], min_bigram_count: Optional[int] = None,
                          additional_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Preprocesses a list of textual paragraphs: cleaning, tokenization, lemmatization, stopword removal, and optional bigram detection.
    """
    # Preprocess each paragraph
    token_lists = [preprocess_text(text=p, additional_stopwords=additional_stopwords).split() for p in paragraphs]

    # Optionally build and apply bigram model
    if min_bigram_count is not None:
        bigram_phrases = Phrases(sentences=token_lists, min_count=min_bigram_count, threshold=0.1)
        bigram_model = Phraser(bigram_phrases)
        token_lists = [bigram_model[doc] for doc in token_lists]

    # Join tokens back into strings for vectorizers
    cleaned_paragraphs = [" ".join(tokens) for tokens in token_lists]
    return cleaned_paragraphs


# 2. Optimal k search using coherence


# 2.1 - TF–IDF Vectorization
def vectorize_corpus(paragraphs: List[str], max_features: Optional[int] = None) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Vectorized preprocessed text with TF–IDF.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,  # recommended for short text
        stop_words=None,  # already applied
    )
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    return vectorizer, tfidf_matrix


# 2.2 - Model Selection (Optimal K by Coherence)


# k selection
def topic_diversity(topics: List[List[str]]) -> float:
    """
    Compute topic diversity (unique words across top topics).
    """
    all_words = [word for topic in topics for word in topic]
    unique_words = len(set(all_words))
    total_words = len(all_words)
    return unique_words / total_words


def select_best_k(tfidf_matrix, tokenized_texts, vectorizer, k_range=range(3, 21), top_n=10):
    """
    Select best k (number of topics) based on coherence and diversity, breaking ties by coherence.
    Prints all ranking and tie-break steps for transparency.
    """
    dictionary = Dictionary(tokenized_texts)
    coherence_scores, diversity_scores = [], []

    for k in k_range:
        print(f"Trying to run NMF with k = {k}")
        nmf = NMF(n_components=k, random_state=42, init='nndsvd', max_iter=500)
        W = nmf.fit_transform(tfidf_matrix)

        # Top words for each topic
        topics = [
            [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-top_n - 1:-1]]
            for topic in nmf.components_
        ]

        # Metrics
        cm = CoherenceModel(topics=topics, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())
        diversity_scores.append(topic_diversity(topics))

    # Rank (lower rank is better)
    coherence_ranks = np.argsort(np.argsort(-np.array(coherence_scores)))
    diversity_ranks = np.argsort(np.argsort(-np.array(diversity_scores)))
    combined_ranks = coherence_ranks + diversity_ranks

    # Tie-breaking logic - candidates with minimum combined rank
    min_rank = np.min(combined_ranks)
    tie_indices = np.where(combined_ranks == min_rank)[0]

    if len(tie_indices) == 1:
        # No tie
        best_idx = tie_indices[0]
    else:
        # Tie exists - use coherence as tie-breaker
        coherences_of_ties = np.array(coherence_scores)[tie_indices]

        # Find the index within ties that has highest coherence
        best_idx_within_ties = np.argmax(coherences_of_ties)
        best_idx = tie_indices[best_idx_within_ties]

    best_k = list(k_range)[best_idx]

    return {
        "best_k": best_k,
        "coherence_scores": coherence_scores,
        "diversity_scores": diversity_scores,
        "k_range": list(k_range),
        "combined_ranks": combined_ranks.tolist(),
        "tie_indices": tie_indices.tolist()
    }


# Plot coherence and diversity scores for analysis
def plot_k_evaluation(k_range, best_k, coherence_scores, diversity_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, coherence_scores, marker='o', label='Coherence Score')
    plt.plot(k_range, diversity_scores, marker='x', label='Diversity Score')
    plt.xticks(k_range, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Number of Topics (k)', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.title(f'Coherence and Diversity vs. Number of Topics - Best Number of Topics: {best_k}', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.savefig('best_k')


# 3. Final NMF Model Fitting

def fit_nmf(tfidf_matrix, vectorizer, k, top_n=10):
    """
    Fit an NMF model and extract top words for each topic.
    Here we Normalize each row of document-topic matrix to sum to 1 for distribution.
    Returns:
        ndarray: Row-normalized document-topic distributions.
    """
    nmf = NMF(n_components=k, random_state=42, init='nndsvd')
    W = nmf.fit_transform(tfidf_matrix)  # Document-topic matrix
    H = nmf.components_  # Topic-term matrix

    # Normalize W rows to sum to 1 (topic distribution per document)

    # Avoid division by zero
    row_sums = W.sum(axis=1, keepdims=True)
    # If a row sum is zero (unlikely for TF-IDF), replace with 1 to avoid NaNs
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums

    # Extract top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = []
    for topic_vec in H:
        top_indices = topic_vec.argsort()[::-1][:top_n]
        topic_keywords.append([feature_names[i] for i in top_indices])

    return nmf, W_norm, topic_keywords


# 4. Auto-label topics (simple join or LLM callback)

def auto_label_topics(topic_keywords: List[List[str]], llm_fn: Optional[Callable[[List[str]], str]] = None) -> List[
    str]:
    if llm_fn is None:
        return [", ".join(words) for words in topic_keywords]
    return [llm_fn(words) for words in topic_keywords]


# 5. Topic distribution profile

def compute_agents_profiles(doc_topic_matrix: np.ndarray, agent_doc_indices: Dict[Any, List[int]]) -> Dict[
    Any, np.ndarray]:
    """
    Computes agent topic profiles as the mean of their selected document-topic distributions.
    Args:
        doc_topic_matrix (ndarray): Normalized document-topic matrix (docs × topics).
        agent_doc_indices (dict): Mapping agent_id -> list of document indices they engaged with.
    Returns:
        agent_profiles: dict of agent_id -> topic profile vector (length = n_topics)
    """
    n_topics = doc_topic_matrix.shape[1]
    agent_profiles = {}
    for agent, indices in agent_doc_indices.items():
        if indices:
            # Average topic vectors of docs the agent has seen/voted on
            agent_profiles[agent] = doc_topic_matrix[indices].mean(axis=0)
        else:
            # If no docs for agent, assign uniform distribution
            agent_profiles[agent] = np.ones(n_topics) / n_topics
    return agent_profiles


def compute_community_profile(doc_topic_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the community topic profile as the mean of all document-topic distributions.
    Args: doc_topic_matrix (ndarray): Normalized document-topic matrix (docs × topics).
    Returns: ndarray of length n_topics (mean over all docs).
    """
    return doc_topic_matrix.mean(axis=0)


# 6. Topic-Coverage reward function

def agent_reward_cosine(agent_profile: np.ndarray, community_profile: np.ndarray) -> float:
    """
    Compute reward as the inverse average cosine distance (similarity).
    Args:
        agent_profile: Agent’s topic distribution vector.
        community_profile: Community topic distribution vector.
    Returns:
        float: Cosine similarity ∈ [-1,1] (higher means more alignment).
    """
    # Compute cosine similarity safely
    dot = np.dot(agent_profile, community_profile)

    norm_prod = norm(agent_profile) * norm(community_profile)
    if norm_prod == 0:
        return 0.0
    return dot / norm_prod


def agent_reward_jsd(agent_profile: np.ndarray, community_profile: np.ndarray) -> float:
    """
    Compute reward as (1 - JSD) alignment between agent and community topic distributions.
    Args:
        agent_profile: Agent’s topic distribution vector (normalized).
        community_profile: Community topic distribution vector (normalized).
    Returns:
        float: 1 - Jensen-Shannon distance (∈ [0,1]); higher means more aligned.
    """
    js_dist = jensenshannon(agent_profile, community_profile, base=2)
    return 1 - js_dist


def compute_paragraphs_topic_matrix(
        paragraphs: List[str],
        k_range=range(3, 21),
        min_bigram_count: Optional[int] = None,
        additional_stopwords: Optional[List[str]] = None,
        max_features: Optional[int] = None,
        top_n: int = 10) -> Tuple[np.ndarray, int, List[List[str]]]:
    """
    Computes the normalized document-topic matrix from a list of Paragraph objects.
    Returns:
        doc_topic_matrix: ndarray (n_paragraphs x n_topics), normalized.
        best_k: int, optimal number of topics chosen.
        topic_keywords: list of top words per topic.
    """
    # Step 1: Preprocess paragraphs
    cleaned = preprocess_paragraphs(paragraphs=paragraphs, min_bigram_count=min_bigram_count, additional_stopwords=additional_stopwords)
    cleaned_token_lists = [doc.split() for doc in cleaned]
    # Step 2: Vectorize corpus
    vectorizer, tfidf_matrix = vectorize_corpus(cleaned, max_features=max_features)
    # Step 3: Select best k
    k_results = select_best_k(tfidf_matrix=tfidf_matrix, tokenized_texts=cleaned_token_lists, vectorizer=vectorizer, k_range=k_range, top_n=top_n)
    best_k = k_results["best_k"]
    # Step 4:  Fit Final NMF Model
    nmf_model, doc_topic_matrix, topic_keywords = fit_nmf(tfidf_matrix, vectorizer, best_k, top_n=top_n)
    return doc_topic_matrix, best_k, topic_keywords


# Example usage:
if __name__ == "__main__":
    # Example toy paragraphs
    P = [
        "Cats are lovely animals and make great pets in the stock market.",
        "Dogs are friendly and loyal companions.",
        "The stock market crashed due to bad economic news.",
        "Investing in stocks can be risky but rewarding.",
        "Cats purr when they are happy.",
        "Dogs bark to communicate.",
        "Economic indicators point to a recovery in the stock market."
    ]
    # Step 1: Preprocess paragraphs
    cleaned = preprocess_paragraphs(paragraphs=P, min_bigram_count=2)
    # Or, without bigrams:
    cleaned = preprocess_paragraphs(paragraphs=P)

    # Step 2: Vectorize corpus
    vectorizer, tfidf_matrix = vectorize_corpus(cleaned)

    # Step 3: Select best k
    cleaned_token_lists = [doc.split() for doc in cleaned]
    k_results = select_best_k(tfidf_matrix, cleaned_token_lists, vectorizer, k_range=range(2, 5))
    best_k = k_results["best_k"]

    print(f"Best k found: {best_k}")
    print("Coherence scores per k:", k_results["coherence_scores"])
    print("Diversity scores per k:", k_results["diversity_scores"])

    # Plot coherence and diversity scores
    plot_k_evaluation(k_range=k_results["k_range"], best_k=best_k, coherence_scores=k_results["coherence_scores"],
                      diversity_scores=k_results["diversity_scores"])

    # Step 4:  Fit Final NMF Model
    nmf_model, doc_topic_matrix, topic_keywords = fit_nmf(tfidf_matrix=tfidf_matrix, vectorizer=vectorizer, k=best_k)

    print(f"Top words for {len(topic_keywords)} topics:")
    for idx, words in enumerate(topic_keywords):
        print(f"Topic {idx}: {words}")

    # Of size number of documents X number of topics
    print("Document-topic matrix shape:", doc_topic_matrix.shape)

    # Step 5: Compute_profiles

    # Profiles (new step):
    agent_doc_indices = {
        'agent1': [0, 2, 4],
        'agent2': [1, 3, 5]
    }

    community_profile = compute_community_profile(doc_topic_matrix=doc_topic_matrix)
    print(f"community profile \n{community_profile}")

    agent_profiles = compute_agents_profiles(doc_topic_matrix=doc_topic_matrix, agent_doc_indices=agent_doc_indices)
    print(f"agent profiles \n{agent_profiles}")

    # Step 6: Topic reward function
    for agent in agent_doc_indices:
        reward_cosine = agent_reward_cosine(agent_profile=agent_profiles[agent], community_profile=community_profile)
        reward_jsd = agent_reward_jsd(agent_profile=agent_profiles[agent], community_profile=community_profile)
        print(f"{agent} -> cosine: {reward_cosine:.4f}, jsd: {reward_jsd:.4f}")

import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
from gpt_index.embeddings.openai import get_embedding
save_file_name ='../olympics-data/olympics_sections_question_answer_embedding_zh.csv'

EMBEDDING_MODEL = "text-embedding-ada-002"
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
    return {
        (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query,EMBEDDING_MODEL)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities
save_file_name ='../data/olympics_sections_document_embeddings_example.csv'

document_embeddings = load_embeddings(save_file_name)

order_document_sections_by_query_similarity("Who won the men's high jump?", document_embeddings)[:5]

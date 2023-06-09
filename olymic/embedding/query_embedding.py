import json
import os

import numpy as np
import openai
import pandas as pd
import pickle
import tiktoken
from dotenv import load_dotenv
from gpt_index.embeddings.openai import get_embedding
save_file_name ='../olympics-data/olympics_sections_question_answer_embedding_zh.csv'

load_dotenv()
openai.api_key=os.environ.get('OPENAI_API_KEY')
EMBEDDING_MODEL = "text-embedding-ada-002"
def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.

    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """

    similary =np.dot(np.array(x), np.array(y))
    print(similary)
    return similary


def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = len(df.iloc[0]['embedding_vector']) - 2
    return {
        (r.context, r.heading): json.loads(r['embedding_vector'] ) for i in range(max_dim + 1) for _, r in df.iterrows()
    }

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[
    (float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    print(openai.api_key)
    query_embedding = get_embedding(query,EMBEDDING_MODEL)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

document_embeddings = load_embeddings(save_file_name)


question = '2020年夏季奥林匹克运动会那个国家获得的金牌最多?'

#获取最相似的一个文档
array = order_document_sections_by_query_similarity(question, document_embeddings)[:1]

for index in array:
    print(index)

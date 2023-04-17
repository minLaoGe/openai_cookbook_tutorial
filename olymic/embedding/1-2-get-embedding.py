import os

import openai
import pandas as pd
from dotenv import load_dotenv

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

load_dotenv()

file_name ='../olympics-data/olympics_sections_question_answer_zh.csv'
save_file_name ='../olympics-data/olympics_sections_question_answer_embedding_zh.csv'

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    df['embedding_vector'] = df['content'].apply(lambda x: get_embedding(x))
    df.to_csv(save_file_name,index=False)
    return '1'


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

print("openai_key",os.environ.get("OPENAI_API_KEY"))
df = pd.read_csv(file_name, header=0)

document_embeddings = compute_doc_embeddings(df)
print(document_embeddings)

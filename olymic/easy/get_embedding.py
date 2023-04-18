import os

import openai
from dotenv import load_dotenv


load_dotenv()
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key=os.environ.get("OPENAI_API_KEY")
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
        model=model,
        input=text
    )
    return result["data"][0]["embedding"]


print(get_embedding("小明上午吃了包子"))
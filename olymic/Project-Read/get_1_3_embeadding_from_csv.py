import openai
import numpy as np
import tiktoken
from itertools import islice


EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

EMBEDDING_MODEL = 'text-embedding-ada-002'

long_text = 'AGI ' * 5000

def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator
def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]
try:
   average_embedding_vector = len_safe_get_embedding(long_text, average=True)
   print(average_embedding_vector)
except openai.InvalidRequestError as e:
    print(e)
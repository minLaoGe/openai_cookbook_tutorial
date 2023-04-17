import os
import openai
from dotenv import load_dotenv

load_dotenv()

print(openai.api_key)

res = openai.File.delete("file-kwvCjzF8auOuPJc95TePsFnn")
res2 =openai.File.delete("file-ognwIlmVuXyghHxgpz4KZBym")
res2 =openai.File.delete("file-UcPIa9RmumH5zALG0p3vg8VU")
print(res)
print(res2)
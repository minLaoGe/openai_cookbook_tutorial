import openai

load_dotenv()

## list all fine-tunie
res = openai.FineTune.list()
print(res)

"""
2.
"""
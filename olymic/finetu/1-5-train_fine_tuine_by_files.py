import requests
import json
import openai


load_dotenv()

## train a fine-tunine
res = openai.FineTune.create(training_file="file-TZgRl3zCONzorN51cH4dCwsT",model='davinci',suffix='olymic')

print(res)




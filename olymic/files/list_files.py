import os
import openai
load_dotenv()



res = openai.File.list()
print(res)
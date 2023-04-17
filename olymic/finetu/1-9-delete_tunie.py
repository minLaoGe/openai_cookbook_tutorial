import os
import openai
load_dotenv()


res = openai.Model.delete("davinci:ft-wahaha:sdfsd-2023-04-11-12-25-48")
res = openai.Model.delete("davinci:ft-wahaha:sdfsd-2023-04-11-12-34-20")
print(res)
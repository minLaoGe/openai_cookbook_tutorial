import textract
from extract_1_1_to_trunks import text_chunks
import os
import numpy as np
import openai
import tiktoken
import pandas as pd
from pathlib import Path




load_dotenv()

document = '<document>'
template_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
Use the following format， for example :\n0. Who is the author\n1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR\n\nDocument: \"\"\"{document}\"\"\"\n\n0. Who is the author: Tom Anderson (Page 1)\n1.'''

# template_prompt=f'''Extract key pieces of information and summarize the sentence and give this setence a head  from this regulation document.
#  and split them by \"KEY WORDS\" and \"SUMMARIZE\" and \"HEAD\"
# If a particular piece of information is not present, output \"Not specified\".
# Document: \"\"\"{document}\"\"\"   \n\n'''


results = []

model = 'text-davinci-003'
embedding_modle = 'gpt-3.5-turbo-0301'
def extract_chunk(document, template_prompt):
    prompt = template_prompt.replace('<document>', document)

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return "1." + response['choices'][0]['text']


for chunk in text_chunks:
    results.append(extract_chunk(chunk, template_prompt)+"\n content:"+chunk)
    # print(chunk)
    print('result:',results[-1])

data_list = []



def caculate_token(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return caculate_token(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return caculate_token(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    num_tokens += len(encoding.encode(messages))
    return num_tokens

for reslut in results:
    # 将数据块按行拆分
    lines = reslut.split('\n')

    # 从文本中提取数据
    data = {}

    for line in lines:

        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            data[key] = value
    data_list.append(data)
    data['count_token']=caculate_token(reslut, embedding_modle)

# 将字典列表转换为DataFrame
df = pd.DataFrame(data_list)

# 将DataFrame保存为CSV文件
df.to_csv('output.csv', index=False)


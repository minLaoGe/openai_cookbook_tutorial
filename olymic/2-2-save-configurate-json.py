import pandas as pd
import json
import re

df = pd.read_csv('olympics-data/olympics_sections_question_answer.csv')

formatted_row= []
for index, row in df.iterrows():
    formatted_row = {
        "prompt"
    }

questions = []
answers = []
for index, row in df.iterrows():
    # 提取 questions 和 answers 列
    questions.extend(row['questions'].split('\n'))
    answers.extend(row['answers'].split('\n'))


# 定义一个函数来移除前面的序号
def remove_number(text):
    return re.sub(r'^\d+\.', '', text).strip()


# 将 JSON 数据保存到本地文件
# 将问题和答案组装成 JSON 格式并保存到本地文件
with open("output.jsonl", "w") as f:
    for q, a in zip(questions, answers):
        json_data = {"prompt": remove_number(q)+'\n\n Answer It:', "completion": remove_number(a)+"\n"}
        f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

print("JSON 文件已保存到 output.json")
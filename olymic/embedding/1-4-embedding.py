import openai
from dotenv import load_dotenv

from query_embedding import array



load_dotenv()
self_model = 'text-davinci-003';
def answer_question(question,content):
    model_question = f"answer the question: \n {question}\n base on the content below:\n content: {content} \n \nAnswer It:"

    response = openai.Completion.create(
        model=self_model,
        prompt=model_question,
    )
    return response["choices"][0]["text"]

question = '2020年夏季奥林匹克运动会那个国家获得的金牌最多?'
for key,value in array[0]:
    for key2,value2 in value:
        print(answer_question(question,key2))
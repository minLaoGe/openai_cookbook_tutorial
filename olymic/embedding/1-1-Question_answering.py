import os

import openai
from dotenv import load_dotenv



load_dotenv()
openai.api_key=os.environ.get("OPENAI_API_KEY")
self_model = 'text-davinci-003'
question = '2020年夏季奥林匹克运动会那个国家获得的金牌最多?'
def answer_question(question):
    model_question = f"answer the question: \n {question}\n  Answer It:"
    print(model_question)
    response = openai.Completion.create(
        model=self_model,
        prompt=model_question,
    )
    return response["choices"][0]["text"]


print(answer_question(question))

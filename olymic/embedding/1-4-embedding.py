import openai
from dotenv import load_dotenv

from query_embedding import array, question

load_dotenv()
self_model = 'text-davinci-003';
def answer_question(question,content):
    model_question = f"answer the question: \n {question}\n base on the content below:\n content: {content} \n \nAnswer It:"
    print(model_question)
    response = openai.Completion.create(
        model=self_model,
        prompt=model_question,
    )
    return response["choices"][0]["text"]


for element in array[0]:
    if isinstance(element, tuple):
        for index, value in enumerate(element):
            if index==0:
                print(answer_question(question,value))

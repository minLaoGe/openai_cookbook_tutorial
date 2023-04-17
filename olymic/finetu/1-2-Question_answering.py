import  openai
from dotenv import load_dotenv

model = 'text-davinci-003'

load_dotenv()
def     create_context(content):
    response = openai.Completion.create(
        prompt=content,
        model=model,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["text"]
print(create_context("Which country won the most gold medals during the 2020 Summer Olympics?"))
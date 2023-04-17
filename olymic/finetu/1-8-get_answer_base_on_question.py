import openai

load_dotenv()

self_model = 'davinci:ft-personal:olymic-2023-04-15-12-26-09';

def answer_question(question):
    model_question = f"{question}\n\nAnswer It:"

    response = openai.Completion.create(
        model=self_model,
        prompt=model_question,
    )
    return response["choices"][0]["text"]

question = 'Which country won the most gold medals during the 2020 Summer Olympics?'
print(answer_question(question))
import  openai

load_dotenv()

modle_id = 'davinci:ft-wahaha:sdfsd-2023-04-11-12-25-48'

res= openai.Completion.create(
  model=modle_id,
  prompt="What is the total amount of loss insured for the 2020 Games?",
  max_tokens=2020,
  temperature=0
)
print(res)
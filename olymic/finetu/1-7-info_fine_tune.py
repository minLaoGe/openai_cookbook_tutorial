import openai
load_dotenv()

modle_id = 'ft-zYnNecSnUIudBKb2NEJzGMQC'
### getInfo about fine_tune job.
res_quest = openai.FineTune.retrieve(id=modle_id)
print(res_quest)
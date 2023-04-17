import pandas as pd
import openai




# df = pd.read_csv('olympics-data/olympics_sections.csv')
df = pd.read_csv('olympics-data/tune-0-0.csv')
df['context'] = df.title + "\n" + df.heading + "\n\n" + df.content
print(df.head())

def get_questions(context):
    try:
        print(f"Write questions by Chinese based on the text below\n\nText: {context}\n\nQuestions:\n1.")
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f"基于下文提出10个问题，并且通过这些问题可以高度概括文章，请用中文提问\n\nText: {context}\n\nQuestions:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    except:
        return ""


df['questions']= df.context.apply(get_questions)
df['questions'] = "1." + df.questions
print(df[['questions']].values[0][0])


def get_answers(row):
    try:
        print(f"Write answer using chinese based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1.")
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f"Write answer based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']
    except Exception as e:
        print (e)
        return ""


df['answers']= df.apply(get_answers, axis=1)
df['answers'] = "1." + df.answers
df = df.dropna().reset_index().drop('index',axis=1)
print(df[['answers']].values[0][0])

# df.to_csv('olympics-data/olympics_sections_question_answer.csv', index=False)
df.to_csv('olympics-data/olympics_sections_question_answer_zh.csv', index=False)
# df = df[df.tokens<2000]
# df[['context', 'tokens']].rename(columns={'context':'text','tokens':'metadata'}).to_json('olympics-data/olympics_search.jsonl', orient='records', lines=True)
#
# search_file = openai.File.create(
#   file=open("olympics-data/olympics_search.jsonl"),
#   purpose='fine-tune'
# )
# olympics_search_fileid = search_file['id']
#
# print("文件id为:",olympics_search_fileid)
#
# print(create_context("Where did women's 4 x 100 metres relay event take place during the 2020 Summer Olympics?", olympics_search_fileid, max_len=400))
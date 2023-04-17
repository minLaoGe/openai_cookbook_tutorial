import random
import  openai
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('olympics-data/olympics_sections_question_answer.csv')


"""
sklearn.model_selection 的 train_test_split作用
train_test_split函数用于将数据划分为训练数据和测试数据。

train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train_data和test_data，形式为：

X_train,X_test, y_train, y_test =

train_test_split(train_data ,  train_target ,  test_size=0.4,   random_state=0)

参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，

其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
"""
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
len(train_df), len(test_df)

df['context']='\n\n###\n\n'.join(df['context'])
print(df.context.str.contains('\n\n###\n\n').sum())

olympics_search_fileid = 'file-SeuSZrYRM9ZEdEMfaEZ3Q9OK'
def get_random_similar_contexts(question, context, file_id=olympics_search_fileid, search_model='ada', max_rerank=10):
    """
    Find similar contexts to the given context using the search file
    """
    try:
        results = openai.Engine(search_model).search(
            search_model=search_model,
            query=question,
            max_rerank=max_rerank,
            file=file_id
        )
        candidates = []
        for result in results['data'][:3]:
            if result['text'] == context:
                continue
            candidates.append(result['text'])
        random_candidate = random.choice(candidates)
        return random_candidate
    except Exception as e:
        print(e)
        return ""


def create_fine_tuning_dataset(df, discriminator=False, n_negative=1, add_related=False):
    """
    Create a dataset for fine tuning the OpenAI model; either for a discriminator model,
    or a model specializing in Q&A, where it says if no relevant context is found.

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the question, answer and context pairs
    discriminator: bool
        Whether to create a dataset for the discriminator
    n_negative: int
        The number of random negative samples to add (using a random context)
    add_related: bool
        Whether to add the related contexts to the correct context. These are hard negative examples

    Returns
    -------
    pd.DataFrame
        The dataframe containing the prompts and completions, ready for fine-tuning
    """
    rows = []
    for i, row in df.iterrows():
        for q, a in zip(("1." + row.questions).split('\n'), ("1." + row.answers).split('\n')):
            if len(q) > 10 and len(a) > 10:
                if discriminator:
                    rows.append(
                        {"prompt": f"{row.context}\nQuestion: {q[2:].strip()}\n Related:", "completion": f" yes"})
                else:
                    rows.append({"prompt": f"{row.context}\nQuestion: {q[2:].strip()}\nAnswer:",
                                 "completion": f" {a[2:].strip()}"})

    for i, row in df.iterrows():
        for q in ("1." + row.questions).split('\n'):
            if len(q) > 10:
                for j in range(n_negative + (2 if add_related else 0)):
                    random_context = ""
                    if j == 0 and add_related:
                        # add the related contexts based on originating from the same wikipedia page
                        subset = df[(df.title == row.title) & (df.context != row.context)]

                        if len(subset) < 1:
                            continue
                        random_context = subset.sample(1).iloc[0].context
                    if j == 1 and add_related:
                        # add the related contexts based on the most similar contexts according to the search
                        random_context = get_random_similar_contexts(q[2:].strip(), row.context, search_model='ada',
                                                                     max_rerank=10)
                    else:
                        while True:
                            # add random context, which isn't the correct context
                            random_context = df.sample(1).iloc[0].context
                            if random_context != row.context:
                                break
                    if discriminator:
                        rows.append(
                            {"prompt": f"{random_context}\nQuestion: {q[2:].strip()}\n Related:", "completion": f" no"})
                    else:
                        rows.append({"prompt": f"{random_context}\nQuestion: {q[2:].strip()}\nAnswer:",
                                     "completion": f" No appropriate context found to answer the question."})

    return pd.DataFrame(rows)


for name, is_disc in [('discriminator', True), ('qa', False)]:
    for train_test, dt in [('train', train_df), ('test', test_df)]:
        print(train_df);
        print(test_df)
        ft = create_fine_tuning_dataset(dt, discriminator=is_disc, n_negative=1, add_related=True)
        ft.to_json(f'{name}_{train_test}.jsonl', orient='records', lines=True)

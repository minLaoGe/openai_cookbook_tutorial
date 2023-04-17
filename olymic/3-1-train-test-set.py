from sklearn.model_selection import train_test_split
import pandas as pd

file_id = 'file-SeuSZrYRM9ZEdEMfaEZ3Q9OK'


df = pd.read_csv('olympics-data/olympics_sections_question_answer.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
len(train_df), len(test_df)

df['context']='\n\n###\n\n'.join(df['context'])
print(df.context.str.contains('\n\n###\n\n').sum())
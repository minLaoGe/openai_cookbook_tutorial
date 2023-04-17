import openai


load_dotenv()
search_file = openai.File.create(
  file=open("output_olmpics_zh.jsonl"),
  purpose='fine-tune'
)
olympics_search_fileid = search_file['id']

print("文件id为:",olympics_search_fileid)

file_id = 'file-TZgRl3zCONzorN51cH4dCwsT';
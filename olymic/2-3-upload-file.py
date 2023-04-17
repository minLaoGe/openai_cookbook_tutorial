import openai


load_dotenv()
search_file = openai.File.create(
  file=open("output.jsonl"),
  purpose='fine-tune'
)
olympics_search_fileid = search_file['id']

print("文件id为:",olympics_search_fileid)

file_id = 'file-9lZBygnhx00xcrjxGpAUgIoV';
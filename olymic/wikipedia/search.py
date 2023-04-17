import wikipedia


wikipedia.set_lang("zh")

res_ids = wikipedia.search('科学怪人',results=1);
print(res_ids)
res = wikipedia.page(res_ids[0],auto_suggest=False)

print(res.content)
print(res.links)
import logging

import pandas as pd
import wikipedia
import re
from typing import Set
from transformers import GPT2TokenizerFast

import numpy as np
from nltk.tokenize import sent_tokenize


wikipedia.set_lang("zh")


def filter_olympic_2020_titles(titles):
    """
    Get the titles which are related to Olympic games hosted in 2020, given a list of titles
    """
    titles = [title for title in titles if '2020' in title or '奥林匹克' in title ]

    return titles


def get_wiki_page(title):
    """
    Get the wikipedia page given a title
    """
    try:
        search='';
        if len(all_pages) <1:
         search = title
        else:
            for item in all_pages:
                if item.title != title:
                    search = item.title
                    break
        print("title:",search)
        res_ids = wikipedia.search(search, results=1)[0];
        return wikipedia.page(title=res_ids,auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        logging.error("出错了",e)
        return wikipedia.page(e.options[0])
    except wikipedia.exceptions.PageError as e:
        return None


all_pages = []

def recursively_find_all_pages(titles, titles_so_far=set(), max_pages=50):
    """
    Recursively find all the pages that are linked to the Wikipedia titles in the list
    """


    titles = list(set(titles) - titles_so_far)
    titles = filter_olympic_2020_titles(titles)
    titles_so_far.update(titles)
    for title in titles:
        page = get_wiki_page(title)
        if page is None:
            continue
        all_pages.append(page)
        if len(all_pages) >= max_pages:
            break
        recursively_find_all_pages(page.links, titles_so_far, max_pages - len(all_pages))
        titles_so_far.update(page.links)
    return all_pages




pages = recursively_find_all_pages(["2020 Summer Olympics"])


"""sesession2"""

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))


def reduce_long(
        long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i]) + "."

    return long_text


# discard_categories = ['See also', 'References', 'External links', 'Further reading', "Footnotes",
#                       "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
#                       "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
#                       "References and notes", ]
discard_categories = [
    'See also', '另見', '另见',
    'References', '參考資料', '参考资料',
    'External links', '外部連結', '外部链接',
    'Further reading', '延伸閱讀', '延伸阅读',
    'Footnotes', '註釋', '注释',
    'Bibliography', '參考書目', '参考书目',
    'Sources', '來源', '来源',
    'Citations', '引用', '引用',
    'Literature', '文獻', '文献',
    'Notes and references', '註釋與參考資料', '注释与参考资料',
    'Photo gallery', '相片集', '相册',
    'Works cited', '引用作品', '引用作品',
    'Photos', '照片', '照片',
    'Gallery', '畫廊', '画廊',
    'Notes', '筆記', '笔记',
    'References and sources', '參考資料與來源', '参考资料与来源',
    'References and notes', '參考資料與筆記', '参考资料与笔记',
]


def extract_sections(
        wiki_text: str,
        title: str,
        max_len: int = 1500,
        discard_categories: Set[str] = discard_categories,
) -> str:
    """
    Extract the sections of a Wikipedia page, discarding the references and other low information sections
    """
    if len(wiki_text) == 0:
        return []

    # find all headings and the coresponding contents
    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont) + 4)]

    # discard the discard categories, accounting for a tree structure
    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []
    for heading, content in zip(headings, contents):
        plain_heading = " ".join(heading.split(" ")[1:-1])
        num_equals = len(heading.split(" ")[0])
        if num_equals <= keep_group_level:
            keep_group_level = max_level

        if num_equals > remove_group_level:
            if (
                    num_equals <= keep_group_level
            ):
                continue
        keep_group_level = max_level
        if plain_heading in discard_categories:
            remove_group_level = num_equals
            keep_group_level = max_level
            continue
        nheadings.append(heading.replace("=", "").strip())
        ncontents.append(content)
        remove_group_level = max_level

    # count the tokens of each section
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    # Create a tuple of (title, section_name, content, number of tokens)
    outputs += [(title, h, c, t) if t < max_len
                else (title, h, reduce_long(c, max_len), count_tokens(reduce_long(c, max_len)))
                for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)]

    return outputs


# Example page being processed into sections


# Example section
# ber[-1]


"""session3"""


res = []
for page in pages:
    res += extract_sections(page.content, page.title)
df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
df = df[df.tokens>40]
df = df.drop_duplicates(['title','heading'])
df = df.reset_index().drop('index',axis=1) # reset index
df.head()


df.to_csv('olympics-data/tune-0-0.csv', index=False)



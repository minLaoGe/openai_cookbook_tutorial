import textract
import os
import openai
import tiktoken
from pathlib import Path
from gpt_index import download_loader


true_file_name = '../data/The web application hacker’s handbook finding and exploiting security flaws (Dafydd Stuttard, Marcus Pinto) (Z-Library).pdf'

test_file_name = '../data/fia_f1_power_unit_financial_regulations_issue_1_-_2022-08-16.pdf'

# Extract the raw text from each PDF using textract
text = textract.process(test_file_name, method='pdfminer').decode('utf-8')
clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';',' ')



# PDFReader = download_loader("PDFReader")
#
# loader = PDFReader()
# documents = loader.load_data(file=Path(true_file_name))
# clean_text=''
# for text in  documents:
#     print('text: ' + text .text+ "\n\n")
    # clean_text = text.text.replace("  ", " ").replace("\n", "; ").replace(';',' ')



# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


tokenizer = tiktoken.get_encoding("cl100k_base")

results = []

chunks = create_chunks(clean_text, 1000, tokenizer)
temp_arr = [tokenizer.decode(chunk) for chunk in chunks]

text_chunks=temp_arr[:5]


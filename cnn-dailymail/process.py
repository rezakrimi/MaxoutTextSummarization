import os
import nltk
from tqdm import tqdm
import json
from collections import defaultdict
import pickle

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

directory = './dailymail/stories/'

examples = list()
counter = defaultdict(int)

for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.story'):
        file = open(os.path.join(directory, filename), 'r')
        is_highlight = False
        content_finished = False
        content = ''
        highlights = []
        for line in file:
            line = line.strip()
            if not content_finished and line:
                content = content + ' ' + line
            if is_highlight and line:
                highlights.append(line)
                is_highlight = False
            if line.startswith('@highlight'):
                is_highlight = True
                content_finished = True
        content = word_tokenize(content[1:].replace("''", '" ').replace("``", '" ').lower())
        highlight = ''
        for i, h in enumerate(highlights):
            if highlight != '':
                highlight += '\n'
            if i < len(highlights)-1:
                highlight += h + '##'
            if i == len(highlights)-1:
                highlight += h + '.'
        example = {
            'context_tokens': content,
            'ques_tokens': word_tokenize(highlight.replace("''", '" ').replace("``", '" ').lower())
            }
        for token in content:
            counter[token] += len(highlights)
        for token in highlight:
            counter[token] += 1
        examples.append(example)
# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(example, f, ensure_ascii=False, indent=4)
# file.close()

with open('dm_examples.pkl', 'wb') as f:
    pickle.dump(examples, f)

with open('dm_counter.pkl', 'wb') as f:
    pickle.dump(counter, f)

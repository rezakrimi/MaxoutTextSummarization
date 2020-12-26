import sys
import pickle
from collections import defaultdict
from random import shuffle

sys.path.insert(0, '../')

import config
from data_utils import (make_conll_format2, make_embedding, make_vocab,
                        make_vocab_from_dm, process_file)




def make_sent_dataset():
    train_src_file = "./para-train.txt"
    train_trg_file = "./tgt-train.txt"

    embedding_file = "./glove.840B.300d.txt"
    embedding = "./embedding.pkl"
    word2idx_file = "./word2idx.pkl"
    # make vocab file
    word2idx = make_vocab(train_src_file, train_trg_file, word2idx_file, config.vocab_size)
    make_embedding(embedding_file, embedding, word2idx)


def make_para_dataset():
    embedding_file = "./glove.840B.300d.txt"
    embedding = "./embedding.pkl"
    src_word2idx_file = "./word2idx.pkl"

    train_squad = "../squad/train-v1.1.json"
    dev_squad = "../squad/dev-v1.1.json"

    train_src_file = "../squad/para-train.txt"
    train_trg_file = "../squad/tgt-train.txt"
    dev_src_file = "../squad/para-dev.txt"
    dev_trg_file = "../squad/tgt-dev.txt"

    test_src_file = "../squad/para-test.txt"
    test_trg_file = "../squad/tgt-test.txt"

    # pre-process training data
    # train_examples have passage question pairs, counter is the word frequency across all passages
    # question and passages are represented as a list of tokens
    with open('../cnn-dailymail/cnn_examples.pkl', 'rb') as f:
        cnn = pickle.load(f)
    print('loaded cnn')
  #  with open('../cnn-dailymail/dm_examples.pkl','rb') as f:
 #       dm = pickle.load(f)
#    print('loaded dailymail')

    counter = defaultdict(int)

    examples = cnn
    shuffle(examples)
    train_size = int(len(examples) * 0.92)
    train_examples = examples[:train_size]
    print(len(train_examples))
    dev_test_examples = examples[train_size:]
    print(len(train_examples), len(dev_test_examples))
    for e in train_examples:
        for token in e['context_tokens']:
            counter[token] += 1
    make_conll_format2(train_examples, train_src_file, train_trg_file)
    # make a dict mapping word to unique index
    word2idx = make_vocab_from_dm(src_word2idx_file, counter, config.vocab_size)
    # makes a dict mapping words from all passages to embedding vectors
    make_embedding(embedding_file, embedding, word2idx)

    # split dev into dev and test
    # random.shuffle(dev_test_examples)
    num_dev = len(dev_test_examples) // 2
    dev_examples = dev_test_examples[:num_dev]
    test_examples = dev_test_examples[num_dev:]
    make_conll_format2(dev_examples, dev_src_file, dev_trg_file)
    make_conll_format2(test_examples, test_src_file, test_trg_file)


if __name__ == "__main__":
    # make_sent_dataset()
    make_para_dataset()


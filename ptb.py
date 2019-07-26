from pdb import set_trace
import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
#from nltk.tokenize import TweetTokenizer
from gensim.utils import simple_tokenize
import pandas as pd

class TweetTokenizer:
    def __init__(self, *args, **kwargs):
        pass

    def tokenize(self, input):
        #return [t for t in simple_tokenize(input)]
        return input.split(' ')

from utils import OrderedCounter

import re
letter_pat = re.compile(r'[a-zA-Z0-9]')

from sklearn.externals import joblib

def rewrite_letters(url):
    return letter_pat.sub('a', url)

tok_pat = re.compile(r'[0-9a-zA-Z]+')
split_pat = re.compile(r'([0-9a-zA-Z]+)')
def to_length(match):
    return str(len(match.group()))

def split_delimeters(splitted):
    for tok in splitted:
        if tok_pat.match(tok):
            yield tok
        else:
            for c in tok:
                yield c

def rewrite_to_toklen(url):
    rewritten = tok_pat.sub(to_length, url)
    splitted = split_pat.split(rewritten)
    return ' '.join(split_delimeters(splitted))

tokenizer = TweetTokenizer(preserve_case=False)
max_sequence_length = 60
w2i = {}

from contextlib import closing
from multiprocessing import Pool

def parallel_apply(df, key, funct, output_key, n_jobs=200):
    if output_key in df.columns:
        del df[output_key]

    inputs = df[key].unique()
    process_number = n_jobs
    with closing(Pool(process_number)) as pool:
        outputs = pool.map(funct, inputs)
    pool.close()
    pool.join()
    output = pd.DataFrame()
    output[key] = inputs
    output[output_key] = outputs
    return df.merge(output, on=key)

def preprocess(line):
    words = tokenizer.tokenize(rewrite_to_toklen(line))
    return words_to_input(words, line)

def preprocess_char(line):
    words = [c for c in line]
    return words_to_input(words)

def words_to_input(words, url):
    input = ['<sos>'] + words
    input = input[:max_sequence_length]

    target = words[:max_sequence_length-1]
    target = target + ['<eos>']

    assert len(input) == len(target), "%i, %i"%(len(input), len(target))
    length = len(input)

    input.extend(['<pad>'] * (max_sequence_length-length))
    target.extend(['<pad>'] * (max_sequence_length-length))

    input = [w2i.get(w, w2i['<unk>']) for w in input]
    target = [w2i.get(w, w2i['<unk>']) for w in target]

    return {'input': input, 'target': target, 'length': length, 'url': url}

class PTB(Dataset):
    def __init__(self, ptb_file=None, vocab_file=None, train_with_vocab=False, train_file=None,**kwargs):
        Dataset.__init__(self)
        if ptb_file is not None:
            return joblib.load(ptb_file)

        self.train_with_vocab = train_with_vocab
        self.train_file = train_file
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)
        self.w2i = None
        self.iw2 = None

        self.vocab_file = vocab_file
        if self.vocab_file is not None:
            self.create_vocab(self.vocab_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length'],
            'url': self.data[idx]['url']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def dump(self, ptb_file=''):
        if not ptb_file:
            ptb_file = '%s.ptb'%(self.vocab_file)
        joblib.dump(self, ptb_file)

    def create_data(self, df):
        assert self.w2i is not None

        global tokenizer
        global w2i
        global max_sequence_length

        tokenizer = TweetTokenizer(preserve_case=False)
        w2i = self.w2i
        max_sequence_length = self.max_sequence_length
        
        df = df
        df = parallel_apply(df, 'url', preprocess, 'preprocess', n_jobs=16)
        #df = parallel_apply(df, 'url', preprocess_char, 'preprocess', n_jobs=16)
        self.data = df['preprocess'].values.tolist()

    def create_vocab(self, vocab_file):
        self.vocab_file = vocab_file
        df = pd.DataFrame()

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.vocab_file, 'r') as file:
            lines = []
            for i, line in enumerate(file):
                lines.append(line)

                line = rewrite_to_toklen(line)
                words = tokenizer.tokenize(line)
                #words = [c for c in line]

                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        self.w2i = w2i
        self.i2w = i2w

        self.dump()

        if self.train_with_vocab:
            df['url'] = lines
            self.create_data(df)

        elif self.train_file is not None:
            df = pd.read_csv(self.train_file, names=['url'])
            self.create_data(df)

class PTBDataset(PTB):
    def __init__(self, ptb, **kwargs):
        Dataset.__init__(self)
        self.data = ptb.data
        self.w2i = ptb.w2i
        self.i2w = ptb.i2w
        del ptb.data
#PTB('./data', 'train', create_data=True)

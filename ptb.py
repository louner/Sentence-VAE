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
    line = rewrite_to_toklen(line)
    words = tokenizer.tokenize(line)

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

    return {'input': input, 'target': target, 'length': length}

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        Dataset.__init__(self)
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.raw_data_path = os.path.join(data_dir, 'ptb.'+split+'.txt')
        self.data_file = 'ptb.'+split+'.json'
        self.vocab_file = 'ptb.vocab.json'

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input': np.asarray(self.data[idx]['input']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
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


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        global tokenizer
        global w2i
        global max_sequence_length

        tokenizer = TweetTokenizer(preserve_case=False)
        w2i = self.w2i
        max_sequence_length = self.max_sequence_length
        
        df = pd.read_csv(self.raw_data_path, names=['url'])
        df = parallel_apply(df, 'url', preprocess, 'preprocess', n_jobs=200)

        data = defaultdict(dict)
        for item in df['preprocess']:
            data[len(data)] = item

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        tokenizer = TweetTokenizer(preserve_case=False)

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        with open(self.raw_data_path, 'r') as file:

            for i, line in enumerate(file):
                line = rewrite_to_toklen(line)

                words = tokenizer.tokenize(line)
                w2c.update(words)

            for w, c in w2c.items():
                if c > self.min_occ and w not in special_tokens:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()

#PTB('./data', 'train', create_data=True)

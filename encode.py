from pdb import set_trace
import pandas as pd
from ptb import *
from model import *
from train import *
from utils import batch_to_var, batch_to_list
from sklearn.externals import joblib

import sys

def read_txt(txt_file):
    with open(txt_file) as f:
        lines = f.read().split('\n')
    df = pd.DataFrame()
    df['url'] = lines
    return df

if __name__ == '__main__':
    model_fpath, test_file = sys.argv[1:]

    model = joblib.load(model_fpath)
    if torch.cuda.is_available():
        model.cuda()
    else:
        torch.set_num_threads(8)

    #test = pd.read_csv(test_file, names=['url'], engine='python', error_bad_lines=False, warn_bad_lines=False)
    urls = read_txt(test_file).values.flatten().tolist()
    batches = model.encode_urls(urls)

    batches.to_csv('%s.encoder_last'%(test_file))

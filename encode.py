from pdb import set_trace
import pandas as pd
from ptb import *
from model import *
from train import *
from utils import batch_to_var
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

    #test = pd.read_csv(test_file, names=['url'], engine='python', error_bad_lines=False, warn_bad_lines=False)
    test = read_txt(test_file)
    model.ptb.create_data(test)

    data_loader = DataLoader(dataset=model.ptb, batch_size=1024, shuffle=False, num_workers=cpu_count())

    encoded = []
    for iteration, batch in enumerate(data_loader):
        batch_to_var(batch)
        logp, mean, logv, z, encoder_last = model(batch['input'], batch['length'])
        z = encoder_last
        #logp, mean, logv, z = model(to_var(batch['input']), to_var(batch['length']))
        encoded.append(z.tolist())
        print(iteration, z.size())
    encoded = np.vstack(encoded)
    joblib.dump(encoded, '%s.encoder_last'%(test_file))

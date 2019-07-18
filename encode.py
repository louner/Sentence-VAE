import pandas as pd
from ptb import *
from model import *
from train import *
from sklearn.externals import joblib

import sys

if __name__ == '__main__':
    model_fpath, data_dir = sys.argv[1:]

    model = joblib.load(model_fpath)

    test = PTB(data_dir, 'test', create_data=True)
    data_loader = DataLoader(dataset=test, batch_size=32, shuffle=False, num_workers=cpu_count())

    encoded = []
    for iteration, batch in enumerate(data_loader):
        logp, mean, logv, z = model(batch['input'], batch['length'])
        encoded.append(z.tolist())
    encoded = np.vstack(encoded)
    np.savetxt('%s.encoded.npy'%(test.raw_data_path), encoded)

import pandas as pd
from ptb import *
from model import *
from train import *
from sklearn.externals import joblib

import sys

if __name__ == '__main__':
    model_fpath, data_dir, create_data = sys.argv[1:]

    model = joblib.load(model_fpath)
    set_trace()
    if torch.cuda.is_available():
        model.cuda()

    test = PTB(data_dir, 'test', create_data=create_data=='1')
    data_loader = DataLoader(dataset=test, batch_size=4096, shuffle=False, num_workers=cpu_count())

    encoded = []
    for iteration, batch in enumerate(data_loader):
        #logp, mean, logv, z, encoder_last, decoder_last = model(batch['input'], batch['length'])
        logp, mean, logv, z = model(to_var(batch['input']), to_var(batch['length']))
        encoded.append(z.tolist())
        print(iteration, z.size())
    encoded = np.vstack(encoded)
    np.savetxt('%s.encoded.npy'%(test.raw_data_path), encoded)

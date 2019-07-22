import pandas as pd
from ptb import *
from model import *
from train import *
from sklearn.externals import joblib

import sys

if __name__ == '__main__':
    model_fpath, test_file = sys.argv[1:]

    model = joblib.load(model_fpath)
    if torch.cuda.is_available():
        model.cuda()

    test = pd.read_csv(test_file, names=['url'], engine='python', error_bad_lines=False)
    model.ptb.create_data(test)

    data_loader = DataLoader(dataset=model.ptb, batch_size=4096, shuffle=False, num_workers=cpu_count())

    encoded = []
    for iteration, batch in enumerate(data_loader):
        #logp, mean, logv, z, encoder_last, decoder_last = model(batch['input'], batch['length'])
        logp, mean, logv, z = model(to_var(batch['input']), to_var(batch['length']))
        encoded.append(z.tolist())
        print(iteration, z.size())
    encoded = np.vstack(encoded)
    np.savetxt('%s.encoded.npy'%(test_file), encoded)

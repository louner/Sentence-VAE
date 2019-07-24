vocab_file="data/char/10m/10m_vocab.txt"
train_file="data/char/10m/10m_train.txt"
max_sequence_length="200"

python train.py -vf $vocab_file -tf $train_file --create_data -bs 1024 --max_sequence_length $max_sequence_length -bin data/char/10m/model

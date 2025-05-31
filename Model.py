import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_df = pd.read_csv("../train_final.csv")
dev_df = pd.read_csv("../dev_final.csv")
test_df = pd.read_csv("../test_final.csv")

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
char_dict = {aa: idx+1 for idx, aa in enumerate(amino_acids)}

def integer_encoding(df, seq_col='sequence'):
    encoded_seqs = []
    for seq in df[seq_col]:
        encoded_seq = [char_dict.get(aa, 0) for aa in seq]
        encoded_seqs.append(np.array(encoded_seq))
    return encoded_seqs

def prepare_sequences(df, seq_col='sequence', maxlen=200):
    encoded = integer_encoding(df, seq_col)
    padded = pad_sequences(encoded, maxlen=maxlen, padding='post', truncating='post')
    return padded

train_pad = prepare_sequences(train_df)
val_pad = prepare_sequences(dev_df)
test_pad = prepare_sequences(test_df)

print(train_pad.shape, val_pad.shape, test_pad.shape)

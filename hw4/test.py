from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Bidirectional
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import sys
import os
import h5py
import pickle
import numpy as np
import pandas as pd
import gensim

np.set_printoptions(suppress=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #titan x

def main(argv):
    file_input = open(argv[0])
    text_input = file_input.read()
    file_input.close()
    data_all = text_input.split('\n')
    x_test = []

    for i in range(1,len(data_all),1):
        if data_all[i] != "":
            temp = data_all[i].split(',', 1)
            x_test.append(temp[1])

    word2vec = gensim.models.Word2Vec.load('embedding')

    for i in range(0, len(x_test), 1):
        x_test[i] = text_to_word_sequence(x_test[i])
        for j in range(0, len(x_test[i]), 1):
            x_test[i][j] = word2vec[x_test[i][j]]
    
    max_len = 40
    x_test = pad_sequences(x_test, maxlen=max_len, dtype=float, padding='post', truncating='post', value=0.)
    
    model = load_model('model.hdf5')
    y_pred = model.predict(x_test, batch_size=1024, verbose=1)

    print ("=====================")
    print (y_pred)
    print ("=====================")
    y_ans = np.argmax(y_pred, axis=1)
    
    df_table = []
    for i in range(0,y_ans.shape[0],1):
        df_table.append([i, y_ans[i]])

    df = pd.DataFrame.from_records(df_table, columns = ["id","label"])
    df.to_csv(argv[1], index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
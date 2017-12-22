import sys
import os
import pandas as pd
import numpy as np
import tensorflow,keras.backend.tensorflow_backend
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, Dot, Add

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #titan x

def get_model(n_user, n_item, latent):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    user_vec = Embedding(n_user, latent, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_item, latent, embeddings_initializer='random_normal')(item_input)
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_user, 1, embeddings_initializer='zeros')(user_input)
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_user, 1, embeddings_initializer='zeros')(item_input)
    item_bias = Flatten()(item_bias)

    r_hat = Dot(axes=1)([user_vec, item_vec])
    r_hat = Add()([r_hat, user_bias, item_bias])
    model = keras.models.Model([user_input, item_input], r_hat)
    model.compile(loss='mse', optimizer='Adam')
    model.summary()
    return model

def main(argv):
    data_table = pd.read_csv(argv[0])
    max_user = data_table['UserID'].drop_duplicates().max()
    max_movie = data_table['MovieID'].drop_duplicates().max()
    input_data = data_table.sample(frac = 1., random_state = 168464)
    users = input_data['UserID'].values - 1
    movies = input_data['MovieID'].values - 1
    ratings = input_data['Rating'].values

    # for d in [64, 128, 256, 512]:
    MF_model = get_model(max_user, max_movie, 512)
    
    earlystopping = EarlyStopping('val_loss', patience = 3, verbose = 1)
    checkpoint = ModelCheckpoint("MF_best_model", verbose = 1 ,save_best_only = True, monitor = 'val_loss')
    history = MF_model.fit([users, movies], ratings, validation_split = 0.1, batch_size = 1024, epochs = 30, callbacks=[checkpoint,earlystopping])

    MF_model.save('MF_model')

if __name__ == '__main__':
    main(sys.argv[1:])
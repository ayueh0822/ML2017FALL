import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import cluster
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, normalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pickle
import h5py
np.set_printoptions(suppress=True)

def train_autoencoder(encoding_dim, x_train):
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='relu')(decoded)
    autoencoder = Model(input_img, decoded)

    #encoder model
    encoder = Model(input_img, encoded)

    #decoder model
    # encoded_input = Input(shape=(encoding_dim,))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adadelta', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[early_stopping]
                    )
    return encoder


def main(argv):
    all_data = np.load('image.npy')

    # all_data = all_data.astype('float32') / 255.
    encoder_model = train_autoencoder(16, all_data)

    # encoder_model.save('encoder_16')
    decoder_model.save('decoder_32')
    # encoder_model = load_model('./model/encoder_32')

    dim_reduced_data = encoder_model.predict(all_data)
    kmeans_model =  cluster.KMeans(n_clusters=2,verbose=1).fit(dim_reduced_data)
    # cluster_labels = kmeans_model.labels_

    with open('./kmeans_model_16.pickle','wb') as f:
        pickle.dump(kmeans_model,f)

if __name__ == '__main__':
    main(sys.argv[0:])
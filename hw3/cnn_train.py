import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, normalization
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import h5py
np.set_printoptions(suppress=True)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #1080
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #titan x

def main(argv):
    data_table = pd.read_csv(argv[0])
    data_table = np.array(data_table)
    # np.random.shuffle(data_table)
    # print (data_table.shape)

    x_all_data = np.zeros((data_table.shape[0], 48*48))
    y_all_data = np.zeros((data_table.shape[0], 7))
    for i in range(0,data_table.shape[0],1):
        x_all_data[i] = data_table[i,1].split(' ')
        y_all_data[i, data_table[i,0]] = 1

    x_all_data = np.array(x_all_data, dtype=float)
    y_all_data = np.array(y_all_data, dtype=float)

    valid_ratio = 0.1
    valid_seg = int(x_all_data.shape[0] * valid_ratio)
    x_train = x_all_data[valid_seg:]
    y_train = y_all_data[valid_seg:]
    x_valid = x_all_data[0:valid_seg]
    y_valid = y_all_data[0:valid_seg]

    ### reshape for CNN input ###
    x_train = np.reshape(x_train, (-1,48,48,1))
    x_valid = np.reshape(x_valid, (-1,48,48,1))

    ### image generator ###
    img_data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.10,
            height_shift_range=0.10,
            shear_range=0.10,
            zoom_range=0.10,
            horizontal_flip=True,
            fill_mode='nearest')
    img_data_gen.fit(x_train)
    train_generator = img_data_gen.flow(x_train, y_train, batch_size=256)

    ### build CNN structure ##
    model = Sequential()
    model.add(Conv2D(64, (5,5), input_shape=(48,48,1), padding='valid', activation='relu'))
    model.add(ZeroPadding2D(padding=(2,2), data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))
    model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))

    model.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
    model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))

    model.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))

    model.add(Conv2D(128, (3,3), padding='valid', activation='relu'))
    model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))

    model.add(Conv2D(128, (3,3), padding='valid', activation='relu'))
    model.add(ZeroPadding2D(padding=(1,1), data_format='channels_last'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())

    ### build DNN structure ##
    for i in range(0,2,1):
        model.add(Dense(units=1024, activation='relu'))
        model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))
    model.summary()
    # opt = 'adam'
    opt = Adam(lr=1e-4)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # history = model.fit(x_all_data, y_all_data, validation_split=0.1, epochs=100, batch_size=128)
    # history = model.fit_generator(train_generator, samples_per_epoch=x_train.shape[0]*2, nb_epoch=150, validation_data=(x_valid, y_valid), callbacks=[early_stopping])
    cnn_history = model.fit_generator(train_generator, samples_per_epoch=x_train.shape[0]*2, nb_epoch=150, validation_data=(x_valid, y_valid))

    model.save('cnn_model')

    ### draw acc-epoch and loss-epoch plots ### 
    # plt.plot(cnn_history.history['acc'])
    # plt.plot(cnn_history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'valid'], loc='upper left')
    # plt.savefig('cnn_acc_plot.png')
    # plt.clf()
    # plt.plot(cnn_history.history['loss'])
    # plt.plot(cnn_history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'valid'], loc='upper left')
    # plt.savefig('cnn_loss_plot.png')
    # plt.clf()

if __name__ == '__main__':
    main(sys.argv[1:])
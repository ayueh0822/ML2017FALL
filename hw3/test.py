import sys
import os
import pandas as pd
import numpy as np
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
import h5py
np.set_printoptions(suppress=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #titan x

def main(argv):
    data_table = pd.read_csv(argv[0])
    data_table = np.array(data_table)

    x_test = np.zeros((data_table.shape[0], 48*48))
    for i in range(0,data_table.shape[0],1):
        x_test[i] = data_table[i,1].split(' ')

    x_test = np.array(x_test, dtype=float)
    x_test = np.reshape(x_test, (data_table.shape[0],48,48,1))
    x_test_rev = np.flip(x_test, axis=2)
    # model = load_model('./model/cnn_model_alldata')
    model = load_model('./cnn_model_alldata')

    y_pred = model.predict(x_test)
    y_pred_rev = model.predict(x_test_rev)

    y_pred_total = y_pred + y_pred_rev
    df_table = []
    for i in range(0,x_test.shape[0],1):
        label = y_pred_total[i].argmax()
        df_table.append([i, label])

    df = pd.DataFrame.from_records(df_table, columns = ["id","label"])
    df.to_csv(argv[1], index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
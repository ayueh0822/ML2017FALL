import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

def main(argv):
    x_data_table = pd.read_csv(argv[4])
    x_data_table = np.array(x_data_table)

    feature_select = [i for i in range(0, 106, 1)]
    # feature_select2 = [i for i in range(0, 106, 1)]
    feature_select2 = [0,1,3,4,5]
    x_test = np.zeros((x_data_table.shape[0],0))
    for i in range(0, 106, 1):
        feature_vec = x_data_table[:,i].reshape(x_data_table.shape[0],1)
        if i in feature_select:
            x_test = np.concatenate((x_test, feature_vec), axis=1)
        if i in feature_select2:
            if i in [0,1,3,4,5]: # continuous value
                x_test = np.concatenate((x_test, feature_vec ** 2), axis=1)
            else: # 0/1 value
                x_test = np.concatenate((x_test, feature_vec * 1), axis=1)

    ### normalization
    for i in range(0, x_test.shape[1], 1):
        mean_val = np.mean(x_test[:,i])
        std_val = np.std(x_test[:,i])
        if std_val != 0:
            x_test[:,i] = (x_test[:,i] - mean_val) / std_val

    w = np.load('./model/model_gw.npy')
    b = np.load('./model/model_gb.npy')
    df_table = []
    z = np.dot(x_test, w) + b
    sigmoid = 1 / (1 + np.exp(-z))
    for i in range(0,x_test.shape[0],1):
        if sigmoid[i] < 0.5:
            df_table.append([i+1, 1])
        else:
            df_table.append([i+1, 0])

    df = pd.DataFrame.from_records(df_table, columns = ["id","label"])
    df.to_csv(argv[5], index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
import sys
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
import pickle

np.set_printoptions(suppress=True)


def main(argv):
    x_data_table = pd.read_csv(argv[4])
    x_data_table = np.array(x_data_table, dtype=float)
    x_test = np.array(x_data_table, dtype=float)

    # for i in [0,1,3,4,5]:
    #     feature_vec = x_data_table[:,i].reshape(x_data_table.shape[0],1)
    #     x_test = np.concatenate((x_test, feature_vec**2), axis=1)
    ### normalization
    # for i in range(0, x_test.shape[1], 1):
    #     mean_val = np.mean(x_test[:,i])
    #     std_val = np.std(x_test[:,i])
    #     if std_val != 0:
    #         x_test[:,i] = (x_test[:,i] - mean_val) / std_val
    ### min-max
    # for i in range(0, x_test.shape[1], 1):
    #     min_val = np.min(x_test[:,i])
    #     max_val = np.max(x_test[:,i])
    #     if (max_val - min_val) != 0:
    #         x_test[:,i] = (x_test[:,i] - min_val) / (max_val - min_val)

    with open('./model/model_sk.pickle','rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(x_test)
    
    df_table = []
    for i in range(0,x_test.shape[0],1):
        if prediction[i] < 0.5:
            df_table.append([i+1, 0])
        else:
            df_table.append([i+1, 1])

    df = pd.DataFrame.from_records(df_table, columns = ["id","label"])
    df.to_csv(argv[5], index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
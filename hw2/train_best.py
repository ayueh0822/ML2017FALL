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
from sklearn import tree
import pickle

np.set_printoptions(suppress=True)

x_data_table = pd.read_csv('X_train')
x_data_table = np.array(x_data_table, dtype=float)
x_all = np.array(x_data_table, dtype=float)

# for i in [0,1,3,4,5]:
#     feature_vec = x_data_table[:,i].reshape(x_data_table.shape[0],1)
#     x_all = np.concatenate((x_all, feature_vec**2), axis=1)

### normalization
# for i in range(0, x_all.shape[1], 1):
#     mean_val = np.mean(x_all[:,i])
#     std_val = np.std(x_all[:,i])
#     if std_val != 0:
#         x_all[:,i] = (x_all[:,i] - mean_val) / std_val

### min-max
# for i in range(0, x_all.shape[1], 1):
#     min_val = np.min(x_all[:,i])
#     max_val = np.max(x_all[:,i])
#     if (max_val - min_val) != 0:
#         x_all[:,i] = (x_all[:,i] - min_val) / (max_val - min_val)

y_data_table = pd.read_csv('Y_train')
y_all = np.array(y_data_table).reshape(y_data_table.shape[0])

# training and validation
x_train, x_valid, y_train, y_valid = train_test_split(x_all,y_all,test_size=0.3)

# model = tree.DecisionTreeClassifier()
model = AdaBoostClassifier()
# model = RandomForestClassifier()
model.fit(x_train,y_train)

# Accuracy
accuracy = accuracy_score(model.predict(x_valid), y_valid)
print (accuracy)
model.fit(x_all,y_all)
with open('./model/model_sk.pickle','wb') as f:
    pickle.dump(model,f)
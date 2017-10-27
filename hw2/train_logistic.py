import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

training_time = 10000
learning_rate = 1e-5

x_data_table = pd.read_csv('X_train')
x_data_table = np.array(x_data_table, dtype=float)

feature_select = [i for i in range(0, 106, 1)]
feature_select2 = [i for i in range(0, 106, 1)]
x_all = np.zeros((x_data_table.shape[0],0))
for i in range(0, 106, 1):
    feature_vec = x_data_table[:,i].reshape(x_data_table.shape[0],1)
    if i in feature_select:
        x_all = np.concatenate((x_all, feature_vec), axis=1)
    if i in feature_select2:
        if i in [0,1,3,4,5]: # continuous value
            x_all = np.concatenate((x_all, feature_vec ** 2), axis=1)
        else: # 0/1 value
            x_all = np.concatenate((x_all, feature_vec * 4), axis=1)

### normalization
for i in range(0, x_all.shape[1], 1):
    mean_val = np.mean(x_all[:,i])
    std_val = np.std(x_all[:,i])
    if std_val != 0:
        x_all[:,i] = (x_all[:,i] - mean_val) / std_val

### min-max
# for i in range(0, x_all.shape[1], 1):
#     min_val = np.min(x_all[:,i])
#     max_val = np.max(x_all[:,i])
#     if (max_val - min_val) != 0:
#         x_all[:,i] = (x_all[:,i] - min_val) / (max_val - min_val)

x_all = np.concatenate((x_all, np.ones((x_all.shape[0],1))), axis=1) #bias
y_data_table = pd.read_csv('Y_train')
y_all = np.array(y_data_table).reshape(y_data_table.shape[0])

# training set
x_train = np.array(x_all[5000:, :])
y_train = np.array(y_all[5000:])
# validation set
x_valid = np.array(x_all[0:5000])
y_valid = np.array(y_all[0:5000])

# x_train = x_all
# y_train = y_all
w = np.zeros(x_all.shape[1])

### training
for i in range(0, training_time, 1):
    z = np.dot(x_train, w)
    sigmoid = 1 / (1 + np.exp(-z))
    cross_entropy =  - ( np.log(sigmoid) * y_train +  np.log(1-sigmoid) * (1-y_train)  )
    loss = np.sum(cross_entropy)
    
    gradient = - np.dot( (y_train - sigmoid), x_train )
    w = w - learning_rate * gradient

    if i%1000 == 0:
        print ("train", i, ":", loss)
print (loss)

### validation
y_test = np.zeros(y_valid.shape[0])
z_test = np.dot(x_valid, w)
sigmoid_test = 1 / (1 + np.exp(-z_test))

for i in range(0,y_valid.shape[0],1):
    if sigmoid_test[i] > 0.5:
        y_test[i] = 1

### calculate validation accuracy
accuracy = 0
for i in range(0,y_valid.shape[0],1):
    if y_test[i] == y_valid[i]:
        accuracy += 1

accuracy = accuracy / y_valid.shape[0]

np.save('./model/model_4.npy',w)
print ("w_list =", w.tolist())
print ("accuracy =", accuracy)

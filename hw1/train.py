import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

training_time = 100000
learning_rate = 5e-9

data_table = pd.read_csv('train.csv', encoding='Big5')
data_hour = []
cnt = 0
for j in range(0, len(data_table), 18):
    for i in range(3, 27, 1):
        data_hour.append([])
        for k in range(0, 18, 1):
            if 'NR' == data_table.loc[k+j][i]:
                data_hour[cnt].append(0.0)
            else:
                data_hour[cnt].append(float(data_table.loc[k+j][i]))
        cnt += 1

x_list = []
y_list = []
# feature_select = [9]
# feature_select2 = []
feature_select = [0,7,8,9,11]
feature_select2 = [8,9]
previous_hour = 9
it = 0
for i in range(0, len(data_hour), 1):
    if (i+previous_hour) % (24*20) < previous_hour:
        continue
    x_list.append([])
    for j in range(0, previous_hour, 1):
        for k in range(0, 18, 1):
            if k in feature_select:
                x_list[it].append(data_hour[i+j][k])
            if k in feature_select2:
                x_list[it].append(data_hour[i+j][k]**2)
    x_list[it].append(1) # bias
    y_list.append(data_hour[i+previous_hour][9])
    # print("x[{0}]: {1} y[{0}]: {2}".format(it, x_list[it], y_list[it]))
    it += 1

fold_data_len = int(it/12)
x_train = np.array(x_list[0 : fold_data_len*11])
y_train = np.array(y_list[0 : fold_data_len*11])
x_valid = np.array(x_list[fold_data_len*11 : fold_data_len*12])
y_valid = np.array(y_list[fold_data_len*11 : fold_data_len*12])
w = np.zeros( ( len(feature_select) + len(feature_select2) ) * previous_hour +1 )
re_lambda = 0.0001
# print (x_valid.shape, y_valid.shape, w.shape)

#### start training w
for i in range(0, training_time, 1):
    loss = 0
    gradient = np.zeros(( len(feature_select) + len(feature_select2) ) * previous_hour)
    
    # get wx by using inner product
    wx_train = np.dot(x_train, w)

    # get Loss func = Σ(y - (wx + b) )^2
    error = y_train - wx_train
    loss = np.sum(error**2) + re_lambda * np.sum(w**2)
    
    # gradient and bias
    gradient = -2 * np.dot(error, x_train)
    gradient += 2 * re_lambda * w

    # update w and bias
    w = w - learning_rate * gradient / (fold_data_len*12)

    if i%10000 == 0:
        print("train", i,":", loss)
print ( "loss =", loss )
print ( "RMSE_train =", np.sqrt(loss/len(y_train)) )
#### close form
# w = np.dot( np.dot(np.linalg.inv(np.dot(x_train.T, x_train)), x_train.T), y_train)

#### validation data testing
y_test = np.dot(x_valid, w)
RMSE = np.sqrt( np.mean( (y_valid - y_test)**2 ) )

print ("RMSE_validation =", RMSE)
print ("w_list =", w.tolist())

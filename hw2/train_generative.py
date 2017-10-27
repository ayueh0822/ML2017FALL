import sys
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

x_data_table = pd.read_csv('X_train')
x_data_table = np.array(x_data_table, dtype=float)

feature_select = [i for i in range(0, 106, 1)]
# feature_select2 = [i for i in range(0, 106, 1)]
feature_select2 = [0,1,3,4,5]
x_all = np.zeros((x_data_table.shape[0],0))
for i in range(0, 106, 1):
    feature_vec = x_data_table[:,i].reshape(x_data_table.shape[0],1)
    if i in feature_select:
        x_all = np.concatenate((x_all, feature_vec), axis=1)
    if i in feature_select2:
        if i in [0,1,3,4,5]: # continuous value
            x_all = np.concatenate((x_all, feature_vec ** 2), axis=1)
        else: # 0/1 value
            x_all = np.concatenate((x_all, feature_vec * 1), axis=1)

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

x_dim = x_all.shape[1]
y_data_table = pd.read_csv('Y_train')
y_all = np.array(y_data_table).reshape(y_data_table.shape[0])

# training set
x_train = x_all
y_train = y_all

### classify
x_class0 = []
x_class1 = []
for i in range(0, y_train.shape[0], 1):
    if y_train[i] == 0:
        x_class0.append(x_train[i])
    else:
        x_class1.append(x_train[i])

x_class0 = np.array(x_class0)
x_class1 = np.array(x_class1)
N0 = x_class0.shape[0]
N1 = x_class1.shape[0]

### calculate each mean
mean_class0 = np.zeros(x_dim)
mean_class1 = np.zeros(x_dim)
for i in range(0, x_dim, 1):
    mean_class0[i] = np.mean(x_class0[:,i])
    mean_class1[i] = np.mean(x_class1[:,i])

## calcute covariance matrix
covmat_class0 = np.zeros((x_dim, x_dim))
covmat_class1 = np.zeros((x_dim, x_dim))
covmat_mutual = np.zeros((x_dim, x_dim))
### Σ1
for i in range(0,N0,1):
    cov_vec = x_class0[i] - mean_class0
    cov_vec = np.reshape(cov_vec, (x_dim,1))
    covmat_class0 += np.matmul( cov_vec, cov_vec.T )
### Σ2
for i in range(0,N1,1):
    cov_vec = x_class1[i] - mean_class1
    cov_vec = np.reshape(cov_vec, (x_dim,1))
    covmat_class1 += np.matmul( cov_vec, cov_vec.T )

covmat_mutual = (covmat_class0 + covmat_class1) / (N0 + N1)

# covmat_class0_l = np.cov(x_class0.T)
# covmat_class1_l = np.cov(x_class1.T)
# covmat_mutual_l = (covmat_class0_l * N0 + covmat_class1_l * N1 ) / (N0 + N1) 

### calcuate generative model
mean_class0 = np.reshape( mean_class0, (x_dim,1) )
mean_class1 = np.reshape( mean_class1, (x_dim,1) )

w = np.matmul( (mean_class0 - mean_class1).T , np.linalg.inv(covmat_mutual) )
b = ((-0.5) * np.dot( np.matmul( mean_class0.T, np.linalg.inv(covmat_mutual) ), mean_class0 ) 
    + (0.5) * np.dot( np.matmul( mean_class1.T, np.linalg.inv(covmat_mutual) ), mean_class1 ) 
    + np.log(N0/N1))

w = w.reshape(x_dim)
b = b[0,0]

np.save('./model/model_gw.npy',w)
np.save('./model/model_gb.npy',b)

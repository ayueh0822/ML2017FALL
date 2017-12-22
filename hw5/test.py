import sys
import pandas as pd
import numpy as np
import tensorflow,keras.backend.tensorflow_backend
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Input, Flatten, Concatenate, Dot, Add

def main(argv):
    data_table = pd.read_csv(argv[0])
    max_user = data_table['UserID'].drop_duplicates().max()
    max_movie = data_table['MovieID'].drop_duplicates().max()
    users = data_table['UserID'].values - 1
    movies = data_table['MovieID'].values - 1

    
    model = load_model(argv[1])
    output = model.predict([users, movies])

    # normalize
    # mean_val = 3.58171208604
    # std_val = 1.11689766115
    # output = output * std_val + mean_val

    df_table = []
    for i in range(0,output.shape[0],1):
        if output[i,0] > 5.0:
            df_table.append([i+1, 5.0])
        elif output[i,0] < 1.0:
            df_table.append([i+1, 1.0])
        else:
            df_table.append([i+1, output[i,0]])

    df = pd.DataFrame.from_records(df_table, columns = ["TestDataID", "Rating"])
    df.to_csv(argv[2], index=False)

if __name__ == '__main__':
    main(sys.argv[1:])
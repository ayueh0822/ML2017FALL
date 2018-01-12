import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import cluster
import pickle

def show_img(image, idx):
    img = Image.new('L',(28,28))
    img_array = img.load()
    for i in range(28):
        for j in range(28):
            img_array[i,j] = int(image[idx][j*28+i])
    img.show()

def main(argv):
    test_data = pd.read_csv(argv[2])
    test_data = np.array(test_data)
    # print (test_data[0])

    with open(argv[4],'rb') as f:
        model = pickle.load(f)
    
    fout = open(argv[3], "w")
    fout.write("ID,Ans\n")
    cnt = 0
    for test_case in test_data:
        idx1 = test_case[1]
        idx2 = test_case[2]
        if model.labels_[idx1] == model.labels_[idx2]:
            fout.write(str(cnt) + "," + str(1) + "\n")
        else:
            fout.write(str(cnt) + "," + str(0) + "\n")
        cnt += 1
    fout.close()

    # img_data = np.load('image.npy')
    # show_img(img_data,test_data[2755][1])
    # show_img(img_data,test_data[2755][2])

    
if __name__ == '__main__':
    main(sys.argv[0:])
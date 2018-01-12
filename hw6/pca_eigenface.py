import sys
import os
import numpy as np
from skimage import io

def main(argv):
    X = []
    file_list = os.listdir(argv[1])
    # file_list.sort()

    for file_name in file_list:
        img_path = argv[1] + '/' + file_name 
        X.append(io.imread(img_path).flatten())
    X = np.array(X)

    # for i in range(0,415,1):
    #     # print ("processing",i)
    #     file_name = "./Aberdeen/" + str(i) + ".jpg"
    #     X.append(io.imread(file_name).flatten())
    # X = np.array(X)

    X_mean = np.mean(X, axis=0)
    # print("X.shape = ", X.shape)
    # print("X_mane.shape = ", X_mean.shape)
    U, s, V = np.linalg.svd((X - X_mean).T, full_matrices=False)
    # print ("U = \n", U)
    # print ("s = \n", s)
    # print ("V = \n", V)
    # np.save('U.npy', U)
    # np.save('s.npy', s)
    # np.save('V.npy', V)

    # U = np.load('U.npy')
    # s = np.load('s.npy')
    # V = np.load('V.npy')
    # print (U[:,0])

    img_path = argv[1] + '/' + argv[2]
    y = io.imread(img_path).flatten()
    y = y - X_mean
    w = np.zeros((4,))
    for i in range(0,4,1):
        w[i] = np.dot(y, U[:,i])

    reconstruct = np.zeros((600*600*3,))
    for i in range(0,4,1):
        reconstruct += w[i] * U[:,i]

    M = reconstruct + X_mean
    M -= np.min(M)
    M /= np.max(M)
    M = (M*255).astype(np.uint8)
    io.imsave("reconstruction.jpg", np.reshape(M, (600,600,3)).astype(np.uint8))

    '''
    1-1 mean face
    '''
    # img_mean = np.mean(X, axis=0)
    # io.imsave('./mean.jpg', np.reshape(img_mean,(600,600,3)).astype(np.uint8))

    '''
    1-2 eigenface
    '''
    # print(eigenface.shape)
    # for i in range(0,4,1):
    #     M = U[:,i]
    #     M -= np.min(M)
    #     M /= np.max(M)
    #     M = (M*255).astype(np.uint8)
    #     filename = "./face" + str(i) + ".jpg"
    #     io.imsave(filename, np.reshape(M,(600,600,3)).astype(np.uint8))

    '''
    1-3 reconstruct
    '''
    # for k in range(0,4,1):
    #     img_path = argv[1] + '/' + str(k) + '.jpg'
    #     y = io.imread(img_path).flatten()
    #     y = y - X_mean
    #     w = np.zeros((4,))
    #     for i in range(0,4,1):
    #         w[i] = np.dot(y, U[:,i])

    #     reconstruct = np.zeros((600*600*3,))
    #     for i in range(0,4,1):
    #         reconstruct += w[i] * U[:,i]

    #     M = reconstruct + X_mean
    #     M -= np.min(M)
    #     M /= np.max(M)
    #     M = (M*255).astype(np.uint8)
    #     filename = "./reconstruct_face" + str(k) + ".jpg"
    #     io.imsave(filename, np.reshape(M, (600,600,3)).astype(np.uint8))

    '''
    1-4 explain covariance ratio

    ratio 0 : 0.041446248382629634 = 4.2%
    ratio 1 : 0.029487322251120673 = 2.9%
    ratio 2 : 0.02387711293208416  = 2.4%
    ratio 3 : 0.022078415569025404 = 2.2%
    '''
    # for i in range(0,4,1):
    #     print ("ratio:", s[i] / np.sum(s))
    

if __name__ == '__main__':
    main(sys.argv[0:])
#!/bin/bash

wget 'https://github.com/ayueh0822/ML2017FALL/releases/download/0.0.1/embedding'
wget 'https://github.com/ayueh0822/ML2017FALL/releases/download/0.0.1/embedding.syn1neg.npy'
wget 'https://github.com/ayueh0822/ML2017FALL/releases/download/0.0.1/embedding.wv.syn0.npy'
wget 'https://github.com/ayueh0822/ML2017FALL/releases/download/0.0.1/model.hdf5'

python3 test.py $1 $2
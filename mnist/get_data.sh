#!/bin/sh
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gunzip -f train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

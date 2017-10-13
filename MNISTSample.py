# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 23:30:23 2017

@author: AMIR
"""

#!/usr/bin/env python

import numpy 
import theano
import theano.tensor as T
import lasagne as L

import sys
import os
import gzip
import pickle



PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'


def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    """Load data from `url` and store the result in `filename`."""
    if not os.path.exists(filename):
        print("Downloading MNIST dataset")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')


def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_data()
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]
    y_train = numpy.asarray(y_train, dtype=numpy.int32)
    y_valid = numpy.asarray(y_valid, dtype=numpy.int32)
    y_test = numpy.asarray(y_test, dtype=numpy.int32)

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1],
        output_dim=10,
    )

input_var = T.matrix(dtype=theano.config.floatX)
target_var = T.vector(dtype='int32')
network = L.layers.InputLayer((50000,784), input_var)
network = L.layers.DenseLayer(network, 100)
network = L.layers.DenseLayer(network, 10, nonlinearity=L.nonlinearities.softmax)
prediction = L.layers.get_output(network)
loss = L.objectives.aggregate(L.objectives.categorical_crossentropy(prediction, target_var), mode='mean')
#loss = L.objectives.categorical_crossentropy(prediction, target_var)
params = L.layers.get_all_params(network, trainable=True)
updates = L.updates.adam(loss, params, learning_rate=0.01)
train_fn = theano.function([input_var, target_var], loss, updates=updates)
test_fn = theano.function([input_var], L.layers.get_output(network, deterministic=True))

tloss = 0.0
ntrain = 0.0
tacc = 0.0
ntest = 0.0

input_batch = X_train
target_batch = y_train
	#print "target batch shape:",target_batch.shape
	#tloss += train_fn(input_batch, target_batch) 
tloss = train_fn(input_batch, target_batch)

input_batch = X_valid
val_output = test_fn(input_batch)
val_predictions = np.argmax(val_output, axis=1)

ntrain += 1
tacc += np.sum(val_predictions == y_valid)
print tacc/val_predictions.shape[0]



def main():
    data = load_data()
    print("Got %i testing datasets." % len(data['X_train']))
    nn_example(data)

if __name__ == '__main__':
    main()
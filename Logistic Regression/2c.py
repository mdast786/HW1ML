import numpy as np
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import keras.backend as K


n_hidden_layer = 2
n_nodes = 5

dataset = pd.read_csv('data/train.csv')
x_train = dataset.iloc[:, :-1]
y_train = dataset.iloc[:, -1].values

dataset = pd.read_csv('data/test.csv')
x_test = dataset.iloc[:, :-1]
y_test = dataset.iloc[:, -1].values


for n_nodes in [5, 10, 25, 50]:
    model = Sequential()
    
    tf.keras.initializers.Zeros()
    
    model.add(Dense(n_nodes, input_dim=4, activation='relu'))
    for k in range(n_hidden_layer):
        model.add(Dense(n_nodes, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    
    classifier = KerasClassifier(model=model)
    
    classifier.fit(x_train, y_train, nb_epochs=100, verbose = 0)
    
    preds = classifier.predict(x_test)
    acc = np.sum(preds.reshape(499,) == y_test)/500
    print("For #node", n_nodes, "\nTest accuracy: %.2f%%" % (acc * 100))



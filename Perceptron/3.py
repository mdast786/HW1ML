import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import os, shutil
import pandas as pd
np.random.seed(100)



def predict(X,w):
    return np.sign(np.dot(X, w[1:])+w[0])



def aperceptron_sgd(X, Y,epochs):    
    # initialize weights
    w = np.zeros(X.shape[1] )
    u = np.zeros(X.shape[1] )
    b = 0
    beta = 0

    # counters    
    final_iter = epochs
    c = 1
    converged = False

    # main average perceptron algorithm
    for epoch in range(epochs):
        # initialize misclassified
        misclassified = 0

        # go through all training examples
        for  x,y in zip(X,Y):
            h = y * (np.dot(x, w) + b)

            if h <= 0:
                w = w + y*x
                b = b + y

                u = u+ y*c*x
                beta = beta + y*c
                misclassified += 1

        # update counter regardless of good or bad classification        
        c = c + 1

        # break loop if w converges
        if misclassified == 0:
            final_iter = epoch
            converged = True
            #print("Averaged Perceptron converged after: {} iterations".format(final_iter))
            break

    if converged == False:
        print("Averaged Perceptron DID NOT converged.")

    # prints
    # print("final_iter = {}".format(final_iter))
    # print("b, beta, c , (b-beta/c)= {} {} {} {}".format(b, beta, c, (b-beta/c)))
    # print("w, u, (w-u/c) {} {} {}".format(w, u, (w-u/c)) )


    # return w and final_iter
    w = w - u/c
    b = np.array([b- beta/c])
    w = np.append(b, w)

    return w, final_iter


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# rename known columns
columns = ['variance', 'skewness', 'kurtosis', 'entropy', 'label']
df_train.columns = columns
df_test.columns = columns


# organize data into input and output
X_train = df_train.drop(columns="label")
y_train = df_train['label']
X_test = df_test.drop(columns="label")
y_test = df_test['label']


for max_iter in range(10):    
    w, final_iter = aperceptron_sgd(X_train.values, y_train, max_iter)

    pred = predict(X_test, w)
    error = np.sum(abs(pred - y_test))
    
    print(max_iter, "Weights", w)
    print(max_iter, "Average error", error/len(X_test))
    
    
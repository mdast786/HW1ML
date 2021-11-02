from pandas import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd


class VotedPerceptron:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.V = []
        self.C = []
        self.k = 0
    
    def fit(self, x, y):
        for i in range(len(y)):
            if y[i] == 0:
                y[i] = -1
        k = 0
        v = [np.ones_like(x)[0]]
        c = [0]
        for epoch in range(self.n_iter): # runs through the data n_iter times
            for i in range(len(x)):
                pred = 1 if np.dot(v[k], x[i]) > 0 else -1 # checks the sing of v*k
                if pred == y[i]: # checks if the prediction matches the real Y
                    c[k] += 1 # increments c
                else:
                    v.append(np.add(v[k], np.dot(y[i], x[i])))
                    c.append(1)
                    k += 1
        self.V = v
        self.C = c
        self.k = k
        return self.V, self.C
        
    def predict(self, X):
        preds = []
        for x in X:
            s = 0
            for w,c in zip(self.V,self.C):
                s = s + c*np.sign(np.dot(w,x))
            preds.append(np.sign(1 if s>= 0 else 0))
        return preds
    
    



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


weights = []

for max_iter in range(10):  
    v_perc = VotedPerceptron(n_iter=max_iter)
    w,c = v_perc.fit(X_train.values, y_train)
    weights.append((w,c))
    print("weight", w)
    
    y_pred = v_perc.predict(X_test.values)
    predict = v_perc.predict(X_test.values)
    error = np.sum(abs(predict - y_test))
    print(max_iter, "Average error", error/len(X_test))

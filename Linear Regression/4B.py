import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import linalg as LA
import matplotlib.pyplot as plt



# linear regression using "mini-batch" gradient descent
# function to compute hypothesis / predictions
def hypothesis(X, theta):
    return np.dot(X, theta)
  
# function to compute gradient of error function w.r.t. theta
def gradient(X, y, theta):
    h = hypothesis(X, theta)
    grad = np.dot(X.transpose(), (h - y))
    return grad
  
# function to compute the error for current values of theta
def cost(X, y, theta):
    h = hypothesis(X, theta)
    J = np.dot((h - y).transpose(), (h - y))
    J /= 2
    return sum(J)

# function to perform mini-batch gradient descent
def gradientDescent(X, y, learning_rate, tolerance):
    theta = np.zeros((X.shape[1], 1))
    
    prev_theta = theta.copy()
    error_list = []
    max_iters = 3
    
    #i =0
    #while i <10000:
    while True:
        for i in range(len(X)):
            x_i = X[i, :]
            y_i = y[i, :]
            prev_theta = theta.copy()
            theta = theta - learning_rate * gradient(X, y, theta)
            error_list.append(cost(X, y, theta))
            print("cost",cost(X, y, theta))
            print(LA.norm(prev_theta - theta))
                
        if LA.norm(prev_theta - theta) <= tolerance:
            break
  
    return theta, error_list


df_train = pd.read_csv('concrete/train.csv')
df_test = pd.read_csv('concrete/test.csv')

# rename known columns
columns = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'Output']

df_train.columns = columns
df_test.columns = columns



# organize data into input and output
X_train = df_train.drop(columns="Output").values
y_train = df_train['Output'].values.reshape(-1,1)
# X_test = df_test.drop(columns="y")
# y_test = df_test["y"]


############# r, tolerance ##########################
theta, error_list = gradientDescent(X_train, y_train, .001, .000001)
plt.plot(error_list)
#print("Bias = ", theta[0])
#print("Coefficients = ", theta[1:])

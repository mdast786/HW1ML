import numpy as np
import pandas as pd

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels, epochs):
        for i in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
        return self.weights
            
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

for epoch in range(10):
    perceptron = Perceptron(X_train.shape[1])
    weights = perceptron.train(X_train.values, y_train, epoch)
    
    print(epoch, "Weights", weights)
    error= 0
    for i in range(len(X_test)):
        predict = perceptron.predict(X_test.iloc[i,:])
        # print(predict, y_test[i])
        error += abs(predict - y_test[i])
    
    
    print(epoch, "Average error", error/len(X_test))
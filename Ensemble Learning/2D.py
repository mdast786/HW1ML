import pandas as pd
from sklearn.metrics import accuracy_score
import random
import numpy as np
from math import log,exp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import random

# Import label encoder
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('datasets/bank/train.csv').iloc[:500,:]
df_test = pd.read_csv('datasets/bank/test.csv').iloc[:500,:]


# rename known columns
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
df_train.columns = columns
df_test.columns = columns

cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
np.median(df_train[cols], axis=0)
df_train[cols] = df_train[cols] < np.median(df_train[cols], axis=0)
df_train[cols] = df_train[cols].astype(int)

df_train['y'] =  pd.Series(pd.factorize(df_train['y'])[0])
#df_train['y'] =  np.where(df_train['y'] == 0, -1, 1)

for col in columns:
    df_train[col]= label_encoder.fit_transform(df_train[col])
    df_test[col]= label_encoder.fit_transform(df_train[col])


num_features = 6
nums = random.sample(range(1, 17), num_features)


# organize data into input and output
X_train = df_train.drop(columns="y").iloc[:, nums]
y_train = pd.Series(pd.factorize(df_train['y'])[0])
X_test = df_test.drop(columns="y").iloc[:,nums]
y_test = df_test["y"]
y_test = pd.Series(pd.factorize(df_test['y'])[0])

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=500)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
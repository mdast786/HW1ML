import pandas as pd
from sklearn.metrics import accuracy_score
import random
import numpy as np
from math import log,exp


class Id3Classifier:
  def __init__(self, depth, split_criteria):
    self.depth = depth
    self.split_criteria = split_criteria
    
  def fit(self, input, output):
    data = input.copy()
    data[output.name] = output
    self.tree = self.decision_tree(data, data, input.columns, output.name)

  def predict(self, input):
    # convert input data into a dictionary of samples
    samples = input.to_dict(orient='records')
    predictions = []

    # make a prediction for every sample
    for sample in samples:
      predictions.append(self.make_prediction(sample, self.tree, 1.0))

    return predictions

  def information_gain_entropy(self, attribute_column):
    # find unique values and their frequency counts for the given attribute
    values, counts = np.unique(attribute_column, return_counts=True)

    # calculate entropy for each unique value
    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(-probability*np.log2(probability))

    # calculate sum of individual entropy values
    total_entropy = np.sum(entropy_list)

    return total_entropy

  def gini_index_entropy(self, attribute_column):
    # find unique values and their frequency counts for the given attribute
    values, counts = np.unique(attribute_column, return_counts=True)

    # calculate entropy for each unique value
    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(probability**2)

    # calculate sum of individual entropy values
    total_entropy = 1 - np.sum(entropy_list)

    return total_entropy

  def majority_error_entropy(self, attribute_column):
    # find unique values and their frequency counts for the given attribute
    values, counts = np.unique(attribute_column, return_counts=True)

    # calculate entropy for each unique value
    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(probability)

    # calculate sum of individual entropy values
    total_entropy = 1 - max(entropy_list)

    return total_entropy

  def information_gain(self, data, feature_attribute_name, target_attribute_name):
    # find total entropy of given subset
    total_entropy = self.information_gain_entropy(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    # calculate weighted entropy of subset
    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.information_gain_entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

    # calculate information gain
    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def gini_index(self, data, feature_attribute_name, target_attribute_name):
    # find total entropy of given subset
    total_entropy = self.gini_index_entropy(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    # calculate weighted entropy of subset
    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.gini_index_entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

    # calculate information gain
    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def majority_error(self, data, feature_attribute_name, target_attribute_name):
    # find total entropy of given subset
    total_entropy = self.majority_error_entropy(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    # calculate weighted entropy of subset
    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.majority_error_entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

    # calculate information gain
    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def decision_tree(self, data, original_data, feature_attribute_names, target_attribute_name, parent_node_class=None):
    # base cases:
    # if data is pure, return the majority class of subset
    unique_classes = np.unique(data[target_attribute_name])
    if len(unique_classes) == 1:
      return unique_classes[0]
    
    # if subset is empty, ie. no samples, return majority class of original data
    elif len(data) == 0:
      majority_class_index = np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])
      return np.unique(original_data[target_attribute_name])[majority_class_index]
    
    # if data set contains no features to train with, return parent node class
    elif len(feature_attribute_names) == 0:
      return parent_node_class
    # if none of the above are true, construct a branch:
    else:
      # determine parent node class of current branch
      majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
      parent_node_class = unique_classes[majority_class_index]

      # determine information gain values for each feature
      # choose feature which best splits the data, ie. highest value
      if self.split_criteria == "information-gain":
          sc_values = [self.information_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      elif self.split_criteria == "gini-index":
          sc_values = [self.gini_index(data, feature, target_attribute_name) for feature in feature_attribute_names]
      elif self.split_criteria == "majority-error":
          sc_values = [self.majority_error(data, feature, target_attribute_name) for feature in feature_attribute_names]
      
      best_feature_index = np.argmax(sc_values)
      best_feature = feature_attribute_names[best_feature_index]

      # create tree structure, empty at first
      tree = {best_feature: {}}

      # remove best feature from available features, it will become the parent node
      feature_attribute_names = [i for i in feature_attribute_names if i != best_feature]

      # create nodes under parent node
      parent_attribute_values = np.unique(data[best_feature])
    
      #print("pav",len(parent_attribute_values))
      height = self.depth
      for value in parent_attribute_values:
        sub_data = data.where(data[best_feature] == value).dropna()

        # call the algorithm recursively
        subtree = self.decision_tree(sub_data, original_data, feature_attribute_names, target_attribute_name, parent_node_class)

        # add subtree to original tree
        tree[best_feature][value] = subtree
        #print("depth", self.depth)
        
        height -= 1
        if height == 0:
            break
      return tree

  def make_prediction(self, sample, tree, default=1):
    # map sample data to tree
    for attribute in list(sample.keys()):
      # check if feature exists in tree
      if attribute in list(tree.keys()):
        try:
          result = tree[attribute][sample[attribute]]
        except:
          return default

        result = tree[attribute][sample[attribute]]

        # if more attributes exist within result, recursively find best result
        if isinstance(result, dict):
          return self.make_prediction(sample, result)
        else:
          return result
      

        
      
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('datasets/bank/train.csv').iloc[:,:]
df_test = pd.read_csv('datasets/bank/test.csv').iloc[:,:]


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



# organize data into input and output
X_train = df_train.drop(columns="y")
y_train = pd.Series(pd.factorize(df_train['y'])[0])
X_test = df_test.drop(columns="y")
y_test = df_test["y"]
y_test = pd.Series(pd.factorize(df_test['y'])[0])



example = df_train.copy()

#Initially assign same weights to each records in the dataset
df_train['probR1'] = 1/(df_train.shape[0])

alphas = []
for i in range(500):
    random.seed(10)
    example = df_train.sample(len(df_train), replace = True, weights = df_train['probR1'])
    
    
    X_train = example.drop(columns="y")
    y_train = pd.Series(pd.factorize(example['y'])[0])
    
    depth = 2
    model = Id3Classifier(depth, "information-gain")
    model.fit(X_train, y_train)
    # return accuracy score
    y_pred = model.predict(X_train)
    df_train['pred'+str(i)] = y_pred
    acc_score = accuracy_score(y_train, y_pred)
    print(acc_score)
    
    
    #misclassified = 0 if the label and prediction are same
    df_train.loc[df_train.y != df_train['pred'+str(i)], 'misclassified'] = 1
    df_train.loc[df_train.y == df_train['pred'+str(i)], 'misclassified'] = 0
    
    #error calculation
    e = sum(df_train['misclassified'] * df_train['probR1'])
    
    
    #calculation of alpha (performance)
    alpha = 0.5*log((1-e)/e)
    alphas.append(alpha)
    #update weight
    new_weight = df_train['probR1']*np.exp(-1*alpha*df_train['y']*df_train['pred'+str(i)])
    
    #normalized weight
    z = sum(new_weight)
    normalized_weight = new_weight/sum(new_weight)
    
    df_train['probR1'] = round(normalized_weight,4)
    
###################################################################################    
#final prediction
sum = 0
for i in range(5):
    sum += alphas[i] * df_train['pred'+str(i)]
sum

#sign of the final prediction
df_train['final_pred'] = np.where(sum<0, 0, 1)
y_pred = df_train['final_pred']
df_train['pred'+str(i)] = y_pred
acc_score = accuracy_score(y_train, y_pred)
print(acc_score)
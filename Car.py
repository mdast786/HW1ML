import pandas as pd
from sklearn.metrics import accuracy_score


import numpy as np

class Id3Classifier:
  def __init__(self, depth, split_criteria):
    self.depth = depth
    self.split_criteria = split_criteria
    
  def fit(self, input, output):
    data = input.copy()
    data[output.name] = output
    self.tree = self.decision_tree(data, data, input.columns, output.name)

  def predict(self, input):
    
    samples = input.to_dict(orient='records')
    predictions = []

    
    for sample in samples:
      predictions.append(self.make_prediction(sample, self.tree, 1.0))

    return predictions

  def information_gain_entropy(self, attribute_column):
    
    values, counts = np.unique(attribute_column, return_counts=True)

   
    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(-probability*np.log2(probability))

    
    total_entropy = np.sum(entropy_list)

    return total_entropy

  def gini_index_entropy(self, attribute_column):
    
    values, counts = np.unique(attribute_column, return_counts=True)


    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(probability**2)

   
    total_entropy = 1 - np.sum(entropy_list)

    return total_entropy

  def majority_error_entropy(self, attribute_column):
    
    values, counts = np.unique(attribute_column, return_counts=True)


    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(probability)

 
    total_entropy = 1 - max(entropy_list)

    return total_entropy

  def information_gain(self, data, feature_attribute_name, target_attribute_name):
  
    total_entropy = self.information_gain_entropy(data[target_attribute_name])


    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

  
    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.information_gain_entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

  
    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def gini_index(self, data, feature_attribute_name, target_attribute_name):

    total_entropy = self.gini_index_entropy(data[target_attribute_name])

    
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)


    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.gini_index_entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

 
    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def majority_error(self, data, feature_attribute_name, target_attribute_name):

    total_entropy = self.majority_error_entropy(data[target_attribute_name])

   
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

  
    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.majority_error_entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)


    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def decision_tree(self, data, original_data, feature_attribute_names, target_attribute_name, parent_node_class=None):
  
    unique_classes = np.unique(data[target_attribute_name])
    if len(unique_classes) == 1:
      return unique_classes[0]
 
    elif len(data) == 0:
      majority_class_index = np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])
      return np.unique(original_data[target_attribute_name])[majority_class_index]
    
   
    elif len(feature_attribute_names) == 0:
      return parent_node_class
   
    else:
    
      majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
      parent_node_class = unique_classes[majority_class_index]

   
      if self.split_criteria == "information-gain":
          sc_values = [self.information_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      elif self.split_criteria == "gini-index":
          sc_values = [self.gini_index(data, feature, target_attribute_name) for feature in feature_attribute_names]
      elif self.split_criteria == "majority-error":
          sc_values = [self.majority_error(data, feature, target_attribute_name) for feature in feature_attribute_names]
      
      best_feature_index = np.argmax(sc_values)
      best_feature = feature_attribute_names[best_feature_index]


      tree = {best_feature: {}}

     
      feature_attribute_names = [i for i in feature_attribute_names if i != best_feature]

  
      parent_attribute_values = np.unique(data[best_feature])
    
     
      height = self.depth
      for value in parent_attribute_values:
        sub_data = data.where(data[best_feature] == value).dropna()

    
        subtree = self.decision_tree(sub_data, original_data, feature_attribute_names, target_attribute_name, parent_node_class)

       
        tree[best_feature][value] = subtree
      
        
        height -= 1
        if height == 0:
            break
      return tree

  def make_prediction(self, sample, tree, default=1):
 
    for attribute in list(sample.keys()):
   
      if attribute in list(tree.keys()):
        try:
          result = tree[attribute][sample[attribute]]
        except:
          return default

        result = tree[attribute][sample[attribute]]

       
        if isinstance(result, dict):
          return self.make_prediction(sample, result)
        else:
          return result
      

        
      
import pandas as pd
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('datasets/car/train.csv')
df_test = pd.read_csv('datasets/car/test.csv')



columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
df_train.columns = columns
df_test.columns = columns



X_train = df_train.drop(columns="label")
y_train = df_train["label"]
X_test = df_test.drop(columns="label")
y_test = df_test["label"]



depth = int(input("Tree depth? "))
for split_criteria in ["information-gain", "gini-index", "majority-error"]:
    print("Using ", split_criteria, "splitting criteria:")
    model = Id3Classifier(depth, split_criteria)

    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    print(acc_score)

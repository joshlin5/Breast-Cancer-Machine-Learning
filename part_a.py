#!/usr/bin/env python
# coding: utf-8

# # CPSC 4300/6300-001 Applied Data Science (Fall 2020)
# 
# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[1]:


NAME = "Joshua Lin"
COLLABORATORS = ""


# # CPSC4300/6300-001 Problem Set #4
# 
# In this data set, you will how to apply several classification methods on two problems: breast cancen prediction and sentiment alanalysis.

# # Part A. Breast Cancer Prediction (I)
# 
# Data science has a wide range of applications in the healthcare industry. In this assignmnet, you apply machine learning algorithms to the problem of breast cancer dignose. You can find a systematic review of the problem in the following publications:
# 
# 1. Salod, Zakia, and Yashik Singh. “A five-year (2015 to 2019) analysis of studies focused on breast cancer prediction using machine learning: A systematic review and bibliometric analysis.” Journal of public health research vol. 9,1 1792. 26 Jun. 2020, doi:10.4081/jphr.2020.1772
# 2. Gardezi, Syed Jamal Safdar et al. “Breast Cancer Detection and Diagnosis Using Mammographic Data: Systematic Review.” Journal of medical Internet research vol. 21,7 e14464. 26 Jul. 2019, doi:10.2196/14464
# 
# In this problem, we use the Wisconsin Breast Cancer Dataset (WDBC) through the sklearn.datasets module (see: https://scikit-learn.org/stable/datasets/index.html#datasets). The original dataset is available at https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29.
# 

# # 0. Set Up Basic Environment

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
sns.set_style("white")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1. Get Data

# In[3]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer(as_frame=True)
features = cancer.data
target = cancer.target.astype('category')


# In[4]:


df = pd.concat([features, target], axis=1)
df.iloc[[10, 50, 85]]


# In[5]:


df.info()


# # 2. Basic Decision Tree Classification

# In[6]:


from collections import defaultdict
train_accuracy = defaultdict(list)
test_accuracy = defaultdict(list)


# ## Question 2.1 Train Decision Tree Models
# 
# Complete the following code to build a decision tree model and assess its performance. Assign the model to variable `clf`. (5 points)

# In[7]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from collections import defaultdict


perf_measures = defaultdict(dict)
experiment_id = 0

X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.2, random_state=1)

# YOUR CODE HERE
#raise NotImplementedError()
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

perf_measures[experiment_id]['model'] = 'Decision_Tree'
perf_measures[experiment_id]['train_accuracy'] = clf.score(X_train, y_train)
perf_measures[experiment_id]['test_accuracy'] = clf.score(X_test, y_test)
perf_measures[experiment_id]['f1_score'] = f1_score(y_test, y_test_pred)
perf_measures[experiment_id]['precision'] = precision_score(y_test, y_test_pred)
perf_measures[experiment_id]['recall'] = recall_score(y_test, y_test_pred)

df_accuracy = pd.DataFrame(perf_measures).T
df_accuracy = df_accuracy[['model', 'precision', 'recall', 'f1_score', 'test_accuracy', 'train_accuracy']]
df_accuracy


# In[8]:


assert isinstance(clf, DecisionTreeClassifier)


# In[9]:


assert  clf.get_depth() > 4 and clf.get_n_leaves() > 10


# In[10]:


assert  clf.criterion in ['gini', 'entropy']


# In[11]:


assert df_accuracy['train_accuracy'].max() > 0.9 and df_accuracy['test_accuracy'].max() > 0.8


# In[12]:


assert df_accuracy['f1_score'].max() > 0.8


# ## Question 2.2 Visualize Decision Trees
# 
# Complete the following code to save the decision tree referred by `clf` to a dot file named `tree01.dot` and then display the data within the Notebook. (5 points)

# In[13]:


from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

tree_file = 'tree01.dot'
# YOUR CODE HERE
#raise NotImplementedError()
export_graphviz(clf, out_file = tree_file, class_names = ["maligant", "benign"],
               feature_names=cancer.feature_names,
               impurity=False, filled=True)
graph = pydotplus.graph_from_dot_file(tree_file)
Image(graph.create_png())


# In[14]:


import os
assert os.path.exists(tree_file)


# In[15]:


assert isinstance(graph, pydotplus.graphviz.Dot)


# In[16]:


assert len(graph.get_edges()) >= 10


# ## Question 2.3 Decision Tree Variance
# 
# From the lecture, you have learned that decision trees are typically not robust. In other words, a small change in the data can cause a large change in the final estimated tree. 
# 
# __Question 2.3(a)__ Complete the following code to perform an experiment on this issue. Note that the missing parts of the code are: split the data, train the model, and record the test accuracy. (5 points)

# In[17]:


from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

perf_measures = defaultdict(dict)

clf = DecisionTreeClassifier()

experiment_id = 0
for test_size in [0.2, 0.25, 0.3]:
    for i, random_state in enumerate(np.random.randint(1, size=30)):
        experiment_id = experiment_id + 1
# YOUR CODE HERE
#raise NotImplementedError()
        X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=test_size, random_state=random_state)
        _ = clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        perf_measures[experiment_id]['test_size'] = float(test_size)
        perf_measures[experiment_id]['run#'] = i
        perf_measures[experiment_id]['train_accuracy'] = clf.score(X_train, y_train)
        perf_measures[experiment_id]['test_accuracy'] = clf.score(X_test, y_test)
        perf_measures[experiment_id]['f1_score'] = f1_score(y_test, y_test_pred)
        perf_measures[experiment_id]['precision'] = precision_score(y_test, y_test_pred)
        perf_measures[experiment_id]['recall'] = recall_score(y_test, y_test_pred)
        
df_accuracy = pd.DataFrame(perf_measures).T
df_accuracy = df_accuracy[['test_size', 'run#', 'precision', 'recall', 'f1_score', 'test_accuracy', 'train_accuracy']]
df_accuracy.head(5)


# In[18]:


assert 'test_accuracy' in df_accuracy.columns and 'test_accuracy' in df_accuracy.columns


# In[19]:


assert 1.0 > df_accuracy['test_accuracy'].mean() > 0.85


# In[20]:


assert 1.0 > df_accuracy['f1_score'].mean() > 0.85


# In[21]:


fig, ax = plt.subplots(1, 2, figsize=(16,6))

_ = sns.lineplot(x=df_accuracy['run#'], y=df_accuracy['test_accuracy'],  label='test_accuracy', ax=ax[0])
_ = sns.lineplot(x=df_accuracy['run#'], y=df_accuracy['train_accuracy'], label='train_accuracy', ax=ax[0])
_ = sns.scatterplot(x=df_accuracy['run#'], y=df_accuracy['train_accuracy'], ax=ax[0])
_ = sns.scatterplot(x=df_accuracy['run#'], y=df_accuracy['test_accuracy'], hue=df_accuracy['test_size'], 
                    palette="deep", legend='auto', ax=ax[0])
_ = ax[0].legend(loc='lower center')
_ = ax[0].set_ylim(0.85, 1.01)
_ = ax[0].set_title('Decision Tree Accuracy Variance with Datasets')

_ = sns.boxplot(x='test_size', y='test_accuracy', data=df_accuracy, ax=ax[1])
_ = ax[1].set_ylim(0.85, 1.01)
_ = ax[1].set_title('Effects of Train-Test Split on Decision Tree Accuracy')
plt.show()


# __Question 2.3(b)__ Which of the following statements is most likely __incorrect__? (3 points)
# 
# ```
# A. Without pruning, a decision tree classifier is susceptible to overfitting.
# B. Increasing the training set generally leads to higher prediction accuracy.
# C. Increasing the testing set can reduce the variance of accuracy estimation for a predictive model.
# D. Increasing the testing set will increase the prediction accuracy of a decision tree classifier.
# ```

# In[22]:


# YOUR CODE HERE
#raise NotImplementedError()
answer = 'D'


# In[23]:


assert answer in ['A', 'B', 'C', 'D']


# In[24]:


# Here is a hidden test to chech the correct answer


# ## Question 2.4 Decision Tree Prunning
# 
# The `DecisionTreeClassifier` provides parameters such as `min_samples_leaf`, `max_depth`, and `ccp_alpha` to prevent a tree from overfiting. Cost complexity pruning provides another option to control the size of a tree. See https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py.
# 
# Pick any tree prunning strategy and then perform experiments to show how the training and testing accuracy will change with the parameter. 
# 
# This question is open-ended. Any __working__ implementation with a reasonable summary of observations will receive full points.
# 
# __Question 2.4(a)__ Write some code to perform the experiments. Similar to Question 2.3, your code should both run the experiments and create one or several plots to summarize the results of your experiments. (10 points: 5 for experiments; 5 for plots).
# 
# Hint: you may reuse the code in Question 2.3. One possible venue is replacing the `test_size` variable with a prunning parameter like `min_samples_leaf`, `max_depth`, or `ccp_alpha`.
# 
# 

# In[25]:


# write your code for conducting the experiments

# YOUR CODE HERE
#raise NotImplementedError()
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

perf_measures = defaultdict(dict)


test_size = 0.25

experiment_id = 0
for i, random_state in enumerate(np.random.randint(1, size=30)):
    for max_depth in [3,4,5,8]:
        clf = DecisionTreeClassifier(max_depth=max_depth)
        experiment_id = experiment_id + 1
# YOUR CODE HERE
#raise NotImplementedError()
        X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=test_size, random_state=random_state)
        _ = clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        perf_measures[experiment_id]['test_size'] = float(test_size)
        perf_measures[experiment_id]['run#'] = i
        perf_measures[experiment_id]['train_accuracy'] = clf.score(X_train, y_train)
        perf_measures[experiment_id]['test_accuracy'] = clf.score(X_test, y_test)
        perf_measures[experiment_id]['f1_score'] = f1_score(y_test, y_test_pred)
        perf_measures[experiment_id]['precision'] = precision_score(y_test, y_test_pred)
        perf_measures[experiment_id]['recall'] = recall_score(y_test, y_test_pred)
        perf_measures[experiment_id]['max_depth'] = int(max_depth)
        
df_accuracy = pd.DataFrame(perf_measures).T
df_accuracy = df_accuracy[['test_size', 'run#', 'precision', 'recall', 'f1_score', 'test_accuracy', 'train_accuracy', 'max_depth']]
df_accuracy.head(5)


# In[26]:


# write your code for create plots


# YOUR CODE HERE
#raise NotImplementedError()
fig, ax = plt.subplots(1, 2, figsize=(16,6))

_ = sns.lineplot(x=df_accuracy['run#'], y=df_accuracy['test_accuracy'],  label='test_accuracy', ax=ax[0])
_ = sns.lineplot(x=df_accuracy['run#'], y=df_accuracy['train_accuracy'], label='train_accuracy', ax=ax[0])
_ = sns.scatterplot(x=df_accuracy['run#'], y=df_accuracy['train_accuracy'], ax=ax[0])
_ = sns.scatterplot(x=df_accuracy['run#'], y=df_accuracy['test_accuracy'], hue=df_accuracy['max_depth'], 
                    palette="deep", legend='auto', ax=ax[0])
_ = ax[0].legend(loc='lower center')
_ = ax[0].set_ylim(0.85, 1.01)
_ = ax[0].set_title('Decision Tree Accuracy Variance with Datasets')

_ = sns.boxplot(x='max_depth', y='test_accuracy', data=df_accuracy, ax=ax[1])
_ = ax[1].set_ylim(0.85, 1.01)
_ = ax[1].set_title('Effects of Train-Test Split on Decision Tree Accuracy')
plt.show()


# __Question 2.4(b)__ Briefly descrine some observation(s) with regard to tree prunning you may draw from your experiments. (5 points)

# ## Briefly describe your observation here.

# In[27]:


#After prunning, the variance of accuracy estimation for a predictive model was reduced.
#The decision tree classifier also seemed to fit better.
#However, increasing the max_depth did not increase the decision tree accuray. 


# # 3. Model Evaluation Utility
# 
# To simplify subsequent studies, we abstract the process of evaluate a model into a function as follows:

# In[28]:


from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from time import time

def evaluate_model(clf, label, perf_data = defaultdict(dict), nrepeats=10, test_size=0.25):
    '''evaluate_model assesses the model performance with several splits from a feature matrix and target vector
    
     Parameters
     ---------
        clf: ClassifierMixin
            a classifier to be assessed
        label: string
            the name of the classifier
        perf_data: defaultdict(dict), optional
            a dict that stores the performance data
        test_size: float, optional
            a real number in the range (0, 1.0) that controls the size of train set and test set
        nrepeats: int, optional
            the number of runs for each model
     
     Returns
     ---------
         perf_data: defaultdict(dict)
    '''
    experiment_id = len(perf_data)
    for i in range(nrepeats):
        X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target,
                                                            test_size=test_size, random_state=i)
        time_start_train = time()
        _ = clf.fit(X_train, y_train)
        time_finish_train = time()
        
        y_test_pred = clf.predict(X_test)
        time_finish_pred = time()
        
        perf_data[experiment_id]['model'] = label
        perf_data[experiment_id]['run#'] = i
        perf_data[experiment_id]['train_accuracy'] = clf.score(X_train, y_train)
        perf_data[experiment_id]['test_accuracy'] = clf.score(X_test, y_test)
        perf_data[experiment_id]['f1_score'] = f1_score(y_test, y_test_pred)
        perf_data[experiment_id]['precision'] = precision_score(y_test, y_test_pred)
        perf_data[experiment_id]['recall'] = recall_score(y_test, y_test_pred)
        perf_data[experiment_id]['train_time'] = time_finish_train - time_start_train
        perf_data[experiment_id]['inference_time'] = time_finish_pred - time_finish_train
        
        experiment_id = experiment_id + 1
    return perf_data


# In[29]:


def plot_train_test_accuracy(df_perf, plot_train_accuracy=False):
    '''plot_train_test_accuracy plots model accuracy data provided in df_perf
    
     Parameters
     ---------
        df_perf: DataFrame
            a DataFrame that stores the model performance data
        plot_train_accuracy: bool, optional
            a flag to control whether to plot the train accuracy
    '''
    fig, ax = plt.subplots(2, 2, figsize=(20,16))
    for model in df_perf.model.unique():
        x = df_perf[df_perf['model'] == model]['run#']
        y1 = df_perf[df_perf['model'] == model]['test_accuracy']
        y2 = df_perf[df_perf['model'] == model]['train_accuracy']
        ax[0, 0].plot(x, y1, label=model+'::test accuracy')
        if plot_train_accuracy:
            ax[0, 0].plot(x, y2, label=model+'::train accuracy')

    _ = ax[0, 0].legend()
    _ = ax[0, 0].set_xlabel('Run #')
    _ = ax[0, 0].set_ylim(0.85, 1.01)
    _ = ax[0, 0].set_title('Variance of Test and Train Accuarcy Over Multiple Runs')
        
    _ = sns.boxplot(x='model', y='test_accuracy', data=df_perf, ax=ax[0, 1])
    _ = ax[0, 1].set_ylim(0.85, 1.01)
    _ = ax[0, 1].set_title('Model Accuracy on Test Set')
    
    _ = sns.boxplot(x='model', y='train_time', data=df_perf, ax=ax[1, 0])
    _ = ax[1, 0].set_title('Training Time')
    _ = ax[1, 0].set_ylabel('Time (seconds)')
    
    _ = sns.boxplot(x='model', y='inference_time', data=df_perf, ax=ax[1, 1])
    _ = ax[1, 1].set_title('Inference Time')
    _ = ax[1, 1].set_ylabel('Time (seconds)')


# In[30]:


from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# define several models
models = {
    'GaussianNB': make_pipeline(StandardScaler(), GaussianNB()),
    'MLP': make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=8, max_iter=2000))
}

# define model_perf_data
model_perf_data = defaultdict(dict)

# evaulate the models
for label, clf in models.items():
    _= evaluate_model(clf, label, model_perf_data)

# plot the results
df_perf = pd.DataFrame(model_perf_data).T
# plot_train_test_accuracy(df_perf, plot_train_accuracy=True)


# In[31]:


plot_train_test_accuracy(df_perf, plot_train_accuracy=True)


# ## Question 3.1  Model Comparison
# 
# __Question 3.1(a)__ Based on the above results, which of the following statements is mostly incorrect? Assign your best answer to the string variable `answer`.
# 
# ```
# A. For the breast cancer dataset, the training accuracy and test accuracy of the GaussianNB model are statistically identical.
# 
# B. For the MLPClassifier (i.e., Multi-layer Perceptron classifier), the training accuracy and test accuracy of the GaussianNB model are statistically identical.
# 
# C. With the same dataset, the GaussianNB classifier can be trained much faster than the MLPClassifier classifier. Therefore, to provide a patient quick results, a clinic should choose GaussianNB classifier over MLPClassifier.
# 
# D. Both GaussianNB and MLPClassifier require standardization of the dataset to achieve higher accuracy. 
# ```

# In[32]:


# YOUR CODE HERE
#raise NotImplementedError()
answer = 'A'


# In[33]:


assert answer in ['A', 'B', 'C', 'D']


# In[34]:


# Here is a hidden test to chech the correct answer


# # 4. KNN Classifier
# 
# In this question, we look at some issues of KNN classification. 

# ## Question 4.1  Standardization of Dataset
# 
# Decision tree classifiers do not require standardization of dataset. But for K-Nearest Neighbors, SVM, Multi-layer Perceptron, and many other machine learning algorithms, standardization of dataset is required in order to achieve desirable performance.
# 
# __Question 4.1__ Complete the following code to compare the performance of k-NN with Standardization and k-NN without Standardization for the Breast Cancer dataset.

# In[35]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# YOUR CODE HERE
#raise NotImplementedError()
models = {
    'KNN-Standard': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'KNN-Not-Standard': make_pipeline(KNeighborsClassifier())
}

# define model_perf_data
model_perf_data = defaultdict(dict)

# evaulate the models
for label, clf in models.items():
    _= evaluate_model(clf, label, model_perf_data)

# plot the results
df_perf = pd.DataFrame(model_perf_data).T
plot_train_test_accuracy(df_perf, plot_train_accuracy=True)


# ## Question 4.2 Effects of Model Hyperparameters
# 
# __Question 4.2__ Complete the following code to investigate the effects of number of neighbors (i.e., `n_neighbors`) of k-NN classifier for the Breast Cancer dataset.

# In[36]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# YOUR CODE HERE
#raise NotImplementedError()
models = {
    'KNN-1': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
    'KNN-4': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=4)),
    'KNN-8': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=8)),
    'KNN-12': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=12)),
    'KNN-16': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=16))
}

# define model_perf_data
model_perf_data = defaultdict(dict)

# evaulate the models
for label, clf in models.items():
    _= evaluate_model(clf, label, model_perf_data, nrepeats=30)

# plot the results
df_perf = pd.DataFrame(model_perf_data).T


# In[37]:


fig, ax = plt.subplots(1, 2, figsize=(20,8))
_ = sns.boxplot(x='model', y='test_accuracy', data=df_perf, ax=ax[0])
_ = sns.boxplot(x='model', y='train_time', data=df_perf, ax=ax[1])


# # 5. Random Forest Classifier

# ## 5.1 Effects of Number of Estimators

# __Question 5.1__ Complete the following code to investigate the effects of n_estimators on the performance of Random Forest Classifier using the WBCD data.

# In[38]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# YOUR CODE HERE
#raise NotImplementedError()
models = {
    'RFC-10': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 10)),
    'RFC-50': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 50)),
    'RFC-100': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 100)),
    'RFC-200': make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 200))
}

# define model_perf_data
model_perf_data = defaultdict(dict)

# evaulate the models
for label, clf in models.items():
    _= evaluate_model(clf, label, model_perf_data, nrepeats=30)

# plot the results
df_perf = pd.DataFrame(model_perf_data).T


# In[39]:


fig, ax = plt.subplots(1, 2, figsize=(20,8))
_ = sns.boxplot(x='model', y='test_accuracy', data=df_perf, ax=ax[0])
_ = sns.boxplot(x='model', y='train_time', data=df_perf, ax=ax[1])


# ## 5.2 Feature Importance
# 
# Random Forest classifiers outputs the impurity-based feature importances as a by-product. The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. The higher the reduction, the more important the feature. However, impurity-based feature importances can be misleading and thus requires further inspection.
# 
# Permutation feature importance is a model inspection technique which uses the decrease in a model score when a single feature value is randomly shuffled. If a features's permutation feature importance score is close to zero, it indicates such feature is not important to the model (see: https://scikit-learn.org/stable/modules/permutation_importance.html#permutation-importance).

# __Question 5.2__ Your experiments in 5.1 shows Random Forest classifiers achives higher prediction accuracy. However, the Permutation feature importance indicates that none of the features are important. Briefly explain what might cause this contradiction.
# 
# Hint: you may read the Scikit Learn example at https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py.

# ## Write your explaination here

# In[40]:


# The problem is that the dataset contains multicollinear features.
#You only need to analyze one feature from each cluster of features that are highly related.
#If you analyze many features that are highly related to each other, you will obviously get a higher accuracy.


# In[41]:


from sklearn.inspection import permutation_importance

# Estimate a RandomForestClassifier model
X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target, test_size=test_size, random_state=4300)
clf = RandomForestClassifier(n_estimators=200)
_ = clf.fit(X_train, y_train)

# Compute permutation_importance
permutated_importance_result = permutation_importance(clf, X_train, y_train, n_repeats=10)

# Arrange the data for plotting
feature_importance_sorted_idx = np.flip(np.argsort(clf.feature_importances_))
s_feature_importances = pd.Series(clf.feature_importances_, index=cancer.feature_names).sort_values(ascending=False)

df_permutation_importances = pd.DataFrame(permutated_importance_result.importances[feature_importance_sorted_idx].T, 
                                     columns=cancer.feature_names[feature_importance_sorted_idx])

# Plot feature_importances and permutation_importance
fig, ax = plt.subplots(1, 2, figsize=(28, 20))
_ = sns.barplot(x=s_feature_importances.values, y=s_feature_importances.index, ax=ax[0])
_ = ax[0].set_title('Feature Importance')
_ = sns.boxplot(data=df_permutation_importances, orient='h', ax=ax[1])


# ## __End of Part A__

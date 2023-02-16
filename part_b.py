#!/usr/bin/env python
# coding: utf-8

# # CPSC 4300/6300-001 Applied Data Science (Fall 2020)
# 
# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# In[ ]:


NAME = "Joshua Lin"
COLLABORATORS = ""


# # CPSC4300/6300-001 Problem Set #4

# # Part B. Classification Models Performance Evaluation
# 
# This part continues Part A. We make it a separate notebook so that we can keep the code clean.

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
sns.set_style("white")


# # 1. Classification Models Comparison
# 
# During this semester, you have learned multiple classification methods. One question you may wonder is how each method performs for a given data set. With the work you have just compeleted in part A, this task looks like a piece of cake. Now, using the framework we have used in this assignment, conduct some experiments to compare the performance of the following classification models:
# 
# 1. RandomForestClassifier
# 2. SVC
# 3. GaussianNB
# 4. KNeighborsClassifier
# 5. LogisticRegression
# 6. DecisionTreeClassifier
# 7. MLPClassifier
# 8. AdaBoostClassifier

# __Question 1.1__ Which one of the following statements is most accurate? (3 Points)
# 
# ```
# A. Given a dataset, the train and test set splits of the dataset will not affect the performance of the classification model.
# B. When compare the performance of several models, a best practice is to run each model multiple times with different train and test splits and to report both the means and variances of all models for a fair comparison.
# C. When solve a classification problem, more complex models like AdaBoost or Random Forest always perform better than a simplier model like Logistc Regression.
# D. A classification model that achives 100% test accuracy at one run is the best model for the problem under standy.
# ```

# In[2]:


# YOUR CODE HERE
#raise NotImplementedError()
answer = 'A'


# __Question 1.2__ Write some Python code to study the performance of the above eight classification models on the Wisconsin Breast Cancer dataset. Your code should complete the following objectives: (30 Points) 
# 
# 1. Run each model for 30 times with different train-test splits 
# 2. Collect the values of 'test_accuracy', 'f1_score', 'precision', 'recall', and 'train_time' for each model run.
# 3. Save the results of all runs to a DataFrame variable named `df_perf`.
# 
# 
# Hints:
# + You can reuse the framework and most code in Part A.
# + To simplify the experiments, you can standarize the train and test set right after the train-test data set split.
# + A proper procedure of the standarization is estimating a StandardScaler using train set and then transforming both the train set and test set using the estimated scaler.
# + To avoid your experiments from being interrupted by a single error, you may validate each model first before putting all the models in a single cell.

# In[83]:


# YOUR CODE HERE
#raise NotImplementedError()
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

cancer = load_breast_cancer(as_frame=True)
features = cancer.data
target = cancer.target.astype('category')


def evaluate_model(clf, label, perf_data = defaultdict(dict), nrepeats=30, test_size=0.25):
    experiment_id = len(perf_data)
    test_size  = 0.18
    for i in range(nrepeats):
        test_size = test_size + 0.02
        X_train, X_test, y_train, y_test = train_test_split(features, target, stratify=target,
                                                                test_size=test_size, random_state=i)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform (X_test)
            
        time_start_train = time()
        _ = clf.fit(X_train, y_train)
        time_finish_train = time()

        y_test_pred = clf.predict(X_test)
        time_finish_pred = time()

        perf_data[experiment_id]['model'] = label
        perf_data[experiment_id]['run#'] = i
        perf_data[experiment_id]['test_accuracy'] = clf.score(X_test, y_test)
        perf_data[experiment_id]['f1_score'] = f1_score(y_test, y_test_pred)
        perf_data[experiment_id]['precision'] = precision_score(y_test, y_test_pred)
        perf_data[experiment_id]['recall'] = recall_score(y_test, y_test_pred)
        perf_data[experiment_id]['train_time'] = time_finish_train - time_start_train
        perf_data[experiment_id]['test_size'] = float(test_size)

        experiment_id = experiment_id + 1
    return perf_data

models = {
    'ABC': make_pipeline(AdaBoostClassifier()),
    'DTC': make_pipeline(DecisionTreeClassifier()),
    'LR': make_pipeline(LogisticRegression()),
    'KNC': make_pipeline(KNeighborsClassifier()),
    'GNB': make_pipeline(GaussianNB()),
    'SVC': make_pipeline(SVC()),
    'RFC': make_pipeline(RandomForestClassifier()),
    'MLPC': make_pipeline(MLPClassifier(hidden_layer_sizes=8, max_iter=2000))
}

# define model_perf_data
model_perf_data = defaultdict(dict)

# evaulate the models
for label, clf in models.items():
    _= evaluate_model(clf, label, model_perf_data)
    
df_perf = pd.DataFrame(model_perf_data).T


# In[258]:


df_perf


# __Question 1.3(a)__ Write some code to print a table of the average values of the performance metrics of your experiments. An example table is shown as below. 

# In[84]:


# YOUR CODE HERE
#raise NotImplementedError()
df_perf.mean(axis = 0)


# __Question 1.3(b)__ Write some code to print a table of the maximum values of the performance metrics of your experiments. An example table is shown as below. 

# In[85]:


# YOUR CODE HERE
#raise NotImplementedError()
df_perf.max(axis = 0)


# __Question 1.4(a)__ Write some code to create a plot to summarize the test accuracy of the classification models you have studied.

# In[86]:


# YOUR CODE HERE
#raise NotImplementedError()
df_perf.test_accuracy.describe()


# __Question 1.4(b)__ A performance model can be measured by various metrics. We can plot all these metrics side-by-side to get an overall picture of the performance of a given model. In the cell after this question, you are provided with a sample code to create a bar plots for comparing the training time and inference time of eight classifcation models. 
# 
# Now, write some code to create the following bar graph to compare the `test_accuracy`, `precision`, `recall`, `f1_score` all the classification models.
# 
# !['metrics'](https://www.palmetto.clemson.edu/dsci/figures/ps04b_metrics.png)

# In[87]:


# YOUR CODE HERE
#raise NotImplementedError()
df_perf['precision'] = df_perf['precision'].astype(float)
df_perf['test_accuracy'] = df_perf['test_accuracy'].astype(float)
df_perf['f1_score'] = df_perf['f1_score'].astype(float)
df_perf['recall'] = df_perf['recall'].astype(float)
perf_columns = ['test_accuracy','f1_score','precision','recall']
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.set_style("ticks")
df = df_perf.groupby('model')[perf_columns].agg('mean').sort_values(by='test_accuracy', ascending=True)
df = df.reset_index()

metrics = ['test_accuracy', 'precision', 'recall', 'f1_score']
width = 1.0 / len(metrics) * 0.80
for i, col in enumerate(metrics):
    ax.bar(df.index + i * width, df[col], width)

ax.set_xticks(df.index + width)
ax.set_yscale('log')
_ = ax.set_xticklabels(df['model'])
ax.legend(metrics, loc='best')


# ### A sample code to plot a bar graph with two or more variables
# 
# ```
# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# sns.set_style("ticks")
# df = df_perf.groupby('model')[perf_columns].agg('mean').sort_values(by='test_accuracy', ascending=True)
# df = df.reset_index()
# 
# metrics = ['train_time', 'inference_time']
# width = 1.0 / len(metrics) * 0.80
# for i, col in enumerate(metrics):
#     ax.bar(df.index + i * width, df[col], width)
# 
# ax.set_xticks(df.index + width)
# ax.set_yscale('log')
# _ = ax.set_xticklabels(df['model'])
# ax.legend(metrics, loc='best')
# 
# fig.tight_layout()
# fig.savefig('ps04b_timing.png')
# ```
# 
# !['metrics'](https://www.palmetto.clemson.edu/dsci/figures/ps04b_timing.png)

# # 2. Confusion Matrix
# 
# In classification, a confusion matrix, also known as an error matrix, is a specific table layout that allows visualization of the performance of an algorithm. 
# 
# The following cell trains a LogisticRegression model and constructs a confusion matrix from the test samples.

# In[241]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

cancer = load_breast_cancer(as_frame=True)
features = cancer.data
target = cancer.target.astype('category')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.25, random_state=8)
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

clf = LogisticRegression(fit_intercept=True, max_iter=5000)
_ = clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)

CM = confusion_matrix(y_test, y_test_pred, labels=target.cat.categories)
print(CM)


# __question 2.1__ Plot the confusion matrix of the predictions on the test set using the above trained model.

# In[242]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.metrics import plot_confusion_matrix
#import matplotlib.pyplot as plt
plot_confusion_matrix(clf, X_test, y_test_pred, display_labels=list(cancer.target_names))
#plt.show()


# In[89]:


## Show the total counts of each class
c = cancer.target.value_counts()
#sns.barplot(x=cancer.target_names, y=c)
ax = sns.barplot(x=c.index, y=c)
ax.set_xticklabels(cancer.target_names)
ax.set_ylabel('Count')
plt.show()


# 
# In sklearn.metrics.confusion_matrix $C$, $C_{i,j}$ is equal to the number of observations known to be in group $i$ and predicted to be in group $j$.
# 
# The terms like 'true negative' and 'true positive' are context-specific. In the breast cancer example, __malignant__ (group __0__) means __positive__ and __benign__ (group __1__) means __negative__.
# 
# For better clarification, we use the following plots to show the connections between this terms for the breast cancer prediction problem.
# 
# __Question 2.2__ Write some code to extract the true negative, false positive, false negative, and true positive in the test set predictions and save the results to `tn`, `fp`, `fn`, and `tp`.

# In[93]:


CM = confusion_matrix(y_test, y_test_pred, labels=target.cat.categories)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
sns.heatmap(CM, annot=True, cbar=False, ax=ax[0])
ax[0].set_xlabel('Predicted Label')
ax[0].set_ylabel('True Label')

for idx in [1, 2]:
    if idx == 1:
        group_names = ['True Malignant', 'False Benign', 'False Malignant', 'True Benign']
        class_labels = ['Malignant', 'Benign']
    elif idx==2:
        group_names = ['True Positive', 'False Negative', 'False Positive', 'True Negative'] 
        class_labels = ['Positive', 'Negative']
        
    group_counts = [f'{value:0.0f}' for value in CM.flatten()]
    labels = [f'{v1}\n{v2}' for v1,v2 in zip(group_names, group_counts)]
    labels = np.array(labels).reshape(2, 2)
    sns.heatmap(CM, annot=labels, fmt='', cmap='YlOrBr', cbar=False, ax=ax[idx])
    ax[idx].set_xlabel('Predicted Label')
    ax[idx].set_ylabel('True Label')
    ax[idx].set_xticks(np.array(range(len(class_labels)))+0.5)
    ax[idx].set_xticklabels(class_labels)
    ax[idx].set_yticks(np.array(range(len(class_labels)))+0.5)
    ax[idx].set_yticklabels(class_labels)

plt.show()


# In[238]:


# YOUR CODE HERE
#raise NotImplementedError()
tp, fn, fp, tn = CM.ravel()
print(f'True Negative = {tn:.0f}')
print(f'False Positive = {fp:.0f}')
print(f'False Negative = {fn:.0f}')
print(f'True Positive = {tp:.0f}')


# In[239]:


assert all(CM.ravel() == [tp, fn, fp, tn])


# __Question 2.3__  Write some code to compute the precision, recall, and F1_score from the variables `tn`, `fp`, `fn`, and `tp`, save your results to variables __recall__, __precision__, and __F1__. (Because the difference in how to labeling the classes, it is possible that the precision, recall, and f1 scores are different from those computed from skleran.metrics.)

# In[122]:


# YOUR CODE HERE
#raise NotImplementedError
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2 * (precision * recall) / (precision + recall)
print(f'Precison = {precision:.4f}')
print(f'Recall = {recall:.4f}')
print(f'F1 = {f1:.4f}')


# In[123]:


assert precision == CM[0, 0] / CM[:, 0].sum() 


# In[124]:


assert recall == CM[0, 0] / CM[0, :].sum()


# In[125]:


assert abs(f1 - 2 * (precision * recall) / (precision + recall)) < 1.0e-5


# # 3. Expected Value
# 
# When we are applying data science to an actual application like Breast Cancer Prediction, one important principle is to return to the key question: what is the actual goal of the application? what is the true value that an classifier can bring in?
# 
# In the above study, the accuracy as a classification metric makes no distinction between false positive and false negative errors. Altough recall and precision provide more specific performance measurement, we still have the question of how to connect the performnace of a classifier to real-world domains.
# 
# Expect value provides a key analytical framework for real-world data science problems. Below is a diagram of expected value calculation borrowed from chapter 7 of "Data Science for Business" by Provost and Fawcett.
# 
# !['expected value'](https://www.palmetto.clemson.edu/dsci/figures/ps04b_excpected_value.jpg)
# 
# In general, the expected value can be calculated by:
# $$EV = \sum_{i=1}^{m}P(o_{i}) \cdot V(o_{i})$$
# 
# Where, $P(o_{i})$ is ith possible pediction outcome and $V(o_{i})$ is the corresponding value for the ith outcome. The value information is domain specific and needs to be learned separately.

# In[128]:


CM = np.array([[52,  1], [ 3, 87]])
VM = np.array([[1000, -20000], [-2000, 10000]])


# __Question 3.1__ Given a confusion matrix of a classifier, __CM__, and a cost_benefit_matrix __VM__ shown as above, compute the __expcted value__ of the classifier?

# In[245]:


import numpy as np
def compute_expected_value(C, V):
    """ computes expected values from a confusion matrix and a cost benefit matrix
    
    Args:
        C (np.ndarray) : a confusion matrix
        V (np.ndarray) : a cost benefit matrix
    
    Returns:
        EV(float) : the expected value
        
    Examples:
        >>> print(compute_expected_value(np.array[[1, 2], [3, 4]], np.array[[1, 1], [1, 1]]))
        
    """
# YOUR CODE HERE
#raise NotImplementedError()
    norm = C/np.sum(C)
    pv = np.multiply(norm,V)
    ev = np.sum(pv)
    return ev


# In[248]:


ev = compute_expected_value(CM, VM)
print(f'Expected Value = ${ev:.2f}')


# In[249]:


assert compute_expected_value(np.array([[1, 2], [3, 4]]), np.array([[1, 1], [1, 1]])) == 1.0


# In[250]:


assert compute_expected_value(np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]]), np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1]])) - 0.61111 < 1e-5


# __Question 3.2__ Briefly explain why expected value could be a better performance metric than other performance metrics like accuracy or precision?

# YOUR ANSWER HERE

# In[251]:


#It takes into account false positive and negative error and
#cost/benefit information. This makes it more applicable to a
#real-world situation.


# # 4. Classification Probability

# Several classifiers not only can predict the label of an input sample but also can output the probability of belongs to a class. In scitkit-learn, a classifier has method `predict_proba()`. The probabilities for each class will be calculated in a way that depends on the specific model, but they should yield probabilities for each class for each sample you feed into it.
# 
# The following code creates the histograms and calibrated probability for four classifiers.

# In[219]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

cancer = load_breast_cancer(as_frame=True)
features = cancer.data
target = cancer.target.astype('category')
X_train_raw, X_test_raw, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.25, random_state=8)
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

models = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC(gamma='auto', probability=True),
    'LogisticRegression': LogisticRegression(fit_intercept=True, max_iter=5000),
    'GaussianNB': GaussianNB(),
}

fig, ax = plt.subplots(2, 4, figsize=(20, 16))
idx = 0
n_bins = 10
for model_name, model in models.items():
    clf = model
    _ = clf.fit(X_train, y_train)

    y_test_probability = clf.predict_proba(X_test)
    prob_postive = y_test_probability[:, 0]
    prob_negative = y_test_probability[:, 1]
    
    fraction_of_positives, predicted_positive_prob = calibration_curve(y_test, prob_postive, n_bins=20)
    fraction_of_negative, predicted_nagative_prob = calibration_curve(y_test, prob_negative, n_bins=20)
    
    ax[0, idx].hist(y_test_probability[:, 0], bins=n_bins)
    ax[0, idx].set_title(model_name)
    ax[0, idx].set_ylim(0, 0.2*len(y_train))
    ax[0, idx].set_xlabel('Predicted Probability')
    ax[0, idx].set_ylabel('Counts')
    handles, labels = ax[0, idx].get_legend_handles_labels()
    labels = ['Malignant', 'Benign']
    ax[0, idx].legend(handles, labels)
    
    
    ax[1, idx].plot(predicted_positive_prob, fraction_of_positives, "s-", label="postive")
    ax[1, idx].plot(predicted_nagative_prob, fraction_of_negative, "s-", label="negative")
    ax[1, idx].set_title(model_name)
    ax[1, idx].set_xlabel('Predicted Probability')
    ax[1, idx].set_ylabel('Fraction of Positives or Negatives')
    ax[1, idx].legend(loc='center right')
    idx = idx + 1


# __Question 4.1__ Based upon the above plots, which of the following statement is likely to be false?
# 
# ```
# A. For Random Forest, SVC, and LogisticRegression classifiers used in this problem, almost all the test samples with a predicted probability of zero are true positive samples.
# B. For Random Forest, SVC, and LogisticRegression classifiers used in this problem, almost all the test samples with a predicted probability of zero are predicted correctly.
# C. For Gaussian Naive Bayes Classifier, even a sample has a 100% predicted probability to be positive, it is still likely that the sample is a negtive case.
# D. In real-world applications, we can recalculate the classification probability of a given classifier (i.e., setting a threshold of predicted probability for a class) to achieve a desired prediction performance (which can be accuracy, precision, or expected values).
# ```

# In[252]:


# YOUR CODE HERE
#raise NotImplementedError()
answer = 'A'


# __Question 4.2__ Complete the following code to combine the prediction of the four classifiers and then evaluate the accuracy of the combined prediction.

# In[255]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import numpy as np 

cancer = load_breast_cancer(as_frame=True)
features = cancer.data
target = cancer.target.astype('category')

X_train_raw, X_test_raw, y_train, y_test = train_test_split(features, target, stratify=target, test_size=0.25, random_state=8)
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

models = {
    'RandomForest': RandomForestClassifier(),
    'SVC': SVC(gamma='auto', probability=True),
    'LogisticRegression': LogisticRegression(fit_intercept=True, max_iter=5000)
}

    
class CombinedClassifier:
    def __init__(self, models, combine_method='geo_mean'):
        self.models = {
            label: clf for label, clf in models.items()
        }
        self.method=combine_method
    
    def fit(self, X, y):
        for label, clf in self.models.items():
            clf.fit(X, y)
        self.n_classes = np.unique(y).shape[0]
            
    def predict(self, X):
# YOUR CODE HERE
#raise NotImplementedError()
        prob = self.predict_proba(X)
        labels = np.argmax(prob, axis=1)
        return labels

    def predict_proba(self, X):
# YOUR CODE HERE
#raise NotImplementedError()
        prob_list = []
        for model_name, model in self.models.items():
            prob_list.append(clf.predict_proba(X))
        probs = np.dstack(prob_list)
        if self.method == 'mean':
            prob = np.mean(probs, axis = 2)
        elif self.method == 'median':
            prob = np.median(probs, axis = 2)
        else:
            prob = np.zeros((probs.shape[0], probs.shape[1]))
            for i in range(probs.shape[0]):
                for j in range(probs.shape[1]):
                    prob[i,j] = probs[i,j].prod()**(1.0/probs.shape[2])
        return prob

combined_clf = CombinedClassifier(models)
combined_clf.fit(X_train, y_train)


# In[256]:


y_test_pred = combined_clf.predict(X_test[:, :])
cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True)
plt.show()


# __Question 4.3__ Complete the following code to find the prediction probability of the test samples with incorrect predictions.

# In[257]:


df = pd.DataFrame(combined_clf.predict_proba(X_test), columns=[f'prob_{label}' for label in cancer.target_names])
df['pred_label'] = y_test_pred
df['true_label'] = np.array(y_test)
# YOUR CODE HERE
#raise NotImplementedError()
df[df['pred_label']!=df['true_label']]


# __End of Part B__

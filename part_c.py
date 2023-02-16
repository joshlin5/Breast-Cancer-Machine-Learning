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

# # Part C. Spam Email Classification
# 
# In this part, you will build a spam email classifier using several classification methods. For your convinience, we have downloaded a dataset from https://www.kaggle.com/venky73/spam-mails-dataset and put the csv file at https://www.palmetto.clemson.edu/dsci/datasets/kaggle/spam_ham_dataset.csv.

# In[2]:


# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
sns.set_style("white")


# ## 1. Get the data
# 
# Read the spam dataset into a Pandas DataFrame and examine the data.

# In[3]:


df = pd.read_csv("https://www.palmetto.clemson.edu/dsci/datasets/kaggle/spam_ham_dataset.csv", index_col=0)
df.head()


# ## 2. Examin the data

# In[4]:


## Look at a summary of the dataset
df.info()


# In[5]:


## Get unique values of the label
np.unique(df['label'])


# In[6]:


### Look at the class distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
class_counts = df['label'].value_counts()
class_percents = class_counts / np.sum(class_counts) * 100
_ = sns.barplot(x=class_counts.index, y=class_counts, ax = ax[0])
ax[0].set_ylabel('Class Count')
_ = sns.barplot(x=class_percents.index, y=class_percents, ax = ax[1])
ax[1].set_ylabel('Class Percentage')
plt.show()


# In[7]:


## write a helper to print the data
def print_email(df, index):
    print(f'email {index}')
    for k, v in df.loc[0].items():
        print(f'\t{k} = {v}')


# In[8]:


## Print email
print_email(df, 1)


# ## 3. Split the data into training set and test set
# 
# Use the function `sklearn.model_selection.train_test_split` to create a train set and a test set from the original data. The test set will contain 20% of the total samples. For repeatable results, set a value for the `random_state` parameter.

# In[9]:



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df["text"],df["label"], test_size = 0.2, random_state = 10)


# In[10]:


## While it is not always necessary, as a good practice, you may look at some statistics of the train set and test set.

class_counts = pd.DataFrame({'Train Set': y_train.value_counts().sort_index(),
                           'Test Set': y_test.value_counts().sort_index()})
class_percents = class_counts / np.sum(class_counts)

class_counts.head(5)
class_percents.head(5)


# ## 4. Preprocess the data using CountVectorizer (5 points)
# 
# Before feed the data into a machine learning algorithm, we need to convert the text to a ser of representative numerical values. One of the simplest methods of encoding data is by word counts: you take each snippet of text, count the occurrences of each word within it, and put the results in a table.
# 
# To vectorizing this email data based on word count, you can construct a column representing each word using the class `sklearn.feature_extraction.text.CountVectorizer`. 

# In[11]:


## Create a demo corpus
corpus = [df.loc[0].text, df.loc[2].text]
corpus


# In[12]:


## Illustrate the concepts and use of CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

vect2 = CountVectorizer()
x2 = vect2.fit_transform(corpus)

print(vect2.vocabulary_ )
print(vect2.get_feature_names())
print(x2.toarray())


# __Question 4.1__ Write some code in the folllowing cell to vectorize the train set data. Save the transformed results to a variable `X_train_tran`. (5 points)

# In[18]:


# YOUR CODE HERE
#raise NotImplementedError()
vect = CountVectorizer()
vect.fit(X_train)
X_train_tran = vect.transform(X_train)
X_train_tran


# __Question 4.2__ What is the type and dimension of the `X_train_tran`? (2 points)

# Write the type and dimension of the `X_train_tran`

# In[19]:


#type: sparse matrix of type '<class 'numpy.int64'>'
#dimension: 4136 x 44573


# ## 5. Train a model (5 points)

# __Question 5.1__ Now build a classifier using `sklearn.ensemble.RandomForestClassifier` and train the model using the proprecessed data. Write some code to train the model. Save the model to a variable `forest_clf`.

# In[24]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit(X_train_tran, y_train)
forest_clf


# ## 6. Evaluate the performances  of the model using the test set (5 points)

# __Question 6.1__ Write some code in the following cell to evalue the model on the test set using `sklearn.metrics.accuracy_score`.

# In[26]:


from sklearn.metrics import accuracy_score

# YOUR CODE HERE
#raise NotImplementedError()
X_test_trans = vect.transform(X_test)
y_test_pred = forest_clf.predict(X_test_trans)
accuracy_score(y_test, y_test_pred)


# ## 7. Make predictions

# In[27]:


message1 ="""Join Red Hat and Emergent at our upcoming, free Ansible Automation Technical Workshop in Greenville
on December 10.
Our most popular workshop, the Red Hat Ansible lab takes you through running ad-hoc commands in Core, 
writing and running a playbook, using more advanced features such as variables, loops, and handlers, and installing, configuring, and running job templates in Tower. The academic delivery model creates comfort and familiarity with this popular tool.
The hands-on workshop includes:
•	Exercise 1.1 - Ad-Hoc Commands
•	Exercise 1.2 - Writing Your First Playbook
•	Exercise 1.3 - Running Your First Playbook
•	Exercise 1.4 - Using Variables, loops, and handlers
•	Exercise 1.5 - Running the apache-basic-playbook
•	Exercise 1.6 - Roles: Making your playbooks reusable
•	Exercise 2.1 - Installing Ansible Tower
•	Exercise 2.2 - Configuring Ansible Tower
•	Exercise 2.3 - Creating and Running a Job Template
•	Resources, Wrap Up
To offer a quality lab experience, seating will be limited so reserve your spot today! 
"""

message2 = """𝚆𝚎𝚕𝚌𝚘𝚖𝚎 𝚝𝚘 𝙰 𝙰𝚋𝚜𝚘𝚕𝚞𝚝𝚎𝚕𝚢 𝙵𝚛𝚎𝚎 $𝟻𝟶 𝙰𝚖𝚊𝚣𝚘𝚗.𝚌𝚘𝚖® 𝙶𝚒𝚏𝚝 𝙲𝚊𝚛𝚍 (𝙻𝙰𝚂𝚃 𝙽𝙾𝚃𝙸𝙲𝙴: 𝟸𝟺 𝙷𝚘𝚞𝚛𝚜 𝙻𝚎𝚏𝚝 𝚝𝚘 𝙲𝚕𝚊𝚒𝚖) <http://en658kejf4nnmi.w0.wincvs5.tk/t/dq7wAALhxoQrfCQAAQnDFXQEAAADZtiiECCxIBAA/g/XQX> 

𝗗𝗲𝗮𝗿 𝗖𝘂𝘀𝘁𝗼𝗺𝗲𝗿, do you love online shopping at Amazon.com®... or hate it? Care to share your experience? Your opinion is worth $50 to us! 

You are one of 5 customers selected to participate in our 30-second anonymous survey, <http://en658kejf4nnmi.w0.wincvs5.tk/t/dq7wAALhxoQrfCQAAQnDFXQIAAADZtiiECCxIBAA/g/XQX>  today, November 4, 2019. 4 participants have already claimed their $50 reward. 𝗪𝗵𝗮𝘁 𝗮𝗯𝗼𝘂𝘁 𝘆𝗼𝘂? 

𝗚𝗲𝘁 𝗬𝗼𝘂𝗿 𝗙𝗥𝗘𝗘  
$𝟱𝟬 𝗔𝗺𝗮𝘇𝗼𝗻.𝗰𝗼𝗺® 
𝗚𝗶𝗳𝘁 𝗖𝗮𝗿𝗱* <http://en658kejf4nnmi.w0.wincvs5.tk/t/dq7wAALhxoQrfCQAAQnDFXQMAAADZtiiECCxIBAA/g/XQX> 

 <http://en658kejf4nnmi.w0.wincvs5.tk/t/dq7wAALhxoQrfCQAAQnDFXQQAAADZtiiECCxIBAA/g/XQX>

Hurry, your code AZ2019 expires in 24 hours! 
Confirm your survey participation status here: <http://en658kejf4nnmi.w0.wincvs5.tk/t/dq7wAALhxoQrfCQAAQnDFXQUAAADZtiiECCxIBAA/g/XQX> 

* Or receive other valuable rewards or discounts valued at or above $50.00 USD. Please answer survey questions honestly; your evaluation is anonymous and answers will not affect your rewards eligibility. Independent survey not affiliated with, or sponsored by, Amazon.com®, whose trademark and/or logo is property of its owner. This is an advertisement
"""


# __Question 7.1__ Use the model you just trained, predict whether the above two messages are spam or ham. (5 points)

# In[32]:


# YOUR CODE HERE
#raise NotImplementedError()
for i,m in enumerate([message1, message2]):
    messages_tran = vect.transform([m])
    messages_pred = forest_clf.predict(messages_tran)
    print("message{} is {}".format(i+1, messages_pred[0]))


# __Question 7.2__ Write some code to print the probability of message1 in the above prediction.

# In[89]:


# YOUR CODE HERE
#raise NotImplementedError()
sample = vect.transform([message1])
p = forest_clf.predict_proba(sample)
probability = p[0][1]
print("probability message1 is spam is {}".format(probability))


# ## 8. Model Robustness

# __Question 8.1__ Repeat step 4 through  step 6 for ten times. Print the accuracy score and the predictions of each model using the test data. After the loop, print the highest score and the lowest score of these models. (5 points)

# In[92]:


from sklearn.metrics import accuracy_score

message_tran = vect.transform([message1, message2])
np.random.seed(10)

def repeat_train_and_test(clf, n_repeats=10): 
    accuracy_scores = []
    message1_pred = []
    message2_pred = []

# YOUR CODE HERE
#raise NotImplementedError()
    for i in range(n_repeats):
        
        vect = CountVectorizer()
        vect.fit(X_train)
        X_train_tran = vect.transform(X_train)
        X_test_tran = vect.transform(X_test)
        
        clf.fit(X_train_tran, y_train)
        y_test_pred = clf.predict(X_test_tran)
        acc_score = accuracy_score(y_test, y_test_pred)
        
        message1_tran = vect.transform([message1])
        pred_message1 = clf.predict(message1_tran)
        
        message2_tran = vect.transform([message2])
        pred_message2 = clf.predict(message2_tran)
        
        accuracy_scores.append(acc_score)
        message1_pred.append(pred_message1)
        message2_pred.append(pred_message2)
        
        df_perf = pd.DataFrame({
            'accuracy' : accuracy_scores ,
            'message1' : message1_pred,
            'message2' : message1_pred,
        })
    
    high_score = df_perf['accuracy'].max()
    low_score = df_perf['accuracy'].min()
    print(df_perf)
    print(f'highest score = {high_score}\nlowest score = {low_score}')
    
    return df_perf

forest_clf_model = RandomForestClassifier(n_estimators=50)
df_perf = repeat_train_and_test(forest_clf_model, 10)


# __Question 8.2__ Based on teh above results, what would you recommend your customer to do when she want to make classification using some machine learning algorithms which the vendor claims an 95% accuracy? (5 points)

# Write your response below:

# In[69]:


#I would recommend my customer to use the RandomForestClassifier since the lowest accuracy is about 95.7%


# ## 9. TFIDF model
# 
# Recall that raw word counts lead to features which put too much weights on words that appear too frequently. One approach to fix this issue is known as term frequency-inverse document frequency (TF–IDF) which weights the word counts by a statistics of how often they appear in the documents. You can use the `sklearn.feature_extraction.text.TfidfVectorizer` to preprocee the data and then train and test a classification model. (5 points)

# __Question 9.1__ Write some code below to transform the corpus using the TFIDF model, then train a RandomForestClassifier model and test the model in the same way as you did in Step 8.

# In[98]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.feature_extraction.text import TfidfVectorizer
        
vect2 = TfidfVectorizer()
vect2.fit(X_train)

X_test_tran = vect2.transform(X_test)
X_train_tran = vect2.transform(X_train)

def repeat_train_and_test(clf, n_repeats=10): 
    accuracy_scores = []
    message1_pred = []
    message2_pred = []
    
    for i in range(n_repeats):
        
        clf.fit(X_train_tran, y_train)
        y_test_pred = clf.predict(X_test_tran)
        acc_score = accuracy_score(y_test, y_test_pred)
        
        message1_tran = vect2.transform([message1])
        pred_message1 = clf.predict(message1_tran)
        
        message2_tran = vect2.transform([message2])
        pred_message2 = clf.predict(message2_tran)
        
        accuracy_scores.append(acc_score)
        message1_pred.append(pred_message1)
        message2_pred.append(pred_message2)
        
        df_perf = pd.DataFrame({
            'accuracy' : accuracy_scores ,
            'message1' : message1_pred,
            'message2' : message1_pred,
        })
    
    high_score = df_perf['accuracy'].max()
    low_score = df_perf['accuracy'].min()
    print(df_perf)
    print(f'highest score = {high_score}\nlowest score = {low_score}')
    
    return df_perf

forest_tdif_model = RandomForestClassifier(n_estimators=50)
repeat_train_and_test(forest_tdif_model, 10)


# __Question 9.2__ Write down any thought or discovery that you may have based upon the above results. (5 points)

# Write your thought or discovery.

# In[100]:


#When comparing the two models, this one says that all the messages are spam with a higher accuracy, which means that this
#model should be more accurate after taking into account the repeated words.


# ## 10. Classification Using Naive Bayes

# In the class, we have discussed that Naive Bayes is a great choice for text analysis. Now train a Naive Bayes model to classify the spam emails. You may use the `CountVectorizer` model to preprocess the data. For the Naive Bayes, you may use the class `sklearn.naive_bayes.MultinomialNB`.
# 
# __Question 10_1__ Write some code to train and test a Naive Bayes modelas what you did in Step 9. (5 poinst)

# In[94]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.naive_bayes import MultinomialNB

vect = CountVectorizer()
vect.fit(X_train)

X_test_tran = vect.transform(X_test)
X_train_tran = vect.transform(X_train)

def repeat_train_and_test(clf, n_repeats=10): 
    accuracy_scores = []
    message1_pred = []
    message2_pred = []
    
    for i in range(n_repeats):
        
        clf.fit(X_train_tran, y_train)
        y_test_pred = clf.predict(X_test_tran)
        acc_score = accuracy_score(y_test, y_test_pred)
        
        message1_tran = vect.transform([message1])
        pred_message1 = clf.predict(message1_tran)
        
        message2_tran = vect.transform([message2])
        pred_message2 = clf.predict(message2_tran)
        
        accuracy_scores.append(acc_score)
        message1_pred.append(pred_message1)
        message2_pred.append(pred_message2)
        
        df_perf = pd.DataFrame({
            'accuracy' : accuracy_scores ,
            'message1' : message1_pred,
            'message2' : message1_pred,
        })
    
    high_score = df_perf['accuracy'].max()
    low_score = df_perf['accuracy'].min()
    print(df_perf)
    print(f'highest score = {high_score}\nlowest score = {low_score}')
    
    return df_perf

navive_bayes_model = MultinomialNB()
_ = repeat_train_and_test(navive_bayes_model, 10)


# ## 11. Classification Using SVM

# __Question 11_1__ Write code to train a SVM classifier. (5 points)

# In[95]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn import svm

vect = CountVectorizer()
vect.fit(X_train)

X_test_tran = vect.transform(X_test)
X_train_tran = vect.transform(X_train)

def repeat_train_and_test(clf, n_repeats=10): 
    accuracy_scores = []
    message1_pred = []
    message2_pred = []
    
    for i in range(n_repeats):
        
        clf.fit(X_train_tran, y_train)
        y_test_pred = clf.predict(X_test_tran)
        acc_score = accuracy_score(y_test, y_test_pred)
        
        message1_tran = vect.transform([message1])
        pred_message1 = clf.predict(message1_tran)
        
        message2_tran = vect.transform([message2])
        pred_message2 = clf.predict(message2_tran)
        
        accuracy_scores.append(acc_score)
        message1_pred.append(pred_message1)
        message2_pred.append(pred_message2)
        
        df_perf = pd.DataFrame({
            'accuracy' : accuracy_scores ,
            'message1' : message1_pred,
            'message2' : message1_pred,
        })
    
    high_score = df_perf['accuracy'].max()
    low_score = df_perf['accuracy'].min()
    print(df_perf)
    print(f'highest score = {high_score}\nlowest score = {low_score}')
    
    return df_perf

svm_model = svm.SVC()
_ = repeat_train_and_test(svm_model, 10)


# __Question 11_2__ Repeat Question 10_1 but now force the SVM classifier to use a linear kernel and set parameter C=2. (5 points)

# In[96]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn import svm

vect = CountVectorizer()
vect.fit(X_train)

X_test_tran = vect.transform(X_test)
X_train_tran = vect.transform(X_train)

def repeat_train_and_test(clf, n_repeats=10): 
    accuracy_scores = []
    message1_pred = []
    message2_pred = []
    
    for i in range(n_repeats):
        
        clf.fit(X_train_tran, y_train)
        y_test_pred = clf.predict(X_test_tran)
        acc_score = accuracy_score(y_test, y_test_pred)
        
        message1_tran = vect.transform([message1])
        pred_message1 = clf.predict(message1_tran)
        
        message2_tran = vect.transform([message2])
        pred_message2 = clf.predict(message2_tran)
        
        accuracy_scores.append(acc_score)
        message1_pred.append(pred_message1)
        message2_pred.append(pred_message2)
        
        df_perf = pd.DataFrame({
            'accuracy' : accuracy_scores ,
            'message1' : message1_pred,
            'message2' : message1_pred,
        })
    
    high_score = df_perf['accuracy'].max()
    low_score = df_perf['accuracy'].min()
    print(df_perf)
    print(f'highest score = {high_score}\nlowest score = {low_score}')
    
    return df_perf

svm_model = svm.SVC(kernel = 'linear', C=2)
_ = repeat_train_and_test(svm_model, 10)


# __Question 11_3__ Based on the above results, do you think that a nonlinear kernel always performs better than a linear kernel? Justify your answer. (5 points)

# 1. Is a nonlinear kernel always performs better than a linear kernel?
# 2. Your justification:

# In[77]:


#I do not think a nonlinear kernel always performs better than a linear kernel since the linear kernel gives a higher
#accuracy score than the nonlinear kernel.


# ## 12. Classifiaction Using Logistic Regression

# __Question 12_1__ Write some code to train a Logistic Regression classifier. You can setthe `max_iter` to 1000 so that the algorithm will converge without warning.(5 points)

# In[97]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.linear_model import LogisticRegression
vect = CountVectorizer()
vect.fit(X_train)

X_test_tran = vect.transform(X_test)
X_train_tran = vect.transform(X_train)

def repeat_train_and_test(clf, n_repeats=10): 
    accuracy_scores = []
    message1_pred = []
    message2_pred = []
    
    for i in range(n_repeats):
        
        clf.fit(X_train_tran, y_train)
        y_test_pred = clf.predict(X_test_tran)
        acc_score = accuracy_score(y_test, y_test_pred)
        
        message1_tran = vect.transform([message1])
        pred_message1 = clf.predict(message1_tran)
        
        message2_tran = vect.transform([message2])
        pred_message2 = clf.predict(message2_tran)
        
        accuracy_scores.append(acc_score)
        message1_pred.append(pred_message1)
        message2_pred.append(pred_message2)
        
        df_perf = pd.DataFrame({
            'accuracy' : accuracy_scores ,
            'message1' : message1_pred,
            'message2' : message1_pred,
        })
    
    high_score = df_perf['accuracy'].max()
    low_score = df_perf['accuracy'].min()
    print(df_perf)
    print(f'highest score = {high_score}\nlowest score = {low_score}')
    
    return df_perf

logistic_model = LogisticRegression(max_iter = 1000)
_ = repeat_train_and_test(logistic_model, 10)


# __Question 12_2__. Logistic Regression classifier comes with a probability estimation. Print the probability distributions for the predictions on the two messages. (5 points)

# In[99]:


# YOUR CODE HERE
#raise NotImplementedError()
logistic_model.fit(X_train_tran, y_train)

tran1 = vect.transform([message1])
pred1 = logistic_model.predict_proba(tran1)
        
tran2 = vect.transform([message2])
pred2 = logistic_model.predict_proba(tran2)
print("probability message1 is spam is {} and is not spam is {}".format(pred1[0][1], pred1[0][0]))
print("probability message1 is spam is {} and is not spam is {}".format(pred2[0][1], pred2[0][0]))


# ## 13. Compare the Performance of Multiple Classifiers

# __Question 13.1__ Complete the following code to compare the performance of the following classifier.
# 
# ```
# LogisticRegression
# NaiveBayes
# RandomForest
# SVC-Rbf with rbf kernel
# SVC-Linear with linear kernel
# KNN
# DecisionTree
# ```

# In[105]:


from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size = 0.2, random_state = 10)
 
models = [LogisticRegression(max_iter = 1000), MultinomialNB(), RandomForestClassifier(n_estimators=50), SVC(), SVC(kernel = 'linear'), KNeighborsClassifier(), DecisionTreeClassifier()]
CountVectorizer_accuracy = []
TfidfVectorizer_accuracy = []

# YOUR CODE HERE
#raise NotImplementedError()

def train_and_test(clf): 
    
    vect = CountVectorizer()
    vect.fit(X_train)

    X_test_tran = vect.transform(X_test)
    X_train_tran = vect.transform(X_train)
    clf.fit(X_train_tran, y_train)
    y_test_pred = clf.predict(X_test_tran)
    CV_acc = accuracy_score(y_test, y_test_pred)
    
    vect2 = TfidfVectorizer()
    vect2.fit(X_train)

    X_test_tran2 = vect2.transform(X_test)
    X_train_tran2 = vect2.transform(X_train)
    clf.fit(X_train_tran2, y_train)
    y_test_pred2 = clf.predict(X_test_tran2)
    TV_acc = accuracy_score(y_test, y_test_pred2)
    
    return CV_acc, TV_acc

for model in models:
    CV_acc, TV_acc = train_and_test(model)
    CountVectorizer_accuracy.append(CV_acc)
    TfidfVectorizer_accuracy.append(TV_acc)


# In[106]:


df_perf = pd.DataFrame({'CountVectorizer': CountVectorizer_accuracy,
                       'TfidfVectorizer': TfidfVectorizer_accuracy})
df_perf


# ## 14.  SMS Message Classification
# 
# Using the same method, you can classify other dataset encoded in other language.

# In[107]:


df = pd.read_csv("https://www.palmetto.clemson.edu/dsci/datasets/kaggle/spam.csv", encoding='latin-1')


# In[108]:


df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)

print(df.shape)
df.head()


# __Question 14.1__ Complete the following code to test the performance of the following classifers. (5 points)
# 
# ```
# LogisticRegression
# NaiveBayes
# RandomForest
# SVC-Rbf with rbf kernel
# SVC-Linear with linear kernel
# KNN
# DecisionTree
# ```

# In[109]:


from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size = 0.2, random_state = 10)
 
models = [LogisticRegression(max_iter = 1000), MultinomialNB(), RandomForestClassifier(n_estimators=50), SVC(), SVC(kernel = 'linear'), KNeighborsClassifier(), DecisionTreeClassifier()]
CountVectorizer_accuracy = []
TfidfVectorizer_accuracy = []

# YOUR CODE HERE
#raise NotImplementedError()

def train_and_test(clf): 
    
    vect = CountVectorizer()
    vect.fit(X_train)

    X_test_tran = vect.transform(X_test)
    X_train_tran = vect.transform(X_train)
    clf.fit(X_train_tran, y_train)
    y_test_pred = clf.predict(X_test_tran)
    CV_acc = accuracy_score(y_test, y_test_pred)
    
    vect2 = TfidfVectorizer()
    vect2.fit(X_train)

    X_test_tran2 = vect2.transform(X_test)
    X_train_tran2 = vect2.transform(X_train)
    clf.fit(X_train_tran2, y_train)
    y_test_pred2 = clf.predict(X_test_tran2)
    TV_acc = accuracy_score(y_test, y_test_pred2)
    
    return CV_acc, TV_acc

for model in models:
    CV_acc, TV_acc = train_and_test(model)
    CountVectorizer_accuracy.append(CV_acc)
    TfidfVectorizer_accuracy.append(TV_acc)


# In[110]:


df_perf = pd.DataFrame({'CountVectorizer': CountVectorizer_accuracy,
                       'TfidfVectorizer': TfidfVectorizer_accuracy})
df_perf


# __Question 14.2__ Write any observation you may have in the above results. (5 points)

# Write your observation and/or findings.

# In[ ]:


#Although TfidfVectorizer is technically more accurate since it accounts for repeated words, its accuracy score is lower.


# ## 15. Multiclass Classification
# 
# Now applying the set of classifers that you have used to a multiclass classification problem. The dataset you will use is the tweets about airline services.

# In[112]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# In[113]:


df = pd.read_csv("https://www.palmetto.clemson.edu/dsci/datasets/kaggle/tweets.csv")
df.head()


# In[114]:


df.info()


# In[115]:


np.unique(df['airline_sentiment'])


# In[116]:


features = df.iloc[:, 10].values
labels = df.iloc[:, 1].values


# __Question 15.1__ Complete the following code to test the performance of the following classifers on the airline tweets data. (10 points)
# 
# ```
# LogisticRegression-ovr (multi_class='ovr')
# LogisticRegression-multinomial (multi_class='multinomial')
# NaiveBayes
# RandomForest
# SVC-Rbf with rbf kernel
# SVC-Linear with linear kernel
# KNN
# DecisionTree
# ```

# In[117]:


X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size = 0.2, random_state = 10)

models = [LogisticRegression(multi_class = 'ovr', max_iter = 1000), LogisticRegression(multi_class = 'multinomial', max_iter = 1000), MultinomialNB(), RandomForestClassifier(n_estimators=50), SVC(), SVC(kernel = 'linear'), KNeighborsClassifier(), DecisionTreeClassifier()]
CountVectorizer_accuracy = []
TfidfVectorizer_accuracy = []

# YOUR CODE HERE
#raise NotImplementedError()

def train_and_test(clf): 
    
    vect = CountVectorizer()
    vect.fit(X_train)

    X_test_tran = vect.transform(X_test)
    X_train_tran = vect.transform(X_train)
    clf.fit(X_train_tran, y_train)
    y_test_pred = clf.predict(X_test_tran)
    CV_acc = accuracy_score(y_test, y_test_pred)
    
    vect2 = TfidfVectorizer()
    vect2.fit(X_train)

    X_test_tran2 = vect2.transform(X_test)
    X_train_tran2 = vect2.transform(X_train)
    clf.fit(X_train_tran2, y_train)
    y_test_pred2 = clf.predict(X_test_tran2)
    TV_acc = accuracy_score(y_test, y_test_pred2)
    
    return CV_acc, TV_acc

for model in models:
    CV_acc, TV_acc = train_and_test(model)
    CountVectorizer_accuracy.append(CV_acc)
    TfidfVectorizer_accuracy.append(TV_acc)


# In[118]:


df_perf = pd.DataFrame({'CountVectorizer': CountVectorizer_accuracy,
                       'TfidfVectorizer': TfidfVectorizer_accuracy})
df_perf


# __Question 15.2__ Plot the confusion matrix of SVC-Linear. (5 points)

# In[123]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.metrics import plot_confusion_matrix
clf = SVC(kernel = 'linear')
vect = TfidfVectorizer()
vect.fit(X_train)

X_test_tran = vect.transform(X_test)
X_train_tran = vect.transform(X_train)
clf.fit(X_train_tran, y_train)
y_test_pred = clf.predict(X_test_tran)
plot_confusion_matrix(clf, X_test_tran, y_test_pred)


# __Question 15.3__ Print the true label, predicted label, and the predicted probabilities for the first five test samples using the `LogisticRegression-multinomial classifer` (5 points)

# In[141]:


# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size = 5)
clf = LogisticRegression(multi_class = 'multinomial', max_iter = 1000)
vect = TfidfVectorizer()
vect.fit(X_train)

X_test_tran = vect.transform(X_test)
X_train_tran = vect.transform(X_train)
clf.fit(X_train_tran, y_train)
y_test_pred = clf.predict(X_test_tran)
#plot_confusion_matrix(clf, X_test_tran, y_test_pred)
pred = clf.predict_proba(X_test_tran)
df = pd.DataFrame(pred, columns = ['Spam', 'Neutral', "Ham"])
df['Predicted Label'] = y_test_pred
df['True Label'] = np.array(y_test)
df


# __End of Part C__

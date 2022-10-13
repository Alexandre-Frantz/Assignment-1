#!/usr/bin/env python
# coding: utf-8

# # Space MSc Lecture: Machine Learning with Sklearn

# ## Installation

# [Installing sklearn (using a pip virtualenv)](https://scikit-learn.org/stable/install.html)

# In[1]:


get_ipython().system('python -m venv sklearn-venv')
get_ipython().system('sklearn-venv\\Scripts\\activate')
get_ipython().system('pip install -U scikit-learn')


# #### Check installation

# Check installed version

# In[2]:


get_ipython().system('python -m pip show scikit-learn')


# List all packages installed in the active virtualenv

# In[ ]:


# !python -m pip freeze 


# Show version and dependencies

# In[3]:


get_ipython().system('python -c "import sklearn; sklearn.show_versions()"')


# ## Import Libraries

# In[4]:


import numpy as np
import pandas as pd
import sklearn
import scipy.stats as st
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Processing

# ### Data Exploration

# Let us load and explore the Titanic dataset. This dataset shows the details of survivals and deaths for the Titanic sinking. 

# In[5]:


train = pd.read_csv('https://github.com/ipython-books'
                    '/cookbook-2nd-data/blob/master/'
                    'titanic_train.csv?raw=true')
test = pd.read_csv('https://github.com/ipython-books/'
                   'cookbook-2nd-data/blob/master/'
                   'titanic_test.csv?raw=true')


# #### Inspect three rows at the head and tail of training and test datasets

# In[6]:


train.head(3)


# In[7]:


train.tail(3)


# In[8]:


test.head(3)


# In[9]:


test.tail(3)


# #### Inspect a specific Feature (column)

# In[10]:


train["Survived"].tail(3)


# In[11]:


test["PassengerId"].tail(3)


# In[12]:


test["Sex"].tail()


# #### Get Feature description 

# In[13]:


train["Survived"].describe()


# #### Get Feature description for Sex in the test dataset

# In[14]:


test["Sex"].describe()


# #### Manual Feature Selection 
# 
# Select subset of Features: e.g., consider we have domain knowledge that Pclass, Sex and Age are important/correlated to survival 

# In[15]:


train[train.columns[[2, 4, 5, 1]]].head()


# In[16]:


data = train[['Age', 'Pclass', 'Survived']]


# #### Dataframe Inspection & Manipulation (using Pandas)

# In[17]:


data.head(3)


# ##### Attributes

# In[18]:


data.empty # check dimension


# In[19]:


data.shape # check dimension


# In[20]:


data.shape[0] # number of rows


# In[21]:


data.shape[1] # number of columns


# In[22]:


data.size #check size


# In[23]:


data.isnull().sum() #check missing value


# In[24]:


data.isnull().sum().sum() #total missing values


# In[25]:


data.dtypes #check data types


# In[26]:


data.info() # get full info


# In[27]:


data.columns #check features


# #### Add a new column: add train data 'Sex' column as 'Female' column

# In[28]:


data = data.assign(Female=train['Sex'] == 'female')


# In[29]:


data.columns #check features


# In[30]:


data.head(3)


# In[31]:


# Reorder the columns.
data = data[['Female', 'Age', 'Pclass', 'Survived']]


# In[32]:


data.head(3)


# In[33]:


data = data.dropna() # Remove missing values


# In[34]:


data.shape # check dimension


# In[35]:


data.head(3)


# ### Feature & Label Extraction 

# In[36]:


data_np = data.astype(np.int32).values


# In[37]:


X = data_np[:, :-1] #feature vector (get all columns but last)


# In[38]:


y = data_np[:, -1] #output labels (get only last column)


# In[39]:


X[:3]


# In[40]:


y[:3]


# ### Data Transformation

# In[41]:


# We define a few boolean vectors.
# The first column is 'Female'.
female = X[:, 0] == 1


# In[42]:


# The last column is 'Survived'.
survived = y == 1


# In[43]:


# This vector contains the age of the passengers.
age = X[:, 1]


# ### Data Visualization

# In[44]:


data.groupby('Female').Survived.mean().plot(kind='bar')


# In[45]:


data.groupby('Age').Survived.mean().plot(kind='line')


# In[46]:


data.groupby('Pclass').Survived.mean().plot(kind='bar') #people in Pclass that survived


# In[47]:


data.query('Female == True').groupby('Pclass').Survived.mean().plot(kind='bar') #people in first class and female that survived


# In[ ]:


# We compute a few histograms.
bins_ = np.arange(0, 81, 5)
S = {'male': np.histogram(age[survived & ~female],
                          bins=bins_)[0],
     'female': np.histogram(age[survived & female],
                            bins=bins_)[0]}
D = {'male': np.histogram(age[~survived & ~female],
                          bins=bins_)[0],
     'female': np.histogram(age[~survived & female],
                            bins=bins_)[0]}


# In[ ]:


# We now plot the data.
bins = bins_[:-1]
fig, axes = plt.subplots(1, 2, figsize=(10, 3),
                         sharey=True)
for ax, sex, color in zip(axes, ('male', 'female'),
                          ('#3345d0', '#cc3dc0')):
    ax.bar(bins, S[sex], bottom=D[sex], color=color,
           width=5, label='survived')
    ax.bar(bins, D[sex], color='k',
           width=5, label='died')
    ax.set_xlim(0, 80)
    ax.set_xlabel("Age (years)")
    ax.set_title(sex + " survival")
    ax.grid(None)
    ax.legend()


# ### Data Splitting

# In[48]:


# We split X and y into train and test datasets
# Spliting into 95% for training set and 5% for testing set
(X_train, X_test, y_train, y_test) =     ms.train_test_split(X, y, test_size=.05, random_state=0) 


# ## Training a Logistic Regression Model

# In[49]:


import sklearn.linear_model as lm
import sklearn.model_selection as ms


# In[50]:


logreg = lm.LogisticRegression() #instantiate the classifier.


# In[51]:


logreg.fit(X_train, y_train)
y_predicted = logreg.predict(X_test)


# In[52]:


result = logreg.score(X_test, y_test)
print(result)


# ### Cross Validation

# In[54]:


ms.cross_val_score(logreg, X, y, cv=5)


# In[56]:


grid = ms.GridSearchCV(
    logreg, {'C': np.logspace(-5, 5, 200)}, n_jobs=4)
grid.fit(X_train, y_train)
grid.best_params_


# In[57]:


scores = ms.cross_val_score(grid.best_estimator_, X, y, cv=5)


# In[58]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# ### Saving Models 

# In[59]:


import pickle


# In[60]:


# save the model to disk
filename = 'final_logreg_model.sav'
pickle.dump(logreg, open(filename, 'wb'))


# ## Training an SVM Model

# In[61]:


import sklearn.svm as svm


# In[62]:


# Declaring the SVC with no tunning
classifier = svm.SVC()

# Fitting the data. This is where the SVM will learn
classifier.fit(X_train, y_train)

# Predicting the result and giving the accuracy
score = classifier.score(X_test, y_test)

print(score)


# In[63]:


# save the model to disk
filename = 'svm_model.sav'
pickle.dump(classifier, open(filename, 'wb'))


# ### Cross Validation

# In[64]:


scores = ms.cross_val_score(classifier, X, y, cv=5)


# In[65]:


scores


# In[66]:


print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# ## Model Assessment Metrics

# In[67]:


from sklearn import metrics


# In[68]:


sklearn.metrics.SCORERS.keys() #get dictionary keys of all available scores


# In[69]:


f1_score = ms.cross_val_score(classifier, X, y, cv=5, scoring='f1')


# In[70]:


print("%0.2f F1-score with a standard deviation of %0.2f" % (f1_score.mean(), f1_score.std()))


# In[71]:


precision_score = ms.cross_val_score(classifier, X, y, cv=5, scoring='precision')


# In[72]:


print("%0.2f precision with a standard deviation of %0.2f" % (precision_score.mean(), precision_score.std()))


# In[73]:


recall_score = ms.cross_val_score(classifier, X, y, cv=5, scoring='recall')


# In[74]:


print("%0.2f recall with a standard deviation of %0.2f" % (recall_score.mean(), recall_score.std()))


# ### Loading Models 

# In[75]:


filename = 'final_logreg_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# In[76]:


filename = 'svm_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)


# ## Summary of Lessons Learned

# Machine Learning with Sklearn including:
# 
# - Sklearn (installation, model training and model assessment)  
# - Data Processing 
#     - inspection, manipulation, splitting, transformation and visualization 
# - Feature Selection & Extraction
# - Model Training with Logistic Regression and SVMs
# - Model Assessment 
# - Cross Validation
# - Saving and Loading Models

# ## Techncial Exercise
# - Available on Moodle Today
# - Due Date: midnight (23:59) on 28th of October 2022. 

# ## Further Resources

# This notebook is mostly based on lectures from [IPython Cookbook, Second Edition (2018)](https://ipython-books.github.io/)

# The following links provide further documentation for the topics discussed in this notebook. 

# - [Scikit-learn](https://scikit-learn.org/stable/index.html)
# 
# - [Scikit Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
# 
# - [Scikit SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# 
# - [Scikit Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# 
# - [Scikit Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
# 
# - [Kaggle SVM Titanic Model](https://www.kaggle.com/code/eltonpaes/titanic-survivals-with-svm)
# 
# 

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/images/IDSNlogo.png" width="300" alt="cognitiveclass.ai logo"  />
# </center>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>
# 

# In this notebook we try to practice all the classification algorithms that we have learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# ### Load Data From CSV File
# 

# In[2]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[3]:


df.shape


# ### Convert to date time object
# 

# In[4]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[5]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[6]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[7]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[8]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[9]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[10]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[11]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[12]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[13]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[14]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[15]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[16]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[17]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[18]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[19]:


from sklearn.neighbors import KNeighborsClassifier


# In[20]:


from sklearn.model_selection import train_test_split
#mach k for Kneigh
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for n in range(1,10):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    print(f"{n} : ",neigh.score(X_test, y_test))


# In[21]:


#I should k=7 form r score
k = 7
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X,y)
neigh


# # Decision Tree
# 

# In[22]:


from sklearn.tree import DecisionTreeClassifier

DescisTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)


# In[23]:


DescisTree.fit(X,y)


# In[ ]:





# # Support Vector Machine
# 

# In[24]:


from sklearn import svm


# In[25]:


m_svm = svm.SVC(kernel='rbf')
m_svm.fit(X, y) 


# # Logistic Regression
# 

# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


lg=LogisticRegression(C=0.01, solver='liblinear').fit(X,y)


# In[ ]:





# # Model Evaluation using Test set
# 

# In[28]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:
# 

# ### Load Test set for evaluation
# 

# In[29]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[30]:


#Clean DATA
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.head()


# In[31]:


#X,Y Selection
Feature_test= test_df[['Principal','terms','age','Gender','weekend']]
Feature_test = pd.concat([Feature_test,pd.get_dummies(test_df['education'])], axis=1)
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
Feature_test.tail()  


# In[32]:


Feature


# In[33]:


#X normalize
X_test = Feature_test
X_test= preprocessing.StandardScaler().fit(X_test).transform(X_test)
X[0:5]


# In[34]:


#Y normalize
test_df['loan_status'].replace(to_replace=['COLLECTION','PAIDOFF'], value=[0,1],inplace=True)
y_test = test_df['loan_status'].values
y_test


# In[80]:


#Predict 
yh_neigh=neigh.predict(X_test)
yh_neigh=np.where(yh_neigh=='PAIDOFF', 1, yh_neigh)
yh_neigh=np.where(yh_neigh=='COLLECTION', 0, yh_neigh)
yh_neigh = yh_neigh.astype('int64')


# In[81]:


yh_tree = DescisTree.predict(X_test)
yh_tree=np.where(yh_tree=='PAIDOFF', 1, yh_tree)
yh_tree=np.where(yh_tree=='COLLECTION', 0, yh_tree)
yh_tree = yh_tree.astype('int64')


# In[87]:


yhat_m_svm=m_svm.predict(X_test)
yhat_m_svm=np.where(yhat_m_svm=='PAIDOFF', 1, yhat_m_svm)
yhat_m_svm=np.where(yhat_m_svm=='COLLECTION', 0, yhat_m_svm)
yhat_m_svm= yhat_m_svm.astype('int64')


# In[89]:


yh_lg=lg.predict(X_test)
yhprob_lg = lg.predict_proba(X_test)
yh_lg=np.where(yh_lg=='PAIDOFF', 1, yh_lg)
yh_lg=np.where(yh_lg=='COLLECTION', 0, yh_lg)
yh_lg= yh_lg.astype('int64')


# In[84]:


y_test


# In[85]:


#Evaluation using Test set
print('jaccard_score_Neigh :',jaccard_score(y_test,yh_neigh))
print('f1_score_Neigh :', f1_score(y_test,yh_neigh, average='weighted'))


# In[86]:


print('jaccard_score_Tree :',jaccard_score(y_test,yh_tree))
print('f1_score_Tree :', f1_score(y_test,yh_tree, average='weighted') )


# In[88]:


print('jaccard_score_SVM :',jaccard_score(y_test,yhat_m_svm))
print('f1_score_SVM :', f1_score(y_test,yhat_m_svm, average='weighted'))


# In[90]:


print('jaccard_score_Logistics :',jaccard_score(y_test,yh_lg))
print('f1_score_Logistics :', f1_score(y_test,yh_lg, average='weighted'))
print('log_loss_Logistics : ', log_loss(y_test, yhprob_lg))


# In[91]:


ls=[['KNN',jaccard_score(y_test,yh_neigh),f1_score(y_test,yh_neigh, average='weighted'),'Na'],
    ['Decistion Tree',jaccard_score(y_test,yh_tree),f1_score(y_test,yh_neigh, average='weighted'),'Na'],
    ['SVM',jaccard_score(y_test,yhat_m_svm),f1_score(y_test,yhat_m_svm, average='weighted'),'Na'],
    ['Logistics',jaccard_score(y_test,yh_lg),f1_score(y_test,yh_lg, average='weighted'),log_loss(y_test, yhprob_lg)]]

ls1 = pd.DataFrame(ls,columns =['Model','Jaccard', 'F1-score','Logloss'])
ls1


# In[ ]:





# # Report
# 
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# | Algorithm          | Jaccard | F1-score | LogLoss |
# | ------------------ | ------- | -------- | ------- |
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |
# 

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2021-01-01">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                                                             |
# | ----------------- | ------- | ------------- | ------------------------------------------------------------------------------ |
# | 2020-10-27        | 2.1     | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |
# | 2020-08-27        | 2.0     | Malika Singla | Added lab to GitLab                                                            |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <p>
# 

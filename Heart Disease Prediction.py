#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[27]:


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv("heart_disease_data.csv")


# In[4]:


# print first 5 rows of the dataset
heart_data.head()


# In[5]:


# print last 5 rows of the dataset
heart_data.tail()


# In[6]:


# number of rows and columns in the dataset
heart_data.shape


# In[7]:


# getting some info about the data
heart_data.info()


# In[8]:


# checking for missing values
heart_data.isnull().sum()


# In[9]:


# statistical measures about the data
heart_data.describe()


# In[32]:


plt.figure(figsize=(10,7))
sns.heatmap(heart_data.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[10]:


# checking the distribution of Target Variable
heart_data['target'].value_counts()


# In[11]:


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


# In[12]:


print(X)


# In[13]:


print(Y)


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[15]:


print(X.shape, X_train.shape, X_test.shape)


# In[16]:


model = LogisticRegression()


# In[17]:


# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)


# In[18]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[34]:


print('Accuracy on Training data : ', training_data_accuracy*100,'%')


# In[20]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[35]:


print('Accuracy on Test data : ', test_data_accuracy*100,'%')


# In[25]:


input_data = (100,1,0,140,268,0,0,160,0,4.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


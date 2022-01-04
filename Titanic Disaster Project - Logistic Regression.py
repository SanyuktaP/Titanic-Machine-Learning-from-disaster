#!/usr/bin/env python
# coding: utf-8

# # Titanic - Machine Learning from Disaster
# Start here! Predict survival on the Titanic and get familiar with ML basics

# In[1]:


import pandas as pd


# Here we read the train.csv file in which the detailed data of Titanic Survival is Recorded

# In[2]:


data=pd.read_csv("train.csv")


# In[3]:


data.shape


# So ,there are details of 891 people.
# Below command shows us first 5 row details

# In[4]:


data.head()


# ## PREPROCESSING DATA

# In[5]:


data=data.drop(["Name","Ticket","Cabin"],axis=1)


# In[6]:


data.head()


# In[7]:


data.shape


# #### Label encoding

# In[8]:


from sklearn.preprocessing import LabelEncoder 


# In[9]:


le=LabelEncoder()


# In[10]:


data.iloc[:,2]=le.fit_transform(data.iloc[:,2])


# In[11]:


data.iloc[:,7]=le.fit_transform(data.iloc[:,7])


# In[12]:


data.head()


# #### Imputation

# In[13]:


from sklearn.impute import SimpleImputer
import numpy as np


# In[14]:


si=SimpleImputer(strategy="mean",missing_values=np.NaN)


# In[15]:


data.iloc[:,3:4]=si.fit_transform(data.iloc[:,3:4])


# In[16]:


data.head()


# In[17]:


data.isnull().sum()


# In[18]:


data.head()


# #### Here  'X' is Independent variable, and 'y' is Dependent Variable. There can be only one dependent variable

# In[19]:


X=data.iloc[:,[0,1,2,3,4,5,6,8]]


# In[20]:


y=data.iloc[:,-1]


# In[21]:


X


# In[22]:


y


# ### Training and Splitting of data
# 

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.10)


# In[25]:


X_train.head(2)


# In[26]:


X_test.head(2)


# In[27]:


y_train.head(2)


# In[28]:


y_test.head(2)


# In[29]:


print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)


# ### Scaling using 'scale' method

# In[30]:


from sklearn.preprocessing import scale


# In[31]:


X_train=scale(X_train)


# In[32]:


X_test=scale(X_test)


# In[33]:


X_train


# In[34]:


X_test


# ##  LOGISTIC REGRESSION

# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


lr=LogisticRegression()


# In[37]:


y_train


# In[38]:


lr.fit(X_train,y_train)


# Below we are passing the X_test data for prediction , whose values is known to us in y_test. But the Prediction of the machine for the passed X_test data is stored in y_pred .

# In[39]:


y_pred=lr.predict(X_test)


# ### Now we create a data frame for the Predicted values by the Machine and the actual values of it.

# In[40]:


df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})


# In[41]:


df


# ### Below we check the accuracy of the predictions

# In[42]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,plot_confusion_matrix


# In[97]:


print(confusion_matrix(y_test,y_pred))


# In[100]:


plot_confusion_matrix(lr,X_test,y_test)


# In[44]:


print(classification_report(y_test,y_pred))


# In[45]:


accuracy_score(y_test,y_pred)*100


# ## Therefore machine is now capable of predicting the survival on titanic .

# ### 

# ### Now we are checking the machine predictions on the 'test.csv' data. Its actual answer is saved in 'gender_submission.csv' file.
# We have already trained machine to predictions of survival on Titanic.

# # 

# We have to preprocess the test.csv file

# In[46]:


test_data=pd.read_csv("test.csv")


# In[47]:


test_data.head()


# In[48]:


test_data=test_data.drop(["Name","Ticket","Cabin"],axis=1)


# ##### Here le is the object of LabelEncoder class we had created

# In[49]:


test_data.iloc[:,2]=le.fit_transform(test_data.iloc[:,2])


# In[50]:


test_data.iloc[:,7]=le.fit_transform(test_data.iloc[:,7])


# In[51]:


test_data.head()


# In[52]:


test_data.isnull().sum()


# #### Imputation 

# In[53]:


from sklearn.impute import SimpleImputer
import numpy as np


# In[54]:


si=SimpleImputer(strategy="mean",missing_values=np.NaN)


# In[55]:


test_data.iloc[:,3:4]=si.fit_transform(test_data.iloc[:,3:4])
test_data.iloc[:,6:7]=si.fit_transform(test_data.iloc[:,6:7])


# In[56]:


test_data.isnull().sum()


# In[57]:


test_data.shape


# In[58]:


test_data=scale(test_data)


# In[59]:


test_data


# # Prediction
# Prediction of machine on test_data is stored in pred .

# In[60]:


pred=lr.predict(test_data)


# In[61]:


pred


# #### As we know 'gender_submission.csv' stores answer for 'test.csv' , we have to access that file in order to check how much correct predictions are made. 

# In[62]:


result=pd.read_csv("gender_submission.csv")


# In[63]:


result=result.drop("PassengerId",axis=1)


# In[64]:


result


# In[85]:


result.shape


# The result must be a 1D array.

# In[86]:


result=np.array(result)


# In[87]:


result


# In[88]:


result=result.flatten()             #for converting 2D array to 1D array flatten() is used


# In[89]:


result


# In[90]:


df=pd.DataFrame({"Prediction":pred,"Actual":result})


# In[91]:


df


# In[92]:


print(classification_report(pred,result))


# In[93]:


print(confusion_matrix(pred,result))


# In[95]:


accuracy_score(pred,ans)*100


# # Therefore, machine is well capable of now predicting survival on Titanic.

# In[ ]:





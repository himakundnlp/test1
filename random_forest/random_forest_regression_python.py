#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


# In[5]:


data=pd.read_csv("C:\\Users\\Gowri Garu\\Desktop\\All_datasets\\Datasets_new\\insurance\\insurance.csv")


# In[6]:


data.head(5)


# In[7]:


data.describe()


# In[8]:


P1 = go.Scatter(x=data.age, y=data.bmi, mode="markers")
P2 = [P1]


# In[9]:


plotly.offline.init_notebook_mode(connected=True)


# In[10]:


plotly.offline.iplot(P2)


# In[11]:


P2 = go.Scatter(x=data.bmi, y=data.charges, mode="markers")


# In[12]:


P2=[P2]


# In[13]:


plotly.offline.iplot(P2)


# In[14]:


BAR1 = sum(data.sex == 'male')
BAR2 = sum(data.sex == 'female')
BAR = [BAR2, BAR1]


# In[15]:


B1 = [
    go.Bar(
        x=[data.sex.unique()[0], data.sex.unique()[1]],
        y=[data.sex.value_counts()[0],
           data.sex.value_counts()[1]])
]


# In[16]:


plotly.offline.iplot(B1)


# In[17]:


B2 = go.Bar(
    x=[data.smoker.unique()[0],
       data.smoker.unique()[1]],
    y=[data.smoker.value_counts()[0],
       data.smoker.value_counts()[1]])
B2 = [B2]


# In[18]:


plotly.offline.iplot(B2)


# In[19]:


H1 = [go.Histogram(x=data.bmi)]


# In[20]:


plotly.offline.iplot(H1)


# In[21]:


data1=pd.get_dummies(data,columns=["sex","smoker","region"])
data1.head(5)


# In[22]:


data1.shape


# In[23]:


X1 = data1.iloc[:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
Y1 = data1.iloc[:, 3]


# In[24]:


print(X1.head(5))


# In[25]:


print(Y1.head(5))
      


# In[47]:


X_train,X_test,Y_train,Y_test = train_test_split(X1,Y1,test_size=0.3)


# In[53]:


X_train.head(3)


# In[56]:


randmod=RandomForestRegressor(n_estimators=500)
randmod.fit(X_train,Y_train)
pred = randmod.predict(X_test)


# In[57]:


pred[1:10]


# In[70]:


print("Mean Absolute Value is:", round(mean_absolute_error(pred, Y_test), 2),"\n")
print("Mean Squared  Value is:", round(mean_squared_error(pred, Y_test), 2),"\n")
print("R Squared Value is :", round(r2_score(pred, Y_test),2))


# ### Model End

#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[42]:


data = pd.read_csv("AXISBANK.csv")
df = pd.DataFrame(data) 
x = df['Volume'].to_numpy(dtype=float)
y = df['Turnover'].to_numpy(dtype=float)
x_scale = preprocessing.scale(x)
y_scale = preprocessing.scale(y)
X_train,X_test,y_train,y_test=train_test_split(x_scale.reshape(-1,1),y_scale.reshape(-1,1),test_size=0.3,random_state=42)
# print(X_train)
rr = Ridge(alpha=50)
rr.fit(X_train,y_train)
Ridge_train_score = rr.score(X_train,y_train)
Ridge_test_score = rr.score(X_test, y_test)
print(Ridge_train_score)
print(Ridge_test_score)
plt.plot(rr.coef_,alpha=50,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 50$',zorder=7)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.scatter(X_test,y_test)
plt.legend(fontsize=13,loc=4)
plt.show()


# In[ ]:





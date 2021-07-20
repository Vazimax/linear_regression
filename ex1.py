#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\aboub\OneDrive\Desktop\ML_class\ex1\ex1data1.txt"
data = pd.read_csv(path,names=['Population','Profit'])

print('data = \n',data.head(10),'\n')
print('data.describe = \n',data.describe())

data.plot(kind="scatter",x="Population",y="Profit",figsize=(5,5))


# In[ ]:





# In[2]:


data.plot(kind="scatter",x="Population",y="Profit",figsize=(5,5))


# In[ ]:





# In[ ]:





# In[3]:


data.insert(0,'ONES',1)


# In[ ]:





# In[4]:


print('new data : \n',data.head(10))


# cols = data.shape[1]

# In[5]:


cols = data.shape[1]


# In[6]:


print(cols)


# In[7]:


x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]


# In[8]:


x = np.matrix(x.values)
y = np.matrix(y.values)


# In[ ]:





# In[9]:


theta = np.matrix(np.array([0,0]))


# In[10]:


# Cost function :
def cost_function(x,y,theta):
    hypothesis = np.power(((x * theta.T)-y),2)
    print(hypothesis)
    m = len(x)
    print(f'm = {m} \n ############################################## ')
    
    return np.sum(hypothesis) / (2 * m)


# In[11]:


print('Cost function is :',cost_function(x,y,theta))


# In[12]:


def gradient_descent(x,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (x * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error,x[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(x))  * np.sum(term))
            
        theta = temp 
        cost[i] = cost_function(x,y,theta)
        
    return theta,cost


# In[13]:


alpha = 0.01
iters = 1000


# In[14]:


theta , cost = gradient_descent(x,y,theta,alpha,iters)


# In[15]:


print(cost_function(x,y,theta))


# In[16]:


x = np.linspace(data.Population.min(),data.Population.max(),100)


# In[17]:


f = theta[0,0] + (x*theta[0,1])


# In[18]:


print(f)


# In[19]:


# The line :
fig , ax = plt.subplots(figsize=(5,5))
ax.plot(x,f,'r',label="Prediction")
ax.scatter(data.Population,data.Profit,label="training data")
ax.legend(loc=2)
ax.set_title('Predict the profit throughout population')
ax.set_xlabel('Population')
ax.set_ylabel('Profit')


# In[21]:


# Errors Graph :
fig ,ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters),cost,'r')
ax.set_title('Errors Graph')
ax.set_xlabel('iterations')
ax.set_ylabel('Cost')


# In[ ]:





# In[ ]:





# In[ ]:





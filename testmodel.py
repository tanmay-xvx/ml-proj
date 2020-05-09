#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle


# In[36]:


data=pd.read_csv('FOODYLYTICSDATASET.csv')


# In[37]:


data.head()


# In[38]:


predict="Wastage"
X = np.array(data.drop([predict], 1))
y= np.array(data[predict])


# In[46]:


def convert_to_int(word):
    word_dict = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    return word_dict[word]

X['Day'] = X['Day'].apply(lambda X : convert_to_int(X['Day']))


# In[39]:


plt.scatter(data['Menu Rating'],data['Wastage'])
plt.title('Menu Rating vs Wastage', fontsize=14)
plt.xlabel('Menu Rating', fontsize=14)
plt.ylabel('Wastage', fontsize=14)


# In[40]:


plt.scatter(data['Amount Of Food Cooked'],data['Wastage'])
plt.title('Amount Of Food Cooked vs Wastage', fontsize=14)
plt.xlabel('Amount Of Food Cooked', fontsize=14)
plt.ylabel('Wastage', fontsize=14)


# In[41]:


plt.scatter(data['Weekend'],data['Wastage'])
plt.title('Weekend vs Wastage', fontsize=14)
plt.xlabel('Weekend', fontsize=14)
plt.ylabel('Wastage', fontsize=14)


# In[ ]:


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)


# In[ ]:


#comparing with actual results
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

data=pd.read_csv('FOODYLYTICSDATASET.csv')

predict="Wastage"
X = np.array(data.drop([predict], 1))
y= np.array(data[predict])
df2=pd.DataFrame(X)

def convert_to_int(word):
    word_dict = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
    return word_dict[word]
X_temp=[]
for x in X[:,1]:
    X_temp.append(convert_to_int(x))
# print(X_temp)
X[:,1]=X_temp

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X[:,1:], y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

pickle.dump(linear,open('model.pkl','wb'))

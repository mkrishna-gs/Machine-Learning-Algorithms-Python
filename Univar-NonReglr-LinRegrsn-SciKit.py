"This is a python code to do perform linear regression using the sci-kit learn library in python"
import pandas as pd
import numpy as np

#### Import the data to be trained and tested using the linear regression model from "sci-kit learn".
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#### Giving names to the dependent and independent variables extracted from the CSV files.
x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

#### Convert the imported data into array.
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

#### Import LinearRegression from "sci-kit learn" 
#### and R2 square to estimate the goodness of the fit. 
#### Doesn't take "biasing" into account.
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))


import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color='red',label='GT')
plt.plot(x_test,y_pred,color='black',label = 'pred')
plt.legend() 
plt.show()

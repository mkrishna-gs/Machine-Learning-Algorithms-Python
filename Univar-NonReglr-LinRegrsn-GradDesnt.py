"This is a python code to do perform linear regression using the non-regularized Gradient descent algorithm"
import pandas as pd
import numpy as np

#### Import the data to be trained and tested using the linear regression model.
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

#### Convert the row to column.
x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)
x_train = np.c_[np.ones(x_train.shape[0]), x_train]
x_test = np.c_[np.ones(x_test.shape[0]), x_test]

#### Train the data using Gradient descent method.
n = y_train.size
alpha = 0.0001
iters = 1000
np.random.seed(1)
#### Randomny set the theta values intially.
theta = np.random.rand(2)
past_cost  = []
past_theta = [theta]

epochs = 0
while(epochs < iters):
    y = np.dot(x_train,theta)
    error = y - y_train
    cost = 1/(2*n)*np.dot(error.T, error)
    past_cost.append(cost)
    theta = theta - (alpha * (1/n) * np.dot(x_train.T, error))
    past_theta.append(theta)
    epochs += 1

#### R2 square value to determine the goodness of the fit.
from sklearn.metrics import r2_score
y_prediction = np.dot(x_test,theta)
print('R2 Score:',r2_score(y_test,y_prediction))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.scatter(x_test[:,1],y_test,color='red',label='GT')
plt.plot(x_test[:,1],y_prediction,color='black',label = 'pred')
plt.legend()
plt.show()

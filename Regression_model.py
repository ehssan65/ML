# Importing necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=42)

linmod = LinearRegression()
slon = linmod.fit(x_train,y_train)
print('Score train:',slon.score(x_train,y_train))

pred = slon.predict(x_test)
print('Mean ERROR: ',np.sqrt(metrics.mean_squared_error(y_test,pred)))
print('Absolute ERROR: ',metrics.mean_absolute_error(y_test,pred))
print('R Scuard score: ',metrics.r2_score(y_test,pred))

linsp = np.linspace(1,x_test.shape[0],x_test.shape[0])
res = pred - y_test
print('xx : ',linsp,'x_test.shape[0]: ',x_test.shape[0])
plt.scatter(linsp,res)

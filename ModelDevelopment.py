import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path of data

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df=pd.read_csv(path)
df.head()
# Lets load the modules for linear regression
from sklearn.linear_model import LinearRegression
# creating linear regression obj
lm=LinearRegression()
lm
# How could Highway-mpg help us predict car price?
X=df[['highway-mpg']]
Y=df['price']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]
# Value of intercept
lm.intercept_

# Value of slope
lm.coef_
Z=df[['horsepower','curb-weight','engine-size','highway-mpg']]
lm.fit(Z,df['price'])
# INtercept
lm.intercept_
# SLope
lm.coef_

import seaborn as sns
%matplotlib inline

# Let's visualize highway-mpg as potential predictor variable of price:
width=12
height=10
plt.figure(figsize=(width,height))
sns.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,)
plt.figure(figsize=(width,height))
sns.regplot(x="peak-rpm",y="price",data=df)
plt.ylim(0,)

df[["peak-rpm","highway-mpg","price"]].corr

width=12
height=10
plt.figure(figsize=(width,height))
sns.residplot(df['highway-mpg'],df['price'])
plt.show()

#  Multiple Linear Regression
Yhat = lm.predict(Z)
plt.figure(figsize=(width,height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()
# Polynomial Regression and Pipelines

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x=df['highway-mpg']
y=df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)
print(p)
# lets plot the function

PlotPolly(p,X,Y,'highway-npg')

# Write your code below and press Shift+Enter to execute
f1=np.polyfit(x,y,11)
p=np.poly1d(f1)
print(p)
PlotPolly(p,x,y,'highway-mpg')
from sklearn.preprocessing import PolynomialFeatures

pr=PolynomialFeatures(degree=2)

prZ_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape
#  Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]

Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
# predict the output

Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)
#  Model 2: Multiple Linear Regression
lm.fit(Z,df['price'])
# Find the R^2
print("the R-square is: ",lm.score(Z,df['price']))
Y_predict_multifit = lm.predict(Z)
# we compare the predicted results with the actual results
print("The mean square error of price and predicted value  using multifit is: ", mean_squared_error(df['price'], Y_predict_multifit))
# Model 3: Polynomial Fit

from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
# MSE

#  Part 5: Prediction and Decision Making


import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline
# Create a new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)

lm.fit(X,Y)
lm
# produce a prediction
yhat=lm.predict(new_input)
yhat[0:5]
# Plot the data
plt.plot(new_input,yhat)
plt.show()


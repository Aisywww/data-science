import pandas as pd 
import numpy as mp
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/Lenovo/Downloads/1.02.+Multiple+linear+regression.csv")
print(data.head(),data.describe())

reg = LinearRegression()

# Define Our Dependet and Indipendent var and confirm count
y = data['GPA']
x1 = data[['SAT','Rand 1,2,3']]
print(f'\ny = {y.shape}\nx1 = {x1.shape}')

# want to find adjusted r square

reg.fit(x1,y)
print(f'\nR square : {reg.score(x1,y)}\nCoeff : {reg.coef_}\nConst : {reg.intercept_}')

# adjst R square fomula = 1 - (1-r**2) * (n-1)/(n-p-1)

r2 = reg.score(x1,y)
n = 84
p = 2
print(f'\nAdjst R square :{1 - (1-r2) * (n-1)/(n-p-1)}')
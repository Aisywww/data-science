import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
 
sns.set()
from sklearn.linear_model import LinearRegression

data = pd.read_csv('C:/Users/Lenovo/Downloads/1.01.+Simple+linear+regression.csv')
print(data.head())

#dependent
x = data['SAT']
#indipendent
y = data["GPA"]
print(x)
#convert 1D x to 2D
x_mat = x.values.reshape(-1,1)

reg = LinearRegression()
res = reg.fit(x_mat,y)
print(reg.score(x_mat,y))
print(reg.coef_)
print(reg.intercept_)

#making table
result = reg.predict([[1740]])#<== must two list convert to 2D
new_data = pd.DataFrame(data = [1740,1620],columns=['SAT'])
new_data.rename(index = {0:'KAMIL',1:"sam"},inplace = True)
new_data["predicted GPA"] = reg.predict(new_data)
print(new_data)

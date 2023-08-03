import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

new_data = pd.read_csv('C:/Users/Lenovo/Downloads/1.02.+Multiple+linear+regression.csv')
print(new_data.describe(),'\n\n',new_data.head())
print('\n',sns.pairplot(new_data))

y = new_data['GPA']
x = new_data[['SAT','Rand 1,2,3']]

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scale.fit(x)

x_scaled = scale.transform(x)
print('\n',x_scaled)

reg = LinearRegression()
sum = reg.fit(x,y)
print('\n',reg.coef_,'\n')
print(reg.intercept_)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
new_data = scaler.fit_transform(new_data)

new_summary = pd.DataFrame(new_data,columns=['SAT','Rand 1,2,3','GPA'])
print(new_summary.describe())

data_summary =pd.DataFrame([['Bias'],['SAT'],['Rand 1,2,3',]],columns= ['Features'])
data_summary['weights'] = reg.intercept_,reg.coef_[0],reg.coef_[1]
print(data_summary)

import pandas as pd 
import numpy as np 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
sns.set()

# calling file 
raw_data = pd.read_csv('C:/Users/Lenovo/Downloads/1.04.+Real-life+example.csv')
print(raw_data.head(),'\n\n',raw_data.describe(include='all'))

# drop model collumn since its not quite usefull to model(use drop function)

data = raw_data.drop(['Model'],axis=1)
print(data.isnull().sum)  # false(no missing value) = 0 , true(missing value) = 1

# Since <5% of data missing its safe to removed all observervation 

data_no_mv = data.dropna(axis = 0)
print(data_no_mv.describe(include='all'))

# check each numerical data using graph (seaborn)

price = sns.distplot(data_no_mv['Price'])
print(plt.show())

# dealing with outliers
q = data_no_mv['Price'].quantile(0.99)
data1 = data_no_mv[data_no_mv['Price'] < q]
print(data1.describe())

q = data1['Mileage'].quantile(0.99)
data2 = data1[data1['Mileage']<q]

data3 = data2[data2['EngineV']<6.5]
sns.distplot(data3['EngineV'])
print(plt.show())

q = data3['Year'].quantile(0.01)
data4 = data3[data3['Year']>q]
sns.distplot(data4['Year'])
plt.show()

data_cleaned = data4.reset_index(drop=True)
print(data_cleaned.describe(include='all'))


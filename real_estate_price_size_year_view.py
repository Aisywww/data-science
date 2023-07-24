import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
 
sns.set()

raw_data = pd.read_csv("C:/Users/Lenovo/Downloads/real_estate_price_size_year_view.csv")
print(raw_data)
data = raw_data.copy()
data['view'] = data["view"].map({"Sea view":1,"No sea view":0})
print(data.describe())

#Regression
y = data['price']
x1 = data[['size','year','view']]
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())

plt.scatter(data["size"],y,c = data["view"],cmap = "RdYlGn_r")
yhat_no = 7.748e+04 + 218.7521*data["size"] +  5.756e+04*0
yhat_yes = 7.748e+04 + 218.7521*data["size"] +  5.756e+04*1
fig = plt.plot(data["size"],yhat_yes, lw = 2, c = "red")
fig = plt.plot(data["size"],yhat_no, lw = 2, c = "blue")
plt.xlabel("Size",fontsize = 20)
plt.ylabel("price",fontsize = 20)
print(plt.show())

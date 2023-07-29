import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

# import files
data = pd.read_csv('C:/Users/Lenovo/Downloads/real_estate_price_size.csv')
x = data['size']
y = data['price']
print(data.head())

#convert into 2d
x_matriks = x.values.reshape(-1,1)
reg = LinearRegression()
reg.fit(x_matriks,y)

# main/useful info
print(f'\nR square = {reg.score(x_matriks,y)}\nSize coeff = {reg.coef_}\nCOnst = {reg.intercept_}\n')

#making prediction + table(database)

new_house = reg.predict([[750]])
print(f'price for 750 sqft is{new_house}\n')
new_data = pd.DataFrame(data = [750,1000],columns= ['Size'])
new_data.rename(index = {0:'lot 2',1:'lot 3'},inplace=True)
new_data['predicted price'] = reg.predict(new_data)
print(new_data)








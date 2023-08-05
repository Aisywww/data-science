import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('C:/Users/Lenovo/Downloads/real_estate_price_size_year.csv')
print(data.describe(),'\n\n',data.head(),'\n')

x = data[['size','year']]
y = data['price']

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale.fit(x)
x_scale = scale.transform(x)

reg = LinearRegression()
reg.fit(x_scale,y)
r2 = reg.score(x_scale,y)
print(r2,'\n')

# adjst r square fomula : 1 - (1-r2) * (n-1)/(n-p-1)

def adjst(x,y):
     n = x.shape[0]
     p = x.shape[1]
     Ar2 = 1 - (1-r2) * (n-1)/(n-p-1)
     return Ar2,"\n"

print(adjst(x_scale,y))

from sklearn.feature_selection import f_regression

p_value = f_regression(x_scale,y)[1].round(3)
print(p_value,'\n') # year feature is not significant to the regression/analysis

# prediction for house size: 750 and year: 2009

pred_data = [[750,2009]]
scale = reg.predict(scale.transform(pred_data))
print(f'prediction: {scale}\n')

# making tables 

summary = pd.DataFrame(data = [['size'],['year']],columns=['Feature'])
summary['Coefficient'] = reg.coef_
summary['p_values'] = p_value
print(summary)

#since year is not significant it can be removed and be replaced with other better feature/variable
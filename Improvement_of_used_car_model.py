import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

rawdata = pd.read_csv('C:/Users/Lenovo/Downloads/1.04.+Real-life+example.csv')
print(rawdata.head(),'\n\n',rawdata.describe(include='all'))

data = rawdata.drop(columns=['Model'],axis=1)
print('\n',data.isnull().sum())

data = data.dropna(axis = 0)
print(data.describe())

price = sns.distplot(data['Price'])
plt.show()
mileage = sns.distplot(data['Mileage'])
plt.show()
EngineV = sns.distplot(data['EngineV'])
plt.show()
year = sns.distplot(data['Year'])
plt.show()

q = data['Price'].quantile(0.98)
data = data[data['Price']<q]
print(data.describe())
sns.distplot(data['Price'])
plt.show()

q = data['Mileage'].quantile(0.98)
data1 = data[data['Mileage']<q]
sns.distplot(data1['Mileage'])
plt.show()

q = 6.5
data2 = data1[data1['EngineV']<q]
sns.distplot(data2["EngineV"])
plt.show()

q = data2['Year'].quantile(0.02)
data3 = data2[data2['Year']>q]
sns.distplot(data3['Year'])
plt.show()

data_cleaned = data3.reset_index(drop=True)
print(data_cleaned.describe(include= 'all'))

log_price = np.log(data_cleaned['Price'])
data_cleaned["Log Price"] = log_price

f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True)
ax1.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax1.set_title('Price Over Mileage')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price over EngineV')
ax3.scatter(data_cleaned['Year'],data_cleaned["Price"])
ax3.set_title('Price Over Year')
plt.show()


#checking each indipendent using faeture selection(p-value) vif

from sklearn.feature_selection import f_regression

p_value_m = f_regression(data_cleaned['Mileage'].values.reshape(-1,1),data_cleaned['Price'])[1].round(2)
p_value_e = f_regression(data_cleaned['EngineV'].values.reshape(-1,1),data_cleaned['Price'])[1].round(2)
p_value_y = f_regression(data_cleaned['Year'].values.reshape(-1,1),data_cleaned['Price'])[1].round(2)
print(f'\nmilage p_value: {p_value_m}\nEngineV p_value: {p_value_e}\nYear p_value: {p_value_y}')

p_value_tables = pd.DataFrame([['Mileage'],['EngineV'],['Year']],columns=['Feature'])
p_value_tables['P Values'] =  p_value_m,p_value_e,p_value_y
print(p_value_tables)

#vif checking
from statsmodels.stats.outliers_influence import variance_inflation_factor

var = data_cleaned[['Mileage','EngineV','Year']]
vif = pd.DataFrame()
vif['Vif'] = [variance_inflation_factor(var.values,i) for i in range(var.shape[1])]    #model rules
vif['features'] = var.columns                                                          #1.Linearity   4.No autocorellation
print('\n',vif)                                                                        #2.No Endogeneity  5.No multicollinearity
                                                                                       #3.Normality and homoscedasticity 

data_cleaned = data_cleaned.drop(['EngineV'],axis = 1)
data_with_dummies = pd.get_dummies(data_cleaned,drop_first = True)

print(data_with_dummies.columns.values)

cols = [ 'Log Price', 'Mileage', 'Year' ,'Brand_BMW', 'Brand_Mercedes-Benz'
 ,'Brand_Mitsubishi', 'Brand_Renault', 'Brand_Toyota' ,'Brand_Volkswagen'
 ,'Body_hatch' ,'Body_other', 'Body_sedan', 'Body_vagon', 'Body_van'
 ,'Engine Type_Gas' ,'Engine Type_Other' ,'Engine Type_Petrol'
 ,'Registration_yes']

data_preprocess = data_with_dummies[cols]
print('\n\n\n',data_preprocess.head())

targets = data_preprocess['Log Price']
inputs = data_preprocess.drop(["Log Price"],axis = 1)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scale.fit(inputs)

inputs_scaled = scale.transform(inputs)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,targets,test_size=0.2,random_state=44) # can change to get better result

reg = LinearRegression()
reg.fit(x_train,y_train)

yhat = reg.predict(x_train)
plt.scatter(y_train,yhat,alpha=0.2)
plt.xlabel('Y train', size = 18)
plt.ylabel('Y hat',size = 18)
plt.show()

sns.displot(y_train-yhat)
plt.title('Residuals',size = 18)
plt.show()

r2 = reg.score(x_train,y_train)
print(r2)

inputs_coeff = reg.coef_
inputs_constant = reg.intercept_
data_summ = pd.DataFrame(inputs.columns.values,columns = ['Features'])
data_summ['weight'] = inputs_coeff
print(data_summ)


#testing our model

y_hat_test = reg.predict(x_test)

plt.scatter(y_hat_test,y_test)
plt.xlabel('Y train',size = 18)
plt.ylabel('Y hat',size = 18)
plt.show()

y_test = y_test.reset_index(drop = True)

df_pf = pd.DataFrame(np.exp(y_hat_test),columns = ['Prediction'])
df_pf['Target'] = np.exp(y_test)
print(df_pf.head())

df_pf["Residual"] = df_pf['Target'] - df_pf['Prediction']
df_pf['Diffrence%'] = np.absolute(df_pf["Residual"]/df_pf['Prediction']*100)
pd.options.display.max_rows = 999
pd.set_option("display.float.format",lambda x:'%.2f' % x ) #or use this:  pd.options.display.float_format = '{:.2f}'.format

print(df_pf.sort_values(by = ["Diffrence%"]))
print(df_pf.describe())

r2_test = reg.score(x_test,y_test)
r2_train = reg.score(x_train,y_train)

def adjst(x):
     n = x.shape[0]
     p = x.shape[1]
     Ar2 = 1 - (1-r2) * (n-1)/(n-p-1)
     return Ar2,"\n"
print(adjst(x_train))
print(r2_test,r2_train)









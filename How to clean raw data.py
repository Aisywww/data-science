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
sns.distplot(data2['Mileage'])

data3 = data2[data2['EngineV']<6.5]
sns.distplot(data3['EngineV'])
#print(plt.show())

q = data3['Year'].quantile(0.01)
data4 = data3[data3['Year']>q]
sns.distplot(data4['Year'])
#plt.show()

# reset old index and replaced with new one 
data_cleaned = data4.reset_index(drop=True)
print(data_cleaned.describe(include='all'))

#checkin ols assumption

f , (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True)
ax1.scatter(data_cleaned['Mileage'],data_cleaned["Price"])
ax1.set_title('Price and Mileage')
ax2.scatter(data_cleaned["Year"],data_cleaned['Price'])
ax2.set_title('Price and Year')
ax3.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax3.set_title("Price and EngineV")


#result showh that graph is exponential thus to make regression we need to transform it to linear
#by using log transformation we can transform exponential graph(scatter plot) to linear 

#undergo log transformation
log_price = np.log(data_cleaned['Price'])
data_cleaned['Log Price'] = log_price


f , (ax1,ax2,ax3) = plt.subplots(1,3,sharey = True)
ax1.scatter(data_cleaned['Mileage'],data_cleaned["Log Price"])
ax1.set_title('Price and Mileage')
ax2.scatter(data_cleaned["Year"],data_cleaned['Log Price'])
ax2.set_title('Price and Year')
ax3.scatter(data_cleaned['EngineV'],data_cleaned['Log Price'])
ax3.set_title("Price and EngineV")

plt.show()

# want to check Multicolinearity

from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data_cleaned[['Mileage','EngineV','Year']]
vif = pd.DataFrame()
vif['Vif'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
print('\n',vif)

# since year vif point is too high which > 10 we drop the collumn of year from our table

data_no_multicolinearity = data_cleaned.drop(['Year'],axis = 1)
print(data_no_multicolinearity.head())

# doing dummies to categorical data and drop first collumn to prevent multicollinear/correlated

data_with_dummmies = pd.get_dummies(data_no_multicolinearity,drop_first= True)
print(data_with_dummmies)

#Rearrange Data
print(data_with_dummmies.columns.values)
cols = ['Log Price','Mileage', 'EngineV',  'Brand_BMW', 'Brand_Mercedes-Benz'
 ,'Brand_Mitsubishi' ,'Brand_Renault', 'Brand_Toyota' ,'Brand_Volkswagen'
, 'Body_hatch', 'Body_other' ,'Body_sedan', 'Body_vagon', 'Body_van'
 ,'Engine Type_Gas', 'Engine Type_Other', 'Engine Type_Petrol'
, 'Registration_yes']

data_preprocess = data_with_dummmies[cols]
print('\n\n\n',data_preprocess.head())

# Standardize the data using sklearn

targets = data_preprocess["Log Price"]
inputs = data_preprocess.drop(['Log Price'],axis = 1)

from sklearn.preprocessing import StandardScaler 

scale = StandardScaler()
scale.fit(inputs)

inputs_scaled = scale.transform(inputs)

# Doing Train Test Split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,targets,test_size = 0.2,random_state = 365)



reg = LinearRegression()
reg.fit(x_train,y_train)

yhat = reg.predict(x_train)

plt.scatter(y_train,yhat,alpha=0.2)
plt.xlabel('y_train',size = 10)
plt.ylabel('y_hat',size = 10 )
plt.show()


# checking residuals(detected that predicted price is more than observed this indicate that our model might overestimate the price whem predict)

sns.displot(y_train - yhat)
plt.title("Residuals",size = 18)
plt.show()

# r score 

r2 = reg.score(x_train,y_train)
print(r2)

#weight/coeff for inputs

inputs_coeff = reg.coef_
inputs_intercept =  reg.intercept_
print(f'{inputs_coeff}\n\n{inputs_intercept}')

#making weight table for each variables
database_weight = pd.DataFrame(inputs.columns.values,columns=['Features'])
database_weight['weights'] = inputs_coeff
print(database_weight)

#testing our model

y_hat_test = reg.predict(x_test)

plt.scatter(y_test,y_hat_test,alpha=0.2)
plt.xlabel('y_test',size = 18)
plt.ylabel('y_hat_test',size = 18)
plt.show()

# we see the sctter plot above that for low price we might overpridict since the most data  is quite far from predicted
# reset index due to pandas want to math with original index
y_test = y_test.reset_index(drop = True)

df_performance = pd.DataFrame(np.exp(y_hat_test),columns= ['prediction'])
df_performance['Targets'] = np.exp(y_test)
print(df_performance.head())

# making our final summary table

df_performance['residual'] =df_performance['Targets'] - df_performance['prediction'] 
df_performance['Diffrence%'] = np.absolute(df_performance['residual']/df_performance['Targets']*100)
pd.options.display.max_rows = 999
pd.set_option('display.float.format',lambda x:'%.2f' % x)


print(df_performance.sort_values(by = ['Diffrence%']),'\n\n',df_performance.describe())



import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

raw_data = pd.read_csv('C:/Users/Lenovo/Downloads/Bank_data (1).csv')
print(raw_data.head())

raw_data = raw_data.copy()

raw_data['y'] = raw_data['y'].map({'yes':1,'no':0})
data = raw_data.drop(['Unnamed: 0'],axis = 1)

print(data.head())


y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_logit = sm.Logit(y,x)
reg_result = reg_logit.fit()
print(reg_result.summary())

plt.scatter(x1,y)
plt.xlabel('Duration',size = 18)
plt.ylabel('y',size = 18)
plt.show()

print(reg_result.params)

reg_logit = sm.Logit(y,x)
reg_result = reg_logit.fit()
def f(x,b0,b1):   #Try with other x regression and find the best x values posible
    return np.array(np.exp(b0 + b1*x)/(1 + np.exp(b0 + x*b1)))

f_sorted = np.sort(f(x1,reg_result.params[0],reg_result.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y)
plt.xlabel('duration',size = 18)
plt.ylabel('y',size = 18)
plt.plot(x_sorted,f_sorted,color = 'C8')
plt.show()

print(np.exp(reg_result.params[1])) # yes = 1.005*no

y = data['y']
estimators = data[['interest_rate','credit','march','previous','duration']]

x = sm.add_constant(estimators)
reg_log = sm.Logit(y,estimators)
log_result = reg_log.fit()
print(log_result.summary())


def confusion_matrix(data,actual_values,models):
    
    pred_value = models.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values,pred_value,bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm,accuracy

print(confusion_matrix(estimators,y,log_result)) #real


test_data = pd.read_csv('C:/Users/Lenovo/Downloads/Bank_data_testing.csv')
test_data = test_data.copy()
test_data['y'] = test_data['y'].map({'yes':1,'no':0})
test_data = test_data.drop(['Unnamed: 0'],axis = 1)
print(data.head())


y_test = test_data['y']
x_test = test_data[['interest_rate','credit','march','previous','duration']]

x_all = sm.add_constant(x_test)# for jupyter only

print(confusion_matrix(x_test,y_test,log_result)) #test

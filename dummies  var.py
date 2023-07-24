import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
 
sns.set()

raw_data =pd.read_csv('C:/Users/Lenovo/Downloads/1.03.+Dummies.csv')
data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1,"No": 0})
print(data.describe())

# Regression
y = data['GPA']
x1 = data[['SAT',"Attendance"]]

x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
print(result.summary())

plt.scatter(data['SAT'],y,c = data["Attendance"],cmap = 'RdYlGn_r')
yhat_yes = 0.6439 + 0.0014*data["SAT"] + 0.2226*1
yhat_no = 0.6439 + 0.0014*data["SAT"] + 0.2226*0
yhat =  0.275 + 0.0017*data["SAT"]
fig = plt.plot(data["SAT"],yhat_no, lw = 2, c = 'green')
fig = plt.plot(data["SAT"],yhat_yes, lw = 2, c = 'red')
fig = plt.plot(data["SAT"],yhat, lw = 2, c = 'blue')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
#print(plt.show())

print(x)
new_data = [['const','SAT','Attendance']]
new_data = pd.DataFrame({"const":1,"SAT":[1700,1670],"Attendance":[0,1]})


predictions = result.predict(new_data)
predictionsdf = pd.DataFrame({"Predictions":predictions})
joined = new_data.join(predictionsdf)
joined.rename(index={0:"Bob",1:"Alice"},inplace = True)
print(joined)
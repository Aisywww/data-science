import numpy as np 
import pandas as pd 
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sns.set()

data = pd.read_csv('C:/Users/Lenovo/Downloads/iris_dataset.csv')
print(data)

plt.scatter(data['sepal_length'],data['sepal_width'])
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('iris scatter plot')
plt.show()

x = data.copy()

sepal = x.copy()
kmeans = KMeans(2,random_state = False)
pred = kmeans.fit_predict(sepal)
sepal['Sepal pred'] = pred

plt.scatter(data['sepal_length'],data['sepal_width'],c = sepal['Sepal pred'],cmap = 'rainbow')
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('iris scatter plot')
plt.show()

from sklearn import preprocessing

new = x.copy()
var_scaled = preprocessing.scale(new)


wssc = []
for i in range(1,10):
    kmeans = KMeans(i,random_state=False)
    predict = kmeans.fit(var_scaled)
    wssc.append(predict.inertia_)
    
number_cluster = range(1,10)
plt.plot(number_cluster,wssc)
plt.xlabel('num cluster')
plt.ylabel('wcss')
plt.title('ELBOW MEthOD')
plt.show()

kmeans = KMeans(2,random_state = False)
prednew = kmeans.fit_predict(var_scaled)
new = x.copy()
new['Sepal pred'] = prednew

plt.scatter(new['sepal_length'],new['sepal_width'],c = new['Sepal pred'],cmap = 'rainbow')
plt.xlabel('Length')
plt.ylabel('Width')
plt.title('standardize var cluster')
plt.show()

kmeans3 = KMeans(3,random_state = False)
prednew3 = kmeans3.fit_predict(var_scaled)
new3 = x.copy()
new3['Sepal pred'] = prednew3

plt.scatter(new3['sepal_length'],new3['sepal_width'],c = new3['Sepal pred'],cmap = 'rainbow')

plt.show()
plt.scatter(new3['petal_length'],new3['petal_length'],c = new3['Sepal pred'],cmap = 'rainbow')

kmeans4 = KMeans(4,random_state = False)
prednew4 = kmeans4.fit_predict(var_scaled)
new4 = x.copy()
new4['Sepal pred'] = prednew4

plt.scatter(new4['sepal_length'],new4['sepal_width'],c = new4['Sepal pred'],cmap = 'rainbow')
plt.show()
kmeans5 = KMeans(5,random_state = False)
prednew5 = kmeans5.fit_predict(var_scaled)
new5 = x.copy()
new5['Sepal pred'] = prednew5

plt.scatter(new5['sepal_length'],new5['sepal_width'],c = new5['Sepal pred'],cmap = 'rainbow')
plt.show()

realdata = pd.read_csv('C:/Users/Lenovo/Downloads/iris_with_answers.csv')

print(realdata['species'].unique())

realdata['species'] = realdata['species'].map({'setosa':0, 'versicolor':1 , 'virginica':2})
print(realdata.head())

plt.scatter(realdata['sepal_length'], realdata['sepal_width'], c= realdata['species'], cmap = 'rainbow')
plt.show()
plt.scatter(realdata['petal_length'], realdata['petal_width'], c= realdata['species'], cmap = 'rainbow')
plt.show()

plt.scatter(new3['sepal_length'],new3['sepal_width'],c = new3['Sepal pred'],cmap = 'rainbow')
plt.show()
plt.scatter(new3['petal_length'],new3['petal_length'],c = new3['Sepal pred'],cmap = 'rainbow')
plt.show( )
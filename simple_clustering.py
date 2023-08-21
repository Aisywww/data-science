# simple clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

#C:/Users/Lenovo/Downloads/Countries-exercise.csv
# C:/Users/Lenovo/Downloads/3.01.+Country+clusters.csv
data = pd.read_csv('C:/Users/Lenovo/Downloads/Countries-exercise.csv')
print(data.head())

plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#select features

x = data.iloc[:,1:] #iloc method slices data frame
print(x.head())

#clustering
kmeans = KMeans(7,random_state=False)
identified_cluster = kmeans.fit_predict(x)
print(identified_cluster)

data_with_cluster = data.copy()
data_with_cluster['Cluster'] = identified_cluster
print(data_with_cluster.head())

plt.scatter(data['Longitude'],data['Latitude'], c = data_with_cluster['Cluster'],cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


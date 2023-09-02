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
print('haha',x.head())

#clustering
kmeans = KMeans(4,random_state=False)
identified_cluster = kmeans.fit_predict(x)
print(identified_cluster)

data_with_cluster = data.copy()
data_with_cluster['Cluster'] = identified_cluster
print(data_with_cluster.head())

plt.scatter(data['Longitude'],data['Latitude'], c = data_with_cluster['Cluster'],cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

print(kmeans.inertia_)

wcss = []

for i in range(1,10):
    kmeans = KMeans(i,random_state=False)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

print(wcss)

number_cluster = range(1,10)
plt.plot(number_cluster,wcss)
plt.xlabel('number of cluster',size = 18)
plt.ylabel('wcss',size = 18)
plt.title('Elbow Method')
plt.show()   # we can see that 2,3 best point that not tooo near with zero

kmeans = KMeans(3,random_state=False)
identified_cluster = kmeans.fit_predict(x)
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_cluster

plt.scatter(data['Longitude'],data['Latitude'], c = data_with_clusters['Cluster'],cmap = 'rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

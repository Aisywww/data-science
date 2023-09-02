import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sn
from sklearn.cluster import KMeans

rawdata = pd.read_csv('C:/Users/Lenovo/Downloads/3.12.+Example.csv')
print(rawdata.head())

plt.scatter(rawdata['Satisfaction'],rawdata['Loyalty'])
plt.xlabel('Satisfaction',size = 20)
plt.ylabel('Loyalty',size = 20)
plt.show()

x = rawdata.copy()
kmeans = KMeans(2,random_state=False)
identified_cluster = kmeans.fit_predict(x)

cluster = x.copy()
cluster['Cluster pred'] = identified_cluster
plt.scatter(cluster['Satisfaction'],cluster['Loyalty'],c = cluster['Cluster pred'],cmap = 'rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

#not accurate cause staisfaction not standardize

#standardize features/var
from sklearn import preprocessing

x_scaled = preprocessing.scale(x)
print(x_scaled)

wcss = []

for i in range(1,10):
    kmeans = KMeans(i,random_state=False)
    identified_cluster = kmeans.fit_predict(x_scaled)
    wcss_itter = kmeans.inertia_
    wcss.append(wcss_itter)
    
print(wcss)

number_cluster = range(1,10)
plt.plot(number_cluster,wcss)
plt.xlabel('Cluster',size = 20)
plt.ylabel('wcss',size =20)
plt.title('ELbow Method')
plt.show()

kmean_new = KMeans(5,random_state=False)
kmean_new.fit_predict(x_scaled)
cluster_new = x.copy()
cluster_new['Cluster Pred'] = kmean_new.fit_predict(x_scaled)
print(cluster_new)

plt.scatter(cluster_new['Satisfaction'],cluster_new['Loyalty'],c = cluster_new['Cluster Pred'],cmap ='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()


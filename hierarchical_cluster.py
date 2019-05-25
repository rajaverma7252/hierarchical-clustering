# hierarchycal cluster
"""
Created on Mon Jul  2 14:53:06 2018

@author: Raja
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:\\Users\\Raja\\Desktop\\ml\\Mall_Customers.csv')

X=dataset.iloc[:,[3,4]].values     

# Using elbow method to find the optimal number of cluster
from sklearn.cluster import KMeans
wcss = []    #Empty array
for i in range(1,11):         #run 10 times
    kmeans = KMeans(n_clusters = i, init = 'k-means++')  #k-measn++ is a algorithm, so that no cluster between one cluster can be formed.
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  #inertia_ is criteria so that min no. of cluster are formed.. 
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#fitting k-means to dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(X)

#using the dendrogram to find th optimal number
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian distances')

#fitting Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage ='ward')
y_hc = hc.fit_predict(X)


plt.scatter(X[y_kmeans == 0, 0],X[y_kmeans == 0, 1], s=100 , c='red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0],X[y_kmeans == 1, 1], s=100 , c='blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0],X[y_kmeans == 2, 1], s=100 , c='green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0],X[y_kmeans == 3, 1], s=100 , c='cyan', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4, 0],X[y_kmeans == 4, 1], s=100 , c='magenta', label = 'Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')
plt.title('Cluster of Customers')

plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()



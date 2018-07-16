#importing dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#Using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method');
plt.xlabel("no of clustres");
plt.ylabel("WCSS");
plt.show();    #IT Show optimal no of clusters


#Applying Kmeans to Mail dataset
kmeans=KMeans(n_clusters=5,random_state=0)
y_kmeans=kmeans.fit_predict(x) #IT filter all data point into its respective cluster

#visualising Clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c="red",label="Cluster 1")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c="blue",label="Cluster 2")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c="green",label="Cluster 3")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c="cyan",label="Cluster 4")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c="black",label="Cluster 5")
plt.title("Cluster of CLients")
plt.xlabel("Annual Income")
plt.ylabel("spending score")
plt.legend()
plt.show();
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

dataframe =pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')
print(dataframe.head())

# Min-Max scaling on the numerical columns
columns_to_scale = ['AVERAGE_BASKET_SIZE', 'NO_OF_TRANSACTIONS', 'AVG_BASKET_DOLLARS', 'TOTAL_SALES_UNITS', 'TOTAL_SALES_DOLLARS']

scaler = MinMaxScaler()
dataframe[columns_to_scale] = scaler.fit_transform(dataframe[columns_to_scale])
# dataframe[columns_to_scale] = dataframe[columns_to_scale].round(4)
clustering_data = dataframe[columns_to_scale]

# Elbow Method to find the optimal k
# Computing the KMeans for different k values and calculating inertia
inertia = []
for k in range(1, 11):  # Testing for k between 1 and 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method to visualize the "elbow"
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.show()
    
# Silhouette Score to validate clustering
# using k=4 and computing silhouette score
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(clustering_data)
    
# Calculating silhouette score
silhouette_avg = silhouette_score(clustering_data, kmeans.labels_)
print(f"Silhouette Score for k=4: {silhouette_avg:.2f}")
    
# Adding cluster labels to the dataframe
dataframe['cluster'] = kmeans.labels_

print(dataframe)

# Visualizing the clustering results
plt.figure(figsize=(8,6))
scatter = plt.scatter(dataframe['AVERAGE_BASKET_SIZE'], dataframe['NO_OF_TRANSACTIONS'], c=dataframe['cluster'], cmap='viridis', s=100, alpha=0.8)
plt.title('Clustering Results (k=4)')
plt.xlabel('SCALED AVERAGE BASKET SIZE')
plt.ylabel('SCALED NO. OF TRANSACTIONS')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha = 0.5)
plt.tight_layout()
plt.show()
    
# Return final dataframe with cluster labels
# dataframe.to_csv('test2.csv', index=False)

# Grouping by cluster and calculating summary statistics
cluster_profiles = dataframe.groupby('cluster')[columns_to_scale].mean()

# Adding the count of members in each cluster
cluster_profiles['Count'] = dataframe['cluster'].value_counts()

# Resetting the index for better readability
cluster_profiles.reset_index(inplace=True)

# Printing the profiles for inspection
print("Cluster Profiles:")
print(cluster_profiles)

# Visualizing the cluster profiles
# Bar plot for each feature
plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_scale, 1):
    plt.subplot(2, 3, i)
    plt.bar(cluster_profiles['cluster'], cluster_profiles[column], color='skyblue', alpha=0.7)
    plt.title(f"Average {column} by Cluster")
    plt.xlabel('Cluster')
    plt.ylabel(column)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


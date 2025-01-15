import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes

# Load the dataset
dataframe = pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')

geography_mapping = {
    "W.A": 1, "N.T": 2, "S.A": 3, "QLD": 4, 
    "NSW": 5, "VIC": 6, "ACT": 7, "TAS": 8, "NTH": 9, "STH": 10, "undefined": 11
}
dataframe["STATE"] = dataframe["STATE"].map(geography_mapping)

# print("Initial Dataframe:")
# print(dataframe.head())

# Selecting categorical features for clustering
categorical_features = ['STORE_TYPE', 'AREA', 'COUNTRY']
dataframe_encoded = pd.get_dummies(dataframe[categorical_features])

# Extracting categorical features for clustering
clustering_data = pd.concat([dataframe[['STATE']], dataframe_encoded], axis=1)

# Elbow curve to find optimal K
cost = []
K = range(1,6)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
    kmode.fit_predict(clustering_data)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

# Using k-Modes for categorical clustering
n_clusters = 3  # Set the desired number of clusters
kmodes = KModes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=1)
clusters = kmodes.fit_predict(clustering_data)

# Adding cluster labels to the dataframe
dataframe['cluster'] = clusters

# Display the first few rows of the dataframe with cluster labels
print("Dataframe with Clusters:")
print(dataframe.head())

# Visualizing the number of stores in each cluster
cluster_counts = dataframe['cluster'].value_counts()

# plt.figure(figsize=(8, 6))
# plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.8)
# plt.title('Number of Stores in Each Cluster')
# plt.xlabel('Cluster')
# plt.ylabel('Number of Stores')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.show()

# dataframe.to_csv('kmode2.csv', index=False)

# Analyzing cluster profiles (mode of categorical features in each cluster)
cluster_profiles = dataframe.groupby('cluster')[['STATE', 'COUNTRY', 'STORE_TYPE', 'AREA']].agg(lambda x: x.mode()[0])
cluster_profiles['Count'] = dataframe['cluster'].value_counts()
print("Cluster Profiles:")
print(cluster_profiles)

# Visualize the number of stores in each cluster
plt.bar(cluster_profiles.index, cluster_profiles['Count'], color='skyblue', alpha=0.8)
plt.title('Number of Stores in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# Visualizing cluster profiles
# plt.figure(figsize=(12, 8))
# for i, column in enumerate(categorical_features, 1):
#     plt.subplot(2, 2, i)
#     cluster_profiles[column].value_counts().plot(kind='bar', color='lightblue', alpha=0.7)
#     plt.title(f"Distribution of {column} by Cluster")
#     plt.xlabel(column)
#     plt.ylabel('Count')
#     plt.grid(axis='y', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()

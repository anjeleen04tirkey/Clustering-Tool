import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

# Sample DataFrames: control stores (original dataset) and test stores (after clustering)
control_df = pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')  # Before clustering (Control dataset)
test_df = pd.read_csv('C:/Users/atirkey/OneDrive/airflow/test.csv')  # After clustering (Test dataset)

control_df['is_test'] = 0
test_df['is_test'] = 1

# Scaled columns to be used for similarity calculation
scaled_columns = ['AVERAGE_BASKET_SIZE', 'NO_OF_TRANSACTIONS', 'AVG_BASKET_DOLLARS', 'TOTAL_SALES_UNITS', 'TOTAL_SALES_DOLLARS']

combined_df = pd.concat([control_df, test_df], ignore_index=True)

kmeans = KMeans(n_clusters=4, random_state=42)
combined_df['cluster'] = kmeans.fit_predict(combined_df[scaled_columns])

print(combined_df.columns)


# Separate the combined dataset back into control and test
control_df_clustered = combined_df[combined_df['is_test'] == 0]  # Control stores (before change)
test_df_clustered = combined_df[combined_df['is_test'] == 1]  # Test stores (after change)

print(control_df_clustered.columns)
# Assuming that 'cluster' column already exists in the test_df (this is the result of clustering)
# Now we need to find the control stores that are most similar to each test store in the same cluster.
def find_similar_stores(test_store, control_stores, scaled_columns):
    # Extract the features of the test store (1D array)
    test_features = test_store[scaled_columns].values.reshape(1, -1)
    
    # Extract the features of all control stores
    control_features = control_stores[scaled_columns].values
    
    # Compute the Euclidean distance between the test store and all control stores
    distances = euclidean_distances(test_features, control_features)
    
    # Add a column for distance to the control stores DataFrame
    control_stores.loc[:, 'distance'] = distances[0]
    
    # Sort the control stores by distance (ascending) and get the closest stores
    closest_stores = control_stores.sort_values(by='distance').head(5)  # Adjust number of stores to return

    result_columns = ['STORE', 'STATE', 'STORE_TYPE', 'AREA', 'COUNTRY', 'distance', 'cluster']
    
    return closest_stores[result_columns]  # Return relevant info

# Example: Iterate through each test store and find similar control stores in the same cluster
similar_stores_list = []

for _, test_store in test_df.iterrows():
    test_cluster = test_store['cluster']
    
    # Filter control stores to only those in the same cluster as the current test store
    control_stores_in_same_cluster = control_df_clustered[control_df_clustered['cluster'] == test_cluster]
    
    # Find similar stores
    similar_stores = find_similar_stores(test_store, control_stores_in_same_cluster, scaled_columns)
    
    # Add the test store info to the result
    similar_stores['test_store'] = test_store['STORE']
    
    # Append the result for this test store to the list
    similar_stores_list.append(similar_stores)

# Combine all similar stores data into one DataFrame
final_similar_stores = pd.concat(similar_stores_list)

# Print the result
print(final_similar_stores)

final_similar_stores.to_csv('similar_control_stores.csv', index=False)

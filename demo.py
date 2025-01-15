from kmodes.kprototypes import KPrototypes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataframe = pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')

# Check for missing values and handle them
print("Missing values before imputation:")
print(dataframe.isnull().sum())

# Impute missing values for categorical columns with the most frequent value
categorical_columns = ['STORE_TYPE', 'AREA', 'COUNTRY', 'STATE']
for col in categorical_columns:
    dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)

# Impute missing values for numerical columns with the mean
numerical_columns = ['AVERAGE_BASKET_SIZE', 'NO_OF_TRANSACTIONS', 'AVG_BASKET_DOLLARS',
                     'TOTAL_SALES_UNITS', 'TOTAL_SALES_DOLLARS']
for col in numerical_columns:
    dataframe[col].fillna(dataframe[col].mean(), inplace=True)

# Verify no missing values remain
print("Missing values after imputation:")
print(dataframe.isnull().sum())

# --- Geographical Encoding (Label Encoding) ---

# Create a label encoder object
label_encoder = LabelEncoder()

# Encode geographical features like STATE, AREA, and COUNTRY using Label Encoding
dataframe['STATE_encoded'] = label_encoder.fit_transform(dataframe['STATE'])
dataframe['AREA_encoded'] = label_encoder.fit_transform(dataframe['AREA'])
dataframe['COUNTRY_encoded'] = label_encoder.fit_transform(dataframe['COUNTRY'])

# You can also use one-hot encoding if needed:
# dataframe = pd.get_dummies(dataframe, columns=['STATE', 'AREA', 'COUNTRY'])

# --- Prepare Data for Clustering ---

# Select the numerical columns and the encoded geographical columns
clustering_data = dataframe[numerical_columns + ['STATE_encoded', 'AREA_encoded', 'COUNTRY_encoded']]

# --- Elbow Method for optimal k ---

cost = []
K = range(1, 6)  # Test clusters from 1 to 5
for k in K:
    kproto = KPrototypes(n_clusters=k, init='Cao', verbose=1, random_state=42)
    kproto.fit_predict(clustering_data, categorical=[clustering_data.columns.get_loc(col) for col in ['STATE_encoded', 'AREA_encoded', 'COUNTRY_encoded']])
    cost.append(kproto.cost_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(K, cost, marker='o', linestyle='-', color='b')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# --- Clustering with optimal number of clusters (example: k=3) ---

optimal_k = 3  # Choose based on the Elbow plot
kproto = KPrototypes(n_clusters=optimal_k, init='Cao', verbose=1, random_state=42)
clusters = kproto.fit_predict(clustering_data, categorical=[clustering_data.columns.get_loc(col) for col in ['STATE_encoded', 'AREA_encoded', 'COUNTRY_encoded']])

# Add the cluster labels to the dataframe
dataframe['Cluster'] = clusters

# Print the resulting clusters
print("Cluster assignments:")
print(dataframe[['Cluster'] + numerical_columns + ['STATE_encoded', 'AREA_encoded', 'COUNTRY_encoded']])

# --- Analyze Cluster Profiles ---

cluster_profiles = dataframe.groupby('Cluster')[numerical_columns + ['STATE_encoded', 'AREA_encoded', 'COUNTRY_encoded']].agg(['mean', 'count'])
print("\nCluster Profiles:")
print(cluster_profiles)

# --- Save the results to a CSV file (optional) ---

dataframe.to_csv('clustered_data.csv', index=False)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Clustering Tool", page_icon="📊", layout="wide")

# Load data
data = pd.read_csv('C:/Users/atirkey/OneDrive/airflow/cleaned_data.csv')
df = pd.DataFrame(data)

st.title("📊 Clustering Tool")
st.sidebar.header("🛠️ Select Features")

cat_features = st.sidebar.multiselect("Categorical Features", ["STATE", "STORE_TYPE", "AREA", "COUNTRY"])
num_features = st.sidebar.multiselect("Numerical Features", ["AVERAGE_BASKET_SIZE", "NO_OF_TRANSACTIONS", "AVG_BASKET_DOLLARS", "TOTAL_SALES_UNITS", "TOTAL_SALES_DOLLARS"])

if not cat_features and not num_features:
    st.warning("⚠️ Select at least one categorical or numerical feature")
    st.stop()

# Preprocessing
selected_data = df[cat_features + num_features].copy()

encoded_cats = pd.DataFrame()
scaled_nums = pd.DataFrame()

# One-hot encoding for categorical variables
if cat_features:
    encoded_cats = pd.get_dummies(selected_data[cat_features], drop_first=False)
    encoded_cats = encoded_cats.astype(int)

# Scaling numerical variables
if num_features:
    scaler = MinMaxScaler()
    scaled_nums = scaler.fit_transform(selected_data[num_features])
    scaled_nums = pd.DataFrame(scaled_nums, columns=num_features)

# Display Preprocessed Data
st.write("Preprocessed Data (Categorical and Numerical)")
st.write("Categorical Variables Encoded:")
st.dataframe(encoded_cats)
st.write("Numerical Variables Scaled:")
st.dataframe(scaled_nums)

# Clustering Type Decision
if not encoded_cats.empty and not scaled_nums.empty:
    clustering_method = "kprototypes"  # Mixed features selected
elif not encoded_cats.empty:
    clustering_method = "kmodes"  # Only categorical features selected
else:
    clustering_method = "kmeans"  # Only numerical features selected

st.header("Clustering Results")

# Perform Clustering Based on Selected Features
if clustering_method == "kmodes":
    st.subheader("Using K-Modes Clustering (All Categorical Features)")
    cat_costs = []
    for k in range(1, 11):
        kmodes = KModes(n_clusters=k, init='Huang', random_state=42)
        kmodes.fit(encoded_cats)
        cat_costs.append(kmodes.cost_)

    # Elbow Method for K-Modes
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), cat_costs, marker="o")
    ax.set_title("Elbow Method for K-Modes")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Cost")
    ax.grid(True)
    st.pyplot(fig)

    num_clusters = st.slider("Select Number of Clusters", 2, 9, 3)
    kmodes = KModes(n_clusters=num_clusters, init='Huang', random_state=42)
    clusters = kmodes.fit_predict(encoded_cats)
    selected_data["Cluster"] = clusters

elif clustering_method == "kmeans":
    st.subheader("Using K-Means Clustering (All Numerical Features)")
    costs = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_nums)
        costs.append(kmeans.inertia_)

    # Elbow Method for K-Means
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), costs, marker="o")
    ax.set_title("Elbow Method for K-Means")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia")
    ax.grid(True)
    st.pyplot(fig)

    num_clusters = st.slider("Select Number of Clusters", 2, 9, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_nums)
    selected_data["Cluster"] = clusters

else:  # K-Prototypes (Mixed Data)
    st.subheader("Using K-Prototypes Clustering (Mixed Features)")
    # Get indices for categorical columns in the combined dataset
    categorical_indices = list(range(encoded_cats.shape[1]))
    
    # Concatenate the encoded categorical data and scaled numerical data
    combined_data = pd.concat([encoded_cats, scaled_nums], axis=1)
    
    costs = []
    for k in range(1, 11):
        kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42)
        kproto.fit(combined_data, categorical=categorical_indices)
        costs.append(kproto.cost_)

    # Elbow Method for K-Prototypes
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), costs, marker="o")
    ax.set_title("Elbow Method for K-Prototypes")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Cost")
    ax.grid(True)
    st.pyplot(fig)

    num_clusters = st.slider("Select Number of Clusters", 2, 9, 3)
    kproto = KPrototypes(n_clusters=num_clusters, random_state=42)
    clusters = kproto.fit_predict(combined_data, categorical=categorical_indices)
    selected_data["Cluster"] = clusters

# Display cluster assignments
selected_data["STORE"] = df["STORE"]
st.write("Cluster Assignments:")
st.dataframe(selected_data)

# Nearest Neighbor Search
st.header("Find Nearest Stores Using KNN")
test_store_id = st.selectbox("Select a Test Store", df["STORE"].unique())
st.write(f"Selected Test Store Data:")
test_store_data = df[df["STORE"] == test_store_id]
st.write(test_store_data)

knn_data = pd.concat([scaled_nums, encoded_cats], axis=1)
knn_data["Cluster"] = selected_data["Cluster"]

knn_data_numeric = knn_data.select_dtypes(include=[float,int])

test_store_index = selected_data[selected_data["STORE"] == test_store_id].index[0]

test_cluster = selected_data.loc[test_store_index, "Cluster"]
cluster_data = knn_data_numeric[selected_data["Cluster"] == test_cluster]

knn_filtered = NearestNeighbors(n_neighbors=5)
knn_filtered.fit(cluster_data)

distances, indices = knn_filtered.kneighbors(cluster_data.iloc[[test_store_index]])


control_stores = df.iloc[indices[0]].copy()
control_stores = df.iloc[selected_data[selected_data["Cluster"] == test_cluster].index[indices[0]]].copy()
control_stores["Cluster"] = test_cluster
st.header("Control Stores")
st.write(control_stores)

# PCA Visualization
st.header("Cluster Visualization")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(knn_data_numeric)

reduced_df = pd.DataFrame(reduced_data, columns=["PC1", "PC2"])
reduced_df["Cluster"] = selected_data["Cluster"]

fig = px.scatter(
    reduced_df,
    x="PC1",
    y="PC2",
    color=reduced_df["Cluster"].astype(str),
    title="2D Cluster Visualization (PCA)",
    hover_data=[selected_data["STORE"]],
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig)     

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from kmodes.kmodes import KModes

data = pd.read_csv('C:/Users/atirkey/OneDrive/airflow/cleaned_data.csv')
df = pd.DataFrame(data)

st.title("Clustering Tool")
st.sidebar.header("Select features")
cat_features = st.sidebar.multiselect("Categorical Features", ["STATE", "STORE_TYPE", "AREA", "COUNTRY"])
num_features = st.sidebar.multiselect("Numerical Features", ["AVERAGE_BASKET_SIZE", "NO_OF_TRANSACTIONS", "AVG_BASKET_DOLLARS", "TOTAL_SALES_UNITS", "TOTAL_SALES_DOLLARS"])

if not cat_features and not num_features:
    st.warning("Please select at least one categorical or numerical feature")
    st.stop()

selected_data = df[cat_features + num_features].copy()

encoded_data = pd.DataFrame()

if cat_features:
    encoded_cats = pd.get_dummies(selected_data[cat_features], drop_first=False)
    encoded_cats = encoded_cats.astype(int)
    encoded_data = pd.concat([encoded_data, encoded_cats], axis=1)

if num_features:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(selected_data[num_features])
    scaled_data = pd.DataFrame(scaled_data, columns=num_features)
    encoded_data = pd.concat([encoded_data, scaled_data], axis=1)

st.write("Preprocessed Data")
st.dataframe(encoded_data)

st.header("Determine Optimal Number of Clusters")
costs = []

for k in range(1,11):
    kmodes = KModes(n_clusters=k, init='Huang', random_state=42)
    kmodes.fit(encoded_cats)
    costs.append(kmodes.cost_)

fig, ax = plt.subplots()
ax.plot(range(1,11), costs, marker="o")
ax.set_title("Elbow method for optimal K")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Cost (Inertia)")
st.pyplot(fig)

optimal_k = st.slider("Select number of Clusters (K)", 2, 9, 3)
kmodes = KModes(n_clusters=optimal_k, random_state=42)
clusters = kmodes.fit_predict(encoded_data)
encoded_data["Cluster"] = clusters

st.header("Clustering results")
df["Cluster"] = clusters
st.dataframe(df)

test_store_id = st.selectbox("Select a Test Store", df["STORE"].unique())
st.write(f"Selected Test Store Data:")
test_store_data = df[df["STORE"] == test_store_id]
st.write(test_store_data)

test_store_encoded = encoded_data.loc[df["STORE"] == test_store_id].drop(columns=["Cluster"])
knn = NearestNeighbors(n_neighbors=5)
knn.fit(encoded_data.drop(columns=["Cluster"]))
distances, indices = knn.kneighbors(test_store_encoded)

st.header("Control Stores")
control_stores = df.reset_index(drop=True).iloc[indices[0]]
st.write(control_stores)
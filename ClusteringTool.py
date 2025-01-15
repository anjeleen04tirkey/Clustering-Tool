import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from kmodes.kprototypes import KPrototypes

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

if cat_features:
    encoded_cats = pd.get_dummies(selected_data[cat_features], drop_first=False)
    encoded_cats = encoded_cats.astype(int)
else:
    encoded_cats = pd.DataFrame()

if num_features:
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(selected_data[num_features]),columns=num_features)
else:
    scaled_data = pd.DataFrame()

combined_data = pd.concat([encoded_cats, scaled_data], axis=1)
categorical_indices = list(range(len(encoded_cats.columns)))

st.write("Preprocessed Data")
st.dataframe(combined_data)

st.header("Determine Optimal Number of Clusters")
costs = []

for k in range(1,11):
    kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
    kproto.fit(combined_data, categorical=categorical_indices)
    costs.append(kproto.cost_)

fig, ax = plt.subplots()
ax.plot(range(1,11), costs, marker="o")
ax.set_title("Elbow method for optimal K")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Cost (Inertia)")
st.pyplot(fig)

optimal_k = st.slider("Select number of Clusters (K)", 2, 9, 3)
kproot = KPrototypes(n_clusters=optimal_k, random_state=42)
clusters = kproto.fit_predict(combined_data, categorical=categorical_indices)

df["Cluster"] = clusters
combined_data["Cluster"] = clusters

st.header("Clustering results")
st.dataframe(df)

test_store_id = st.selectbox("Select a Test Store", df["STORE"].unique())
st.write(f"Selected Test Store Data:")
test_store_data = df[df["STORE"] == test_store_id]
st.write(test_store_data)

test_store_encoded = combined_data.loc[df["STORE"] == test_store_id].drop(columns=["Cluster"])
knn = NearestNeighbors(n_neighbors=5)
knn.fit(combined_data.drop(columns=["Cluster"]))
distances, indices = knn.kneighbors(test_store_encoded)

st.header("Control Stores")
control_stores = df.reset_index(drop=True).iloc[indices[0]]
st.write(control_stores)
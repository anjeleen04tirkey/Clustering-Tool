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
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Clustering Tool", page_icon="📊", layout="wide")

def calculate_silhouette_scores(data, cluster_labels):
    if len(set(cluster_labels)) > 1:  # Silhouette requires at least two clusters
        return silhouette_score(data, cluster_labels)
    return None

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

    # User Input for Cluster Options
    st.write("### Choose three K values to calculate Silhouette Scores:")
    cluster_options = st.text_input("Enter three numbers separated by commas (e.g., 3,4,5):")
    if cluster_options:
        try:
            cluster_choices = [int(x.strip()) for x in cluster_options.split(",")]
            silhouette_scores = []
            for k in cluster_choices:
                kmodes = KModes(n_clusters=k, init='Huang', random_state=42)
                cluster_labels = kmodes.fit_predict(encoded_cats)
                score = calculate_silhouette_scores(encoded_cats, cluster_labels)
                silhouette_scores.append((k, score))
            st.write("### Silhouette Scores:")
            for k, score in silhouette_scores:
                st.write(f"K={k}: Silhouette Score={score:.4f}")

            # Final Selection of K
            optimal_k = st.selectbox("Select the optimal number of clusters (K) based on Silhouette Scores:", cluster_choices)
            kmodes = KModes(n_clusters=optimal_k, init='Huang', random_state=42)
            clusters = kmodes.fit_predict(encoded_cats)
            selected_data["Cluster"] = clusters

        except ValueError:
            st.warning("Invalid input. Please enter three integers separated by commas.")


    # num_clusters = st.slider("Select Number of Clusters", 2, 9, 3)
    # kmodes = KModes(n_clusters=num_clusters, init='Huang', random_state=42)
    # clusters = kmodes.fit_predict(encoded_cats)
    # selected_data["Cluster"] = clusters

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

    # User Input for Cluster Options
    st.write("### Choose three K values to calculate Silhouette Scores:")
    cluster_options = st.text_input("Enter three numbers separated by commas (e.g., 3,4,5):")
    if cluster_options:
        try:
            cluster_choices = [int(x.strip()) for x in cluster_options.split(",")]
            silhouette_scores = []
            for k in cluster_choices:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_nums)
                score = calculate_silhouette_scores(scaled_nums, cluster_labels)
                silhouette_scores.append((k, score))
            st.write("### Silhouette Scores:")
            for k, score in silhouette_scores:
                st.write(f"K={k}: Silhouette Score={score:.4f}")

            # Final Selection of K
            optimal_k = st.selectbox("Select the optimal number of clusters (K) based on Silhouette Scores:", cluster_choices)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(scaled_nums)
            selected_data["Cluster"] = clusters

        except ValueError:
            st.warning("Invalid input. Please enter three integers separated by commas.")


    # num_clusters = st.slider("Select Number of Clusters", 2, 9, 3)
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # clusters = kmeans.fit_predict(scaled_nums)
    # selected_data["Cluster"] = clusters

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

    # User Input for Cluster Options
    st.write("### Choose three K values to calculate Silhouette Scores:")
    cluster_options = st.text_input("Enter three numbers separated by commas (e.g., 3,4,5):")
    if cluster_options:
        try:
            cluster_choices = [int(x.strip()) for x in cluster_options.split(",")]
            silhouette_scores = []
            for k in cluster_choices:
                kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42)
                cluster_labels = kproto.fit_predict(combined_data, categorical=categorical_indices)
                score = calculate_silhouette_scores(combined_data, cluster_labels)
                silhouette_scores.append((k, score))
            st.write("### Silhouette Scores:")
            for k, score in silhouette_scores:
                st.write(f"K={k}: Silhouette Score={score:.4f}")

            # Final Selection of K
            optimal_k = st.selectbox("Select the optimal number of clusters (K) based on Silhouette Scores:", cluster_choices)
            kproto = KPrototypes(n_clusters=optimal_k, random_state=42)
            clusters = kproto.fit_predict(combined_data, categorical=categorical_indices)
            selected_data["Cluster"] = clusters

        except ValueError:
            st.warning("Invalid input. Please enter three integers separated by commas.")


    # num_clusters = st.slider("Select Number of Clusters", 2, 9, 3)
    # kproto = KPrototypes(n_clusters=num_clusters, random_state=42)
    # clusters = kproto.fit_predict(combined_data, categorical=categorical_indices)
    # selected_data["Cluster"] = clusters

# Display cluster assignments
selected_data["STORE"] = df["STORE"]
selected_data = selected_data[["STORE", "Cluster"] + cat_features + num_features].sort_values(by="Cluster")
# st.write("Cluster Assignments:")
# st.dataframe(selected_data)

st.subheader("Select Test Stores")
gb = GridOptionsBuilder.from_dataframe(selected_data)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
grid_options = gb.build()

grid_response = AgGrid(
    selected_data,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=True,
    theme="streamlit",
    enable_enterprise_modules=True,
)

# Retrieve selected rows
selected_rows = grid_response["selected_rows"]

if st.button("Find Control Stores"):
    if selected_rows is not None and not selected_rows.empty:
    # if selected_rows and len(selected_rows) > 0:
        selected_rows = pd.DataFrame(selected_rows)
        selected_rows = selected_rows.to_dict(orient="records")

        knn_data = pd.concat([scaled_nums, encoded_cats], axis=1)
        knn_data["Cluster"] = selected_data["Cluster"]
        knn_data["STORE"] = selected_data["STORE"]

        st.subheader("Control Stores for selected test stores")
        control_stores_list = []

        for row in selected_rows:
            if isinstance(row, dict) and "STORE" in row:
                store_id = row["STORE"]
                test_store_index = selected_data[selected_data["STORE"] == store_id].index
                if len(test_store_index) > 0:
                    test_store_index = test_store_index[0]
                else:
                    st.warning(f"Store ID {store_id} not found in selected data")
                    continue

                test_cluster = selected_data.loc[test_store_index, "Cluster"]
                cluster_data = knn_data[(knn_data["Cluster"] == test_cluster) & (knn_data["STORE"] != store_id)]

                if cluster_data.empty:
                    st.warning(f"No control stores available in the same cluster for Store ID {store_id}")
                    continue

                knn_filtered = NearestNeighbors(n_neighbors=min(5, len(cluster_data)))
                knn_filtered.fit(cluster_data.drop(["STORE", "Cluster"], axis=1))

                test_store_data = knn_data[knn_data["STORE"] == store_id].drop(["STORE", "Cluster"], axis=1).values
                distances, indices = knn_filtered.kneighbors(test_store_data)
                control_stores = cluster_data.iloc[indices[0]].copy()
                control_stores_list.append((store_id, control_stores))
            else:
                st.warning("Selected row format is invalid or 'STORE' key is missing.")
    
        for store_id, control_stores in control_stores_list:
            test_store_details = df[df["STORE"] == store_id]

            test_store_cluster = selected_data[selected_data["STORE"] == store_id]["Cluster"].values[0]
            test_store_details["Cluster"] = test_store_cluster
        
            store_col = test_store_details.pop("STORE")
            test_store_details.insert(0, "STORE", store_col)

            st.write(f"Test Store: {store_id}")
            st.dataframe(test_store_details)

            control_store_ids = control_stores["STORE"].values
            control_store_details = df[df["STORE"].isin(control_store_ids)]

            control_store_clusters = selected_data[selected_data["STORE"].isin(control_store_ids)][["STORE", "Cluster"]]
            control_store_details = pd.merge(control_store_details, control_store_clusters, on="STORE", how="left")

            numerical_columns = control_store_details.select_dtypes(include=["float", "int"]).columns
            control_store_details[numerical_columns] = control_store_details[numerical_columns].round(2)
            
            store_col = control_store_details.pop("STORE")
            control_store_details.insert(0, "STORE", store_col)

            st.write(f"Control Stores for Test Store: {store_id}")
            st.dataframe(control_store_details)

    else:
        st.info("No stores selected")
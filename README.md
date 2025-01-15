# Clustering Tool 🛠️📊
An interactive web application built with Streamlit for performing and analyzing clustering on datasets with categorical, numerical, or mixed features. This tool empowers users to preprocess data, explore optimal clustering configurations, and analyze cluster assignments.

# Features ✨
## 🛠️ Preprocessing
One-Hot Encoding: Encodes categorical features for clustering.
Scaling: Scales numerical features using MinMaxScaler.

## 📊 Clustering
K-Modes: For datasets with categorical features.
K-Means: For datasets with numerical features.
K-Prototypes: For mixed datasets with both feature types.
Elbow Method: Helps determine potential cluster numbers.
Silhouette Scores: Provides clarity on cluster quality for user-selected cluster counts.

## 🔍 Cluster Analysis
View preprocessed data (encoded categorical and scaled numerical features).
Assign clusters to the dataset and display results.

## 🔄 Test-Control Store Matching
Select test stores and find matching control stores using Nearest Neighbors.
Analyze clusters and compare stores within the same cluster.

## 📈 Visualizations
Elbow graph for cluster evaluation.
Interactive tables using st-aggrid for exploring cluster assignments.

# Getting Started 🚀
## Prerequisites
Python 3.7 or higher
Required libraries (see requirements.txt)

## Installation
1. Clone the repository: git clone https://github.com/your-repo-name/clustering-tool.git
cd clustering-tool
2. Install dependencies: pip install -r requirements.txt
3. Run the application: streamlit run app.py

# How to Use 🧑‍💻
1. Upload Dataset
   - Prepare a dataset in CSV format containing categorical and/or numerical features.
   - Update the dataset path in the script or modify the app to accept file uploads.
2. Feature Selection
   - Use the sidebar to select categorical and numerical features for clustering.
3. Clustering Configuration
   - View the elbow graph and select multiple potential cluster counts (e.g., 3, 4, 5).
   - Analyze the silhouette scores for the selected cluster counts and choose the optimal number of clusters.
4. Test-Control Store Selection
   - Select test stores and find control stores from the same cluster using Nearest Neighbors.

# Technologies Used 🛠️
1. Streamlit: Interactive web app framework.
2. Pandas: Data manipulation and analysis.
3. Scikit-learn: Preprocessing, clustering, silhouette score calculation.
4. KModes & KPrototypes: Clustering categorical and mixed datasets.
5. Matplotlib & Plotly: Visualizations.
6. st-aggrid: Interactive tables for cluster analysis.

# Future Improvements 🌟
1. Add file upload functionality for datasets.
2. Support additional clustering algorithms.
3. Improve visualization for cluster distribution and feature impact.

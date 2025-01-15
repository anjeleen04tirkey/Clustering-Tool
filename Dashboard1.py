import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# Load the data
data = pd.read_csv('C:/Users/atirkey/OneDrive/airflow/.vscode/KMeansClustering.csv')
df = pd.DataFrame(data)

# Set a custom theme for Streamlit
st.set_page_config(page_title="Store Performance Dashboard", page_icon="📊", layout="wide")

# Streamlit Dashboard Title with custom font and color
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📊 Store Performance Live Dashboard</h1>", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🎛️ Filters")

# Filter by State
states = df['STATE'].unique()
selected_states = st.sidebar.multiselect('🌍 Select State', options=states, default=states, help="Select one or more states to filter stores.")

# Filter by Store Type
store_types = df['STORE_TYPE'].unique()
selected_store_types = st.sidebar.multiselect('🏬 Select Store Type(s)', options=store_types, default=store_types, help="Select one or more store types.")

# Filter by Metric
metrics = ['TOTAL_SALES_DOLLARS', 'TOTAL_SALES_UNITS', 'AVERAGE_BASKET_SIZE', 'NO_OF_TRANSACTIONS', 'AVG_BASKET_DOLLARS']
selected_metric = st.sidebar.selectbox('📈 Select Metric for Analysis', options=metrics, key='metric', help="Choose the metric you want to analyze.")

# Filter Data Based on Selections
filtered_data = df.copy()

if selected_states:
    filtered_data = filtered_data[filtered_data['STATE'].isin(selected_states)]

if selected_store_types:
    filtered_data = filtered_data[filtered_data['STORE_TYPE'].isin(selected_store_types)]

clusters = df['Cluster'].unique()
selected_cluster = st.sidebar.selectbox('🔍 Select Cluster', options=clusters, help="Select a cluster to analyze similar stores.")

cluster_data = filtered_data[filtered_data['Cluster'] == selected_cluster]

# Display similar stores in the selected cluster
st.markdown(f"<h3 style='color: #1E40AF;'>📋 Similar Stores in Cluster {selected_cluster}</h3>", unsafe_allow_html=True)
similar_stores = cluster_data[['STORE', 'STATE', 'STORE_TYPE', 'AREA', 'COUNTRY', 'AVERAGE_BASKET_SIZE', 'NO_OF_TRANSACTIONS', 'AVG_BASKET_DOLLARS', 'TOTAL_SALES_UNITS', 'TOTAL_SALES_DOLLARS']]
st.dataframe(similar_stores)

# Displaying Filtered Data
st.markdown("<h3 style='color: #1E40AF;'>📋 Filtered Data</h3>", unsafe_allow_html=True)
st.write(filtered_data)

# Visualization: Total Sales or Selected Metric by Store
st.markdown(f"<h3 style='color: #16A34A;'>📈 {selected_metric} by Store</h3>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(12, 6))

# Creating dynamic plot based on the selected metric
sns.barplot(x="STORE", y=selected_metric, data=filtered_data, ax=ax, color="#60A5FA")
ax.set_title(f"{selected_metric} by Store", fontsize=18, fontweight='bold', color='#4A90E2')
ax.set_xlabel("Store", fontsize=14)
ax.set_ylabel(f"{selected_metric} (in dollars)" if selected_metric == 'TOTAL_SALES_DOLLARS' else selected_metric, fontsize=14)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Top 10 Stores based on Selected Metric
# st.subheader(f"Top 10 Stores by {selected_metric}")

# Grouping and sorting the data
top_10_stores = (
    filtered_data.groupby("STORE")[selected_metric]
    .sum()
    .reset_index()
    .sort_values(by=selected_metric, ascending=False)
    .head(10)
)

st.markdown("<h3 style='color: #9333EA;'> Top 10 Stores Table</h3>", unsafe_allow_html=True)
st.dataframe(
    top_10_stores,
    column_order=["STORE", selected_metric],
    hide_index=True,
    width=None,
    column_config={
        "STORE": st.column_config.TextColumn("🏬 Store"),
        selected_metric: st.column_config.ProgressColumn(
            f"📊 {selected_metric}",
            format="%f",
            min_value=0,
            max_value=max(top_10_stores[selected_metric]),
        )
    }
)

# Scatter Plot: Number of Transactions vs Total Sales
st.markdown("<h3 style='color: #DC2626;'>🔍 Transactions vs Total Sales (Scatter Plot)</h3>", unsafe_allow_html=True)
scatter_chart = (
    alt.Chart(filtered_data)
    .mark_circle(size=100, color="#4A90E2")
    .encode(
        x=alt.X("NO_OF_TRANSACTIONS", title="Number of Transactions"),
        y=alt.Y("TOTAL_SALES_DOLLARS", title="Total Sales (in dollars)"),
        tooltip=["STORE", "NO_OF_TRANSACTIONS", "TOTAL_SALES_DOLLARS"]
    )
    .properties(width=800, height=400, title="Number of Transactions vs Total Sales Dollars")
    .configure_title(fontSize=18, font='Arial', anchor='start', color='#4A90E2')
)
st.altair_chart(scatter_chart, use_container_width=True)

# Percentage of Total Sales Dollars for Each Area
st.markdown("<h3 style='color: #9333EA;'>🔍 Percentage of Total Sales Dollars for Each Area</h3>", unsafe_allow_html=True)

# Calculating the percentage of total sales dollars by area
area_sales_percentage = (
    filtered_data.groupby("AREA")["TOTAL_SALES_DOLLARS"]
    .sum()
    .reset_index()
)
area_sales_percentage["PERCENTAGE"] = (
    area_sales_percentage["TOTAL_SALES_DOLLARS"]
    / area_sales_percentage["TOTAL_SALES_DOLLARS"].sum()
) * 100

# Plotting the bar chart
area_chart = (
    alt.Chart(area_sales_percentage)
    .mark_bar(color="#FFA07A")
    .encode(
        x=alt.X("AREA", sort="-y", title="Area"),
        y=alt.Y("PERCENTAGE", title="Percentage of Total Sales Dollars"),
        tooltip=["AREA", "PERCENTAGE"]
    )
    .properties(
        width=800,
        height=400,
    )
    .configure_title(fontSize=18, font='Arial', anchor='start', color='#FF4500')
    .configure_axis(labelFontSize=12, titleFontSize=14)
)
st.altair_chart(area_chart, use_container_width=True)

# Distribution of Average Basket Size by State (Bar Plot)
st.markdown("<h3 style='color: #059669;'>📦 Average Basket Size by State</h3>", unsafe_allow_html=True)
state_avg_basket = (
    filtered_data.groupby("STATE")["AVERAGE_BASKET_SIZE"]
    .mean()
    .reset_index()
)

state_chart = (
    alt.Chart(state_avg_basket)
    .mark_bar(color="#1E3A8A")
    .encode(
        x=alt.X("STATE", sort="-y", title="State"),
        y=alt.Y("AVERAGE_BASKET_SIZE", title="Average Basket Size"),
        tooltip=["STATE", "AVERAGE_BASKET_SIZE"]
    )
    .properties(width=800, height=400, title="Average Basket Size by State")
    .configure_title(fontSize=18, font='Arial', anchor='start', color='#4A90E2')
    .configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
)
st.altair_chart(state_chart, use_container_width=True)

# Showing the raw data (for deep dive)
st.sidebar.subheader("Raw Data (for Deep Dive)")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df)

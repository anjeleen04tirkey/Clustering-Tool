import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Data (Replace with pd.read_csv() for real data)
data = pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')

# Create DataFrame
df = pd.DataFrame(data)

# Streamlit Dashboard Title
st.title("Store Performance Live Dashboard")

# 1. Filters (Interactive Widgets)
st.sidebar.header("Filters")

# Filter by State
states = df['STATE'].unique()
selected_state = st.sidebar.selectbox('Select State', options=['All States'] + list(states))

# Filter by Store Type
store_types = df['STORE_TYPE'].unique()
selected_store_type = st.sidebar.selectbox('Select Store Type', options=['All Store Types'] + list(store_types))

# Filter by Metric
metrics = ['TOTAL_SALES_DOLLARS', 'AVERAGE_BASKET_SIZE', 'NO_OF_TRANSACTIONS', 'AVG_BASKET_DOLLARS']
selected_metric = st.sidebar.selectbox('Select Metric for Analysis', options=metrics)

# 2. Filter Data Based on Selections
filtered_data = df.copy()

# Filter by State
if selected_state != 'All States':
    filtered_data = filtered_data[filtered_data['STATE'] == selected_state]

# Filter by Store Type
if selected_store_type != 'All Store Types':
    filtered_data = filtered_data[filtered_data['STORE_TYPE'] == selected_store_type]

# 3. Display Filtered Data
st.subheader("Filtered Data")
st.write(filtered_data)

# 4. Visualization: Total Sales vs Store (or any selected metric)
st.subheader(f"{selected_metric} by Store")
fig, ax = plt.subplots(figsize=(10, 6))

# Create a dynamic plot based on the selected metric
sns.barplot(x="STORE", y=selected_metric, data=filtered_data, ax=ax)
ax.set_title(f"{selected_metric} by Store")
ax.set_xlabel("Store")
ax.set_ylabel(f"{selected_metric} (in dollars)" if selected_metric == 'TOTAL_SALES_DOLLARS' else selected_metric)
st.pyplot(fig)

# 5. Scatter Plot: No. of Transactions vs Total Sales (live update)
st.subheader("Transactions vs Total Sales (Scatter Plot)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x="NO_OF_TRANSACTIONS", y="TOTAL_SALES_DOLLARS", data=filtered_data, ax=ax)
ax.set_title("Number of Transactions vs Total Sales Dollars")
ax.set_xlabel("Number of Transactions")
ax.set_ylabel("Total Sales (in dollars)")
st.pyplot(fig)

# 6. Distribution of Average Basket Size by State (Bar Plot)
st.subheader("Average Basket Size by State")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="STATE", y="AVERAGE_BASKET_SIZE", data=filtered_data, ax=ax)
ax.set_title("Average Basket Size by State")
ax.set_ylabel("Average Basket Size")
st.pyplot(fig)

# Optional: Show the raw data (for deep dive)
st.sidebar.subheader("Raw Data (for Deep Dive)")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(filtered_data)

# Real-time Updates: This works when you adjust any filters
st.sidebar.subheader("Live Dashboard Interaction")
st.sidebar.markdown("""
    You can adjust the filters above to dynamically update the dashboard.
    The charts and data will update in real time based on your selections.
    Use this dashboard to explore key metrics for your retail stores.
""")

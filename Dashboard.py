import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')
df = pd.DataFrame(data)

# Streamlit Dashboard Title
st.title("Store Performance Live Dashboard")

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

# Filter Data Based on Selections
filtered_data = df.copy()

# Filter by State
if selected_state != 'All States':
    filtered_data = filtered_data[filtered_data['STATE'] == selected_state]

# Filter by Store Type
if selected_store_type != 'All Store Types':
    filtered_data = filtered_data[filtered_data['STORE_TYPE'] == selected_store_type]

# Displaying Filtered Data
st.subheader("Filtered Data")
st.write(filtered_data)

# Visualization: Total Sales vs Store (or any selected metric)
st.subheader(f"{selected_metric} by Store")
fig, ax = plt.subplots(figsize=(10, 6))

# Creating dynamic plot based on the selected metric
sns.barplot(x="STORE", y=selected_metric, data=filtered_data, ax=ax)
ax.set_title(f"{selected_metric} by Store")
ax.set_xlabel("Store")
ax.set_ylabel(f"{selected_metric} (in dollars)" if selected_metric == 'TOTAL_SALES_DOLLARS' else selected_metric)
st.pyplot(fig)

# **Top 10 Stores by Selected Metric (Horizontal Bar Chart)**
# st.subheader(f"Top 10 Stores by {selected_metric}")
top_10_stores = (
    filtered_data.groupby("STORE")[selected_metric]
    .sum()
    .reset_index()
    .sort_values(by=selected_metric, ascending=False)
    .head(10)
)

# Plotting the top 10 stores as a horizontal bar chart
# fig, ax = plt.subplots(figsize=(10, 6))
# sns.barplot(
#     data=top_10_stores,
#     x=selected_metric,
#     y="STORE",
#     color="skyblue",  # Adjust the color as needed
#     ax=ax
# )
# ax.set_title(f"Top 10 Stores by {selected_metric}")
# ax.set_xlabel(f"{selected_metric} (in dollars)" if selected_metric == 'TOTAL_SALES_DOLLARS' else selected_metric)
# ax.set_ylabel("Store")
# st.pyplot(fig)

# Displaying the top 10 stores in a dataframe with progress bars
st.write("### Top 10 Stores Table")
st.dataframe(
    top_10_stores,
    column_order=["STORE", selected_metric],
    hide_index=True,
    width=None,
    column_config={
        "STORE": st.column_config.TextColumn("Store"),
        selected_metric: st.column_config.ProgressColumn(
            f"{selected_metric}",
            format="%f",
            min_value=0,
            max_value=max(top_10_stores[selected_metric]),
        )
    }
)

# Scatter Plot: No. of Transactions vs Total Sales (live update)
st.subheader("Transactions vs Total Sales (Scatter Plot)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x="NO_OF_TRANSACTIONS", y="TOTAL_SALES_DOLLARS", data=filtered_data, ax=ax)
ax.set_title("Number of Transactions vs Total Sales Dollars")
ax.set_xlabel("Number of Transactions")
ax.set_ylabel("Total Sales (in dollars)")
st.pyplot(fig)

# Distribution of Average Basket Size by State (Bar Plot)
st.subheader("Average Basket Size by State")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="STATE", y="AVERAGE_BASKET_SIZE", data=filtered_data, ax=ax)
ax.set_title("Average Basket Size by State")
ax.set_ylabel("Average Basket Size")
st.pyplot(fig)

# Showing the raw data (for deep dive)
st.sidebar.subheader("Raw Data (for Deep Dive)")
if st.sidebar.checkbox("Show Raw Data"):
    st.write(df)

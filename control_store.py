import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame (replace this with your actual control stores dataset)
control_stores_df = pd.read_csv('C:/Users/atirkey/OneDrive/airflow/similar_control_stores.csv')

# Streamlit UI elements
st.title('Store Similarity Finder')

# Input: Enter a test store number
test_store_number = st.number_input('Enter Test Store Number:', min_value=1, value=1074)

# Step 1: Filter the dataframe based on the test store number
filtered_data = control_stores_df[control_stores_df['test_store'] == test_store_number]

# Step 2: If no matching data is found, show a warning message
if filtered_data.empty:
    st.warning(f"No data found for test store {test_store_number}.")
else:
    # Step 3: Display the filtered rows
    st.write(f"Data for Test Store {test_store_number}:")
    st.dataframe(filtered_data)

    # Step 4: Visualize the data
    # Example: Plotting the distribution of a numerical column (replace 'column_name' with actual column)
    if 'AVERAGE_BASKET_SIZE' in filtered_data.columns:  # Replace 'column_name' with an actual column name
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data['AVERAGE_BASKET_SIZE'], bins=30, kde=True)
        plt.title(f'Distribution of AVERAGE BASKET SIZE for Store {test_store_number}')
        plt.xlabel('AVERAGE BASKET SIZE')  # Replace with actual column name
        plt.ylabel('Frequency')
        st.pyplot(plt)  # Display the plot in Streamlit
        plt.clf()  # Clear the figure for the next plot

    # Example: Scatter plot of two numerical columns (replace 'column_x' and 'column_y' with actual column names)
    if 'AVERAGE_BASKET_SIZE' in filtered_data.columns and 'NO_OF_TRANSACTIONS' in filtered_data.columns:  # Replace with actual column names
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=filtered_data, x='AVERAGE_BASKET_SIZE', y='NO_OF_TRANSACTIONS', hue='STATE')  # Replace with actual column names
        plt.title(f'Scatter Plot of AVERAGE BASKET SIZE vs NO OF TRANSACTIONS for Store {test_store_number}')
        plt.xlabel('AVERAGE BASKET SIZE')  # Replace with actual column name
        plt.ylabel('NO OF TRANSACTIONS')  # Replace with actual column name
        st.pyplot(plt)  # Display the plot in Streamlit
        plt.clf()  # Clear the figure for the next plot


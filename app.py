import streamlit as st
import pandas as pd
import numpy as np

similar_stores = pd.read_csv('C:/Users/atirkey/OneDrive/airflow/similar_control_stores.csv')

st.title('Control Stores')

test_store_number = st.number_input('Enter Test Store Number:', min_value=1, value=1074)

categorical_vars = ['STATE', 'STORE_TYPE', 'AREA', 'COUNTRY']
selected_vars = st.multiselect('Select Two Categorical Variables', categorical_vars, default=['STATE', 'STORE_TYPE'])

if len(selected_vars) != 2:
    st.error("Please select exactly two categorical variables.")
else:
    test_store_data = similar_stores[similar_stores['test_store'] == test_store_number]

    if test_store_data.empty:
        st.error(f"No data found for test store {test_store_number}.")
    else:
        test_store_data = test_store_data.iloc[0]
        st.write(f"Test Store Data: {test_store_data}")

        condition = True
        for var in selected_vars:
            condition &= similar_stores[var] == test_store_data[var]

        similar_control_stores = similar_stores[condition]

        if similar_control_stores.empty:
            st.warning(f"No similar control stores found for the test store {test_store_number} based on the selected criteria.")
        else:
            st.write(f"Similar Control Stores for Test Store {test_store_number} based on {', '.join(selected_vars)}:")
        
            cols_to_display = ['STORE', 'distance', 'cluster', 'test_store'] + selected_vars
            filtered_display = similar_control_stores[cols_to_display]

            st.dataframe(filtered_display)

import pandas as pd

# Load your dataset (replace 'your_file.csv' with your actual file name)
data = pd.read_csv('C:/Users/atirkey/OneDrive - Kmart Australia Limited/Downloads/Control.csv')

# Remove rows where any of the specified columns have the value 'undefined'
columns_to_check = ['STATE', 'STORE_TYPE', 'AREA', 'COUNTRY']
cleaned_data = data[~data[columns_to_check].isin(['undefined']).any(axis=1)]

# Save the cleaned dataset to a new file
cleaned_data.to_csv('cleaned_data.csv', index=False)

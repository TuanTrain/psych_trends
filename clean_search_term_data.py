import pandas as pd
import os

# Directory where the folders for each search term are located
base_directory = './data/search_terms'

# Prepare a list to store DataFrames for each search term
search_term_dfs = []


# Iterate over each folder (each representing a search term)
for folder in os.listdir(base_directory):
    folder_path = os.path.join(base_directory, folder)

    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Initialize an empty DataFrame for this search term
        term_df = pd.DataFrame()

        # Iterate over each file in the directory
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            # Read the first row to get the year from the column name
            first_row = pd.read_csv(file_path, skiprows=2, nrows=1)
            year = first_row.columns[1].split(': (')[-1].split(')')[0]  # Extracting year from column name like 'depression: (2005)'


            # Read the CSV file, skipping the first two rows
            data = pd.read_csv(file_path, skiprows=2)

            # Rename the first column to 'State' and add 'Year' and 'Value'
            data.columns = ['State', 'Value']
            data['Year'] = year

            # Append this data to the term DataFrame
            term_df = pd.concat([term_df, data])

        term_df_pivot = term_df.pivot_table(index=['Year', 'State'], values='Value', aggfunc='first').reset_index()
        term_df_pivot.rename(columns={'Value': folder}, inplace=True)

        # Append the pivoted DataFrame to the list
        search_term_dfs.append(term_df_pivot)

# Merge all search term DataFrames
combined_data = pd.DataFrame()
for df in search_term_dfs:
    if combined_data.empty:
        combined_data = df
    else:
        combined_data = pd.merge(combined_data, df, on=['Year', 'State'], how='outer')

# Fill missing values with zeros
combined_data.fillna(0, inplace=True)

combined_data.sort_values(by=['Year', 'State'], inplace=True)

# Export the combined DataFrame to a new CSV file
output_file = './data/search_terms/combined_data.csv'  # Replace with your desired output path
combined_data.to_csv(output_file, index=False)

print(f"Combined data exported to {output_file}")
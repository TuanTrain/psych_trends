import pandas as pd

# Read the suicide rate files
suicide_data_1 = pd.read_csv("./data/suicide_1999-2020.txt", sep='\t')
suicide_data_2 = pd.read_csv("./data/suicide_2018-2021.txt", sep='\t')

# Combine the two files
combined_suicide_data = pd.concat([suicide_data_1, suicide_data_2], ignore_index=True)

# Drop any unnecessary columns or rows (e.g., headers repeated in the middle of files)
combined_suicide_data = combined_suicide_data.dropna(subset=['Year', 'State'])

# Calculate the suicide rate if not already present - some values in raw data are not calculated
combined_suicide_data['Suicide Rate'] = combined_suicide_data['Deaths'] / combined_suicide_data['Population'] * 100000

# Load the search terms data (Replace with your file path)
search_terms_data_path = './data/search_terms/combined_data.csv'  # Replace with the path to your search terms data
search_terms_data = pd.read_csv(search_terms_data_path)

# combined_suicide_data = combined_suicide_data[['State', 'Year', 'Crude Rate', 'Suicide Rate']]
combined_suicide_data = combined_suicide_data[['State', 'Year', 'Suicide Rate']]


# Merge the suicide rate data with the search terms data
merged_data = pd.merge(search_terms_data, combined_suicide_data, on=['Year', 'State'], how='left')

# remove rows without suicide rate data; remove duplicate rows, clean some values to int
merged_data = merged_data.dropna(subset=['Suicide Rate']).drop_duplicates().replace("<1", 1)

# Export the merged data to a new CSV file
output_file = './data/merged_data.csv'
merged_data.to_csv(output_file, index=False)

print(f"Merged data exported to {output_file}")
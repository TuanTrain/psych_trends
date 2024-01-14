import pandas as pd

# Load your data
data = pd.read_csv('./data/merged_data.csv')

first_term_idx=2
last_term_idx=len(data.columns)-1

search_term_columns = data.columns[first_term_idx:last_term_idx]

# Calculate the correlation of each search term with the suicide rate
correlations = data[search_term_columns].apply(lambda x: x.corr(data['Suicide Rate']))

# Create a DataFrame for the correlations
correlation_df = pd.DataFrame(correlations, columns=['Correlation'])

# Function to calculate the previous year's suicide rate only if consecutive
def calculate_prev_year_suicide_rate(group):
    group = group.sort_values('Year')
    group['Prev_Year_Suicide_Rate'] = group['Suicide Rate'].shift(1)
    group['Year_Diff'] = group['Year'].diff()
    group.loc[group['Year_Diff'] != 1, 'Prev_Year_Suicide_Rate'] = pd.NA
    return group

# Apply the function to each state's group
data = data.groupby('State').apply(calculate_prev_year_suicide_rate)

# Drop NA values in 'Prev_Year_Suicide_Rate'
data = data.dropna(subset=['Prev_Year_Suicide_Rate'])

prev_year_correlation = data['Prev_Year_Suicide_Rate'].astype(float).corr(data['Suicide Rate'])
correlation_df.loc["Prev_Year_Suicide_Rate"] = prev_year_correlation


# Sort the search terms by their correlation with the suicide rate
sorted_correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

# Display the sorted correlations
output_file = './analysis/correlation_data.csv'
sorted_correlation_df.to_csv(output_file, index=True)

print(f"Correlation data exported to {output_file}")

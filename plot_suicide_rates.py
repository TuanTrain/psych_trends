import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv('./data/merged_data.csv')

plt.figure(figsize=(24, 14))

# Plotting suicide rates for each state over the years
for state in data['State'].unique():
    state_data = data[data['State'] == state]
    plt.plot(state_data['Year'], state_data['Suicide Rate'], label=state)

# Setting up the plot with the required specifications
plt.xlabel('Year', fontsize=24)  # Increased font size for x-axis label
plt.xticks(data['Year'].unique(), fontsize=18)  # Increased font size for x-axis ticks
plt.ylabel('Suicide Rate per 100,000', fontsize=24)  # Increased font size for y-axis label
plt.yticks(fontsize=18)  # Increased font size for x-axis ticks
plt.title('Adolescent Suicide Rates Per State 2004-2021', fontsize=34)  # Increased font size for title
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(False)  # Removing grid lines
plt.subplots_adjust(right=0.8)

plt.savefig("./analysis/suicide_rates_by_state_2004_2021.png")


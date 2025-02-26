import pandas as pd
import plotly.express as px

# Load the new CSV
suicide_data = pd.read_csv("predictions.csv")

# Calculate the absolute error for baseline and model predictions
suicide_data['Baseline_Absolute_Diff'] = abs(suicide_data['Suicide Rate'] - suicide_data['Baseline_Predictions'])
suicide_data['Model_Absolute_Diff'] = abs(suicide_data['Suicide Rate'] - suicide_data['Model_Predictions'])

# Calculate the average absolute error per state
avg_abs_error_data = suicide_data.groupby('State').agg(
    Average_Baseline_Absolute_Diff=('Baseline_Absolute_Diff', 'mean'),
    Average_Model_Absolute_Diff=('Model_Absolute_Diff', 'mean'),
    Average_Suicide_Rate=('Suicide Rate', 'mean')
).reset_index()

# Add state abbreviations
state_abbreviations = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
    'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
    'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
    'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

avg_abs_error_data['State_Abbreviation'] = avg_abs_error_data['State'].map(state_abbreviations)

# Calculate the relative improvement in absolute error
avg_abs_error_data['Relative_Improvement'] = 100 * (avg_abs_error_data['Average_Baseline_Absolute_Diff'] - avg_abs_error_data['Average_Model_Absolute_Diff']) / avg_abs_error_data['Average_Suicide_Rate']

# Create the map for the relative improvement in absolute error
fig1 = px.choropleth(avg_abs_error_data, 
                     locations='State_Abbreviation', 
                     locationmode="USA-states", 
                     scope="usa", 
                     color='Relative_Improvement', 
                     color_continuous_scale=[(0, 'red'), (0.5, 'white'), (1, 'green')],
                     color_continuous_midpoint=0,
                     range_color=(-30, 30))

# Add custom tick labels with percentages
tick_vals = [-30, -20, -10, 0, 10, 20, 30]
tick_text = ['-30%', '-20%', '-10%', '0%', '+10%', '+20%', '+30%']

fig1.update_layout(
    title='Percent Improvement in Absolute Error of Suicide Rate Prediction by State',
    title_font=dict(size=38),
    title_y=0.9,
    width=1800, 
    height=1240, 
    coloraxis_colorbar=dict(
        title='',
        len=.9,
        tickvals=tick_vals,
        ticktext=tick_text,
        tickfont=dict(size=24)
    )
)

fig1.show()

fig1.write_image("relative_improvement_absolute_error.png")

# Count and list the states with positive and negative improvement
positive_improvement_states = avg_abs_error_data[avg_abs_error_data['Relative_Improvement'] > 0]['State'].tolist()
negative_improvement_states = avg_abs_error_data[avg_abs_error_data['Relative_Improvement'] <= 0]['State'].tolist()

print("States with positive improvement:", positive_improvement_states)
print("Number of states with positive improvement:", len(positive_improvement_states))
print("States with negative improvement:", negative_improvement_states)
print("Number of states with negative improvement:", len(negative_improvement_states))

# Sort the DataFrame by Relative Improvement
avg_abs_error_data_sorted = avg_abs_error_data.sort_values(by='Relative_Improvement', ascending=False)
print(avg_abs_error_data_sorted)

output_file = 'averaged_abs_err_2020_21.csv'
avg_abs_error_data_sorted.to_csv(output_file, index=False)

print(f"Correlation data exported to {output_file}")

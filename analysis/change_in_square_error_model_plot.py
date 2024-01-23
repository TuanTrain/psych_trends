import pandas as pd
import plotly.express as px

suicide_data = pd.read_csv("updated_averaged_square_difference.csv")

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

suicide_data['State_Abbreviation'] = suicide_data['State'].map(state_abbreviations)

suicide_data['Change_MSE'] =  suicide_data['Average_Model_Square_Diff'] - suicide_data['Average_Baseline_Square_Diff']


# Creating the geographic plot using Plotly Express
fig = px.choropleth(suicide_data, 
                    locations='State_Abbreviation', 
                    locationmode="USA-states", 
                    scope="usa", 
                    color='Change_MSE', 
                    color_continuous_scale=px.colors.diverging.RdBu_r,
                    color_continuous_midpoint=0,
                    range_color=(-8,8))
                    # title='Difference in Mean Square Error for Model Compared to Baseline')


# fig.update(layout_coloraxis_showscale=False)
fig.update_layout(width=1800, 
                  height=1240, 
                  coloraxis_colorbar=dict(
                      title='', 
                      len=.9,
                      # title_font=dict(size=18),  # Adjust title font size
                      tickfont=dict(size=24)     # Adjust tick labels font size
                  ))
fig.show()

fig.write_image("model_square_error_improvement.png") 
# fig.write_image("model_square_error_improvement.svg", width=1800, height=1240) 
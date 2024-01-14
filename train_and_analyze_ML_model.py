import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the merged data
file_path = './data/merged_data.csv'
data = pd.read_csv(file_path)

# get indices of the search terms
first_term_idx = 2
last_term_idx = len(data.columns) - 1

# Sort data by State and Year to ensure correct alignment
data.sort_values(by=['State', 'Year'], inplace=True)

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

# Filter the columns to include only the search terms, the suicide rate, and the previous year's suicide rate
X = data.iloc[:, first_term_idx:last_term_idx]
X['Prev_Year_Suicide_Rate'] = data['Prev_Year_Suicide_Rate']

y = data['Suicide Rate']  # Suicide rate column

# Splitting the data into training and validation sets
# Training data: 2004 to 2019, Validation data: 2020 and 2021
X_train = X[data['Year'] <= 2019]
y_train = y[data['Year'] <= 2019]
X_val = X[data['Year'] >= 2020]
y_val = y[data['Year'] >= 2020]

# Creating and training the Random Forest model
model = RandomForestRegressor(n_estimators=110, random_state=420)
model.fit(X_train, y_train)

# Predicting on the validation set
predictions = model.predict(X_val)

# Compare the model to the baseline:
# The baseline prediction is assuming the rate stays the same year to year
# Calculate the predictions of the baseline model (previous year's rate)
baseline_predictions = data['Prev_Year_Suicide_Rate'][data['Year'] >= 2020]

# Calculate the actual suicide rates for the validation set
actual_rates = data['Suicide Rate'][data['Year'] >= 2020]

# Calculate MSE and MAE for the model
model_mse = mean_squared_error(actual_rates, predictions)
model_mae = mean_absolute_error(actual_rates, predictions)

# Calculate MSE and MAE for the baseline model
baseline_mse = mean_squared_error(actual_rates, baseline_predictions)
baseline_mae = mean_absolute_error(actual_rates, baseline_predictions)

# Print the results
print(f"Model MSE: {model_mse}, Baseline MSE: {baseline_mse}")
print(f"Model MAE: {model_mae}, Baseline MAE: {baseline_mae}")

# Calculate the improvement
mse_improvement = (baseline_mse - model_mse) / baseline_mse
mae_improvement = (baseline_mae - model_mae) / baseline_mae

print(f"MSE Improvement: {mse_improvement * 100:.2f}%")
print(f"MAE Improvement: {mae_improvement * 100:.2f}%")

# Analysis #1
# save a predictions table
validation_data = data[data['Year'] >= 2020][:]

# Add baseline and model predictions to the validation data
validation_data['Baseline_Predictions'] = validation_data['Prev_Year_Suicide_Rate']
validation_data['Model_Predictions'] = predictions

# Select the relevant columns
result_table = validation_data[['Year', 'State', 'Suicide Rate', 'Baseline_Predictions', 'Model_Predictions']]

# Save the table to a CSV file
output_file = './analysis/predictions.csv'
result_table.to_csv(output_file, index=False)
print(f"Prediction data exported to {output_file}")


# Analysis #2
# Extracting feature importances
feature_importances = model.feature_importances_

# Creating a DataFrame to hold feature names and their importance scores
features = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sorting the features based on importance
features = features.sort_values(by='Importance', ascending=False)

# Displaying the feature importances
output_file = './analysis/feature_importance_data.csv'
features.to_csv(output_file, index=False)

print(f"ML model feature importance data exported to {output_file}")


# Plotting feature importances for better visualization
plt.figure(figsize=(12, 8))
plt.barh(features['Feature'], features['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances in Random Forest Model')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.subplots_adjust(left=0.2)
plt.savefig("./analysis/model_feature_importance.png")




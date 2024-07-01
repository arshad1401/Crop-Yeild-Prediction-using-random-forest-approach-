import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets with only the first 2000 rows
yield_data = pd.read_csv("yield.csv").head(2000)
temp_data = pd.read_csv("temp.csv").head(2000)
rainfall_data = pd.read_csv("rainfall.csv").head(2000)

# Combine temperature and rainfall datasets
weather_data = pd.merge(temp_data, rainfall_data, on=[' Year', ' Country', ' ISO3'], how='inner')

# Merge weather data with yield data based on 'Year'
merged_df = pd.merge(yield_data, weather_data, left_on='Year', right_on=' Year', how='inner')

# Drop redundant columns
merged_df.drop([' Year', ' Country', ' ISO3'], axis=1, inplace=True)

# Feature selection
X = merged_df[['Temperature - (Celsius)', 'Rainfall - (MM)']]
y = merged_df['Value']  # Total yield per hectare

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor for feature selection
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Selecting important features
important_features = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
selected_features = important_features[:1].index.tolist()  # Select top feature

# Use selected features
X_selected = X[selected_features]

# Split data into training and testing sets using selected features
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train the linear regression model with selected features
model = LinearRegression()
model.fit(X_train_selected, y_train)

# Make predictions
y_pred = model.predict(X_test_selected)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# Plot predicted vs. actual yield
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title('Actual vs. Predicted Crop Yield')
plt.legend()
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X_selected.columns, 'Importance': model.coef_})
print("Feature Importance:")
print(feature_importance)

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Analysis')
plt.show()

# Seasonal Analysis
plt.figure(figsize=(10, 6))
sns.lineplot(data=merged_df, x='Year', y='Value', estimator=np.mean, errorbar=None)
plt.xlabel('Year')
plt.ylabel('Average Total Yield per Hectare')
plt.title('Seasonal Analysis of Crop Yield')
plt.show()

# Sensitivity Analysis
sensitivity_features = selected_features
for feature in sensitivity_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=merged_df, x=feature, y='Value')
    plt.xlabel(feature)
    plt.ylabel('Total Yield per Hectare')
    plt.title(f'Sensitivity Analysis: {feature} vs. Total Yield per Hectare')
    plt.show()

# Prediction for a specific area (e.g., Afghanistan in 2024)
# Assume 'temperature' and 'rainfall' are the temperature (in Celsius) and rainfall (in mm) for the area you want to predict
temperature = 25.0  # Example temperature
rainfall = 100.0  # Example rainfall

# Create a DataFrame with the features for the area
area_features = pd.DataFrame({'Temperature - (Celsius)': [temperature], 'Rainfall - (MM)': [rainfall]})

# Use only the selected feature for prediction
area_features_selected = area_features[selected_features]

# Use the trained model to make predictions for the area
predicted_yield = model.predict(area_features_selected)

# Print the predicted number of crops
print("Predicted total yield per hectare for the specified area:", predicted_yield[0])

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the Excel file
file_path = 'Bacteria.xlsx'
data = pd.read_excel(file_path)

# Step 2: Extract the chemical concentrations and yield
concentrations = data.iloc[:, :-1]  # Assuming all columns except the last are chemical concentrations
yield_column = data.iloc[:, -1]     # Assuming the last column is the 'Yield'

# Step 3: Normalize the chemical concentrations
scaler = MinMaxScaler()
normalized_concentrations = pd.DataFrame(scaler.fit_transform(concentrations), columns=concentrations.columns)

# Step 4: Combine normalized concentrations and yield into one DataFrame
normalized_data = pd.concat([normalized_concentrations, yield_column], axis=1)

# Step 5: Split the data into training and testing sets
X = normalized_data.drop('Yield', axis=1)  # Features: all columns except the 'Yield'
y = normalized_data['Yield']  # Target: 'Yield'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize XGBoost regressor
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)

# Step 7: Train the XGBoost model
xgboost_model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred_train_xgb = xgboost_model.predict(X_train)
y_pred_test_xgb = xgboost_model.predict(X_test)

# Step 9: Evaluate the model
train_mse_xgb = mean_squared_error(y_train, y_pred_train_xgb)
test_mse_xgb = mean_squared_error(y_test, y_pred_test_xgb)
train_r2_xgb = r2_score(y_train, y_pred_train_xgb)
test_r2_xgb = r2_score(y_test, y_pred_test_xgb)

# Step 10: Display the results
print(f"XGBoost Training MSE: {train_mse_xgb}")
print(f"XGBoost Testing MSE: {test_mse_xgb}")
print(f"XGBoost Training R²: {train_r2_xgb}")
print(f"XGBoost Testing R²: {test_r2_xgb}")

# Step 11: Optimize the key features based on previous analysis (x8, x7, x10, x3)
important_features = ['x8', 'x7', 'x10', 'x3']

# Generate a grid of values for the important features
feature_grid = {
   'x8': np.linspace(0, 1, 10),  # 10 evenly spaced values between 0 and 1
   'x7': np.linspace(0, 1, 10),
   'x10': np.linspace(0, 1, 10),
   'x3': np.linspace(0, 1, 10)
}

# Generate all possible combinations of the grid values
combinations = list(itertools.product(
   feature_grid['x8'], 
   feature_grid['x7'], 
   feature_grid['x10'], 
   feature_grid['x3']
))

# Convert combinations to a DataFrame
optimization_df = pd.DataFrame(combinations, columns=important_features)

# Set the remaining features (x1, x2, x4, x5, x6, x9) to their mean value from the normalized dataset
for feature in ['x1', 'x2', 'x4', 'x5', 'x6', 'x9']:
   optimization_df[feature] = normalized_concentrations[feature].mean()

# Reorder the columns to match the original feature order used during model training
feature_order = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
optimization_df = optimization_df[feature_order]

# Step 12: Predict the yield for each combination using the trained XGBoost model
predicted_yields = xgboost_model.predict(optimization_df)

# Add the predicted yields to the DataFrame
optimization_df['Predicted_Yield'] = predicted_yields

# Find the combination of feature values that maximizes the predicted yield
best_combination = optimization_df.loc[optimization_df['Predicted_Yield'].idxmax()]
print("Best combination of feature values for maximum yield:")
print(best_combination)

# Output the maximum predicted yield
max_yield = best_combination['Predicted_Yield']
print(f"Maximum predicted yield: {max_yield}")

# Step 13: Plot feature importances from XGBoost model
xgb_feature_importances = xgboost_model.feature_importances_

# Create a DataFrame to store the feature names and their importance scores
xgb_feature_importance_df = pd.DataFrame({
   'Feature': X.columns,  # Feature names from the original dataset
   'Importance': xgb_feature_importances  # Importance scores from the XGBoost model
}).sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(8, 6))
plt.barh(xgb_feature_importance_df['Feature'], xgb_feature_importance_df['Importance'], color='skyblue')

# Increase y-axis tick label size
plt.yticks(fontsize=12, fontweight='bold')  # Adjust the font size as needed

# Make the title bold and increase its size
plt.xlabel('Importance', fontsize=12, fontweight='bold')  # You can also increase this size if needed
plt.title('Feature Importances from XGBoost', fontsize=14, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
# Invert y-axis to show the most important features at the top
plt.gca().invert_yaxis()

plt.show()
import os # For file path operations
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For plotting
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.linear_model import LinearRegression # For linear regression model
from sklearn.metrics import mean_absolute_error, r2_score # For model evaluation

# Load dataset (adjust path if needed)
data = pd.read_csv('data/co2_emissions_kt_by_country.csv')

# Display basic info
print("Columns:", data.columns)
print(data.head())

# Keep only relevant columns
data = data[['year', 'value']].dropna()

# Rename columns for consistency
data = data.rename(columns={'year': 'Year', 'value': 'Emissions'})

# Convert Year to integer (in case it isn’t)
data['Year'] = data['Year'].astype(int)

# Define features (X) and target (y)
X = data[['Year']]
y = data['Emissions']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Ensure folder exists for saving plots
os.makedirs('screenshots', exist_ok=True)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
sorted_idx = X_test['Year'].argsort()
plt.plot(X_test['Year'].values[sorted_idx], y_pred[sorted_idx], color='red', label='Predicted')
plt.title('CO₂ Emissions Forecast (SDG 13: Climate Action)')
plt.xlabel('Year')
plt.ylabel('CO₂ Emissions (kt)')
plt.legend()
plt.savefig('screenshots/results.png')
plt.show()

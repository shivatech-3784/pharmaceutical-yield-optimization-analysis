from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Prepare Data for Modeling ---
X = df.drop('Yield_Percent', axis=1) # Features
y = df['Yield_Percent']               # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Build and Train the Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 3. Make Predictions ---
predictions = model.predict(X_test)

# --- 4. Evaluate Model Performance ---
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model Mean Absolute Error (MAE): {mae:.2f}%")
print(f"Model R-squared (R²): {r2:.2f}")

# --- 5. Visualize Predictions vs. Actuals ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Yield (%)')
plt.ylabel('Predicted Yield (%)')
plt.title('Actual vs. Predicted Yield')
plt.show()
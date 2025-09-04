# --- 1. Get Feature Importances ---
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

# --- 2. Visualize Feature Importances ---
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index)
plt.title('Feature Importance for Yield Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Process Variable')
plt.show()

# --- 3. Conclusion & Business Impact ---
# In a text cell, write a summary based on the results.
# Example:
# "The model identified pH, Temperature, and Catalyst Concentration as the most significant drivers of batch yield. This suggests that a focus on maintaining tight control over these parameters can directly lead to higher product output and reduced waste. A tool based on this model could be used by chemical engineers to set optimal process parameters before starting a new batch."
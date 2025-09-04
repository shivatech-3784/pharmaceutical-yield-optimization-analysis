import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Basic Data Info ---
print(df.info())
print(df.describe())

# --- 2. Correlation Heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Process Parameters and Yield')
plt.show()

# --- 3. Scatter Plots of Key Variables vs. Yield ---
# Use the heatmap to identify features with high correlation to yield
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x='Temperature_C', y='Yield_Percent', data=df)
plt.title('Yield vs. Temperature')

plt.subplot(1, 3, 2)
sns.scatterplot(x='pH', y='Yield_Percent', data=df)
plt.title('Yield vs. pH')

plt.subplot(1, 3, 3)
sns.scatterplot(x='Catalyst_Concentration', y='Yield_Percent', data=df)
plt.title('Yield vs. Catalyst Concentration')

plt.tight_layout()
plt.show()
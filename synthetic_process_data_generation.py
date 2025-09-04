import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Define Process Variables (Input Features) ---
n_batches = 1000

# Create simulated data for key process parameters
temperature = np.random.uniform(90, 110, n_batches) # °C
pressure = np.random.uniform(5, 10, n_batches)      # bar
ph = np.random.uniform(6.5, 8.5, n_batches)         # pH
stirring_speed = np.random.randint(50, 150, n_batches) # RPM
catalyst_concentration = np.random.uniform(0.1, 0.5, n_batches) # % weight

# --- 2. Define the 'Ground Truth' Yield (The Target Variable) ---
# Create a yield function with a lower average and a wider range
yield_noise = np.random.normal(0, 2, n_batches) # Gaussian noise

yield_percent = (
    70                                    # Start with a lower base yield
    + 0.5 * (temperature - 100)           # Temp effect, centered around 100
    - 4 * (ph - 7.5)**2                   # Strong non-linear effect of pH
    + 0.1 * stirring_speed
    + 20 * catalyst_concentration
    + yield_noise
)

# Use a wider, more realistic clipping range
yield_percent = np.clip(yield_percent, 65, 95)

# --- 3. Create the DataFrame ---
df = pd.DataFrame({
    'Temperature_C': temperature,
    'Pressure_bar': pressure,
    'pH': ph,
    'Stirring_Speed_RPM': stirring_speed,
    'Catalyst_Concentration': catalyst_concentration,
    'Yield_Percent': yield_percent
})

# Display the first few rows of your synthetic dataset
print(df.head())
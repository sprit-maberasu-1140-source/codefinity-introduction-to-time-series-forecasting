import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
flights = sns.load_dataset("flights").copy()

# Ensure 'month' is treated as a string (not categorical)
flights["month"] = flights["month"].astype(str)

# Convert to datetime and set index
flights["date"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"])
flights.set_index("date", inplace=True)

# Extract target series
series = flights["passengers"]

# Perform decomposition
decomposition = seasonal_decompose(series, model="additive", period=12)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
axes[0].plot(series, label="Original")
axes[1].plot(decomposition.trend, label="Trend")
axes[2].plot(decomposition.seasonal, label="Seasonal")
axes[3].plot(decomposition.resid, label="Residual")

for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
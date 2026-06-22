import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

# 1. Define AR and MA parameters
ar = np.array([1, -0.75, 0.25])
ma = np.array([1, 0.65])

# 2. Initialize ARMA process
arma_process = ArmaProcess(ar, ma)

# 3. Generate simulated data
simulated_data = arma_process.generate_sample(nsample=500, scale=1.0)

# 4. Plot the simulated time series
plt.figure(figsize=(10, 4))
plt.plot(simulated_data)
plt.title("Simulated ARIMA(2,0,1) Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.show()

# 5. Display first 10 values
print("First 10 simulated values:")
print(np.round(simulated_data[:10], 2))
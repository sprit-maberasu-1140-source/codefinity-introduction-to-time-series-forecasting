import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load and prepare dataset
flights = sns.load_dataset("flights").copy()
flights["month"] = flights["month"].astype(str)
flights["date"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"])
flights.set_index("date", inplace=True)
series = flights["passengers"]

# 2. Split into train/test
train = series.iloc[:-12]
test = series.iloc[-12:]

# 3. Fit ARIMA model
model = ARIMA(train, order=(2, 1, 2))
fitted_model = model.fit()

# 4. Forecast next 12 months
forecast = fitted_model.forecast(steps=12)

# 5. Evaluate forecasts
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 6. Plot actual vs forecast
plt.figure(figsize=(10, 5))
plt.plot(series, label="Actual")
plt.plot(test.index, forecast, label="Forecast", color="red")
plt.legend()
plt.title("ARIMA Forecast vs Actual")
plt.show()
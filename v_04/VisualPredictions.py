import pandas as pd
import matplotlib.pyplot as plt

# مسیر فایل داده‌های پیش‌بینی‌شده
predictions_data_path = "C:/AAA/predictions.csv"

# بارگذاری داده‌های پیش‌بینی‌شده
predictions_data = pd.read_csv(predictions_data_path, header=None)

# رسم نمودار پیش‌بینی‌ها
plt.figure(figsize=(12, 6))
plt.plot(predictions_data.index, predictions_data[0], label="Predicted EUR/USD Prices", color='blue')

plt.title("Predicted EUR/USD Exchange Rates")
plt.xlabel("Time (Data Points)")
plt.ylabel("Predicted Exchange Rate")
plt.grid(True)
plt.legend()

# نمایش نمودار
plt.show()

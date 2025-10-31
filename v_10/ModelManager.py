import pandas as pd
import numpy as np
import os
from tensorflow import keras 
import tensorflow


class ModelManager:
    def __init__(self, model_path: str, time_steps: int, num_features: int):
        self.model_path = model_path
        self.time_steps = time_steps
        self.num_features = num_features
        self.model = None

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found")

    def load_data_from_csv(self, csv_path: str):
        """
        بارگذاری داده‌ها از فایل CSV و تبدیل آنها به فرمت مناسب برای مدل.
        """
        if os.path.exists(csv_path):
            # بارگذاری داده‌ها از CSV با جداکننده‌های مختلف (کاما یا تب)
            try:
                # ابتدا با جداکننده کاما
                data = pd.read_csv(csv_path, sep=',', header=None)
                if len(data.columns) == 1:
                    # اگر فقط یک ستون شناسایی شد، احتمالا فایل با تب جدا شده است
                    data = pd.read_csv(csv_path, sep='\t', header=None)
            except Exception as e:
                raise ValueError(f"Error loading CSV file: {e}")
            
            # بررسی اینکه آیا تعداد ستون‌ها با انتظار شما سازگار است
            if data.shape[1] != 6:
                raise ValueError(f"Expected 6 columns, but found {data.shape[1]} columns in the data.")
            
            # تنظیم نام ستون‌ها
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # تبدیل مقادیر به نوع float (به جز ستون Date)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')  # مقادیر نامعتبر را به NaN تبدیل می‌کند

            # حذف ردیف‌هایی که مقادیر NaN دارند
            data = data.dropna()

            # فقط ستون‌های عددی را انتخاب کنید
            numerical_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            # ایجاد داده‌های سری زمانی با استفاده از تعداد time_steps
            sequences = []
            for i in range(len(numerical_data) - self.time_steps + 1):
                sequences.append(numerical_data[i:i + self.time_steps])
            
            sequences = np.array(sequences)
            
            print(f"Data loaded and reshaped from {csv_path}")
            return sequences
        else:
            raise FileNotFoundError(f"CSV file {csv_path} not found")


    def predict(self, input_data: np.ndarray):
        if self.model is None:
            raise ValueError("Model not loaded. Call `load_model()` first.")
        
        predictions = self.model.predict(input_data)
        return predictions


    def save_predictions(self, predictions: np.ndarray, file_path: str):
        # تبدیل فرمت علمی به فرمت اعشاری معمولی
        np.savetxt(file_path, predictions, delimiter=',', fmt='%.8f')  # 8 رقم اعشار
        print(f"Predictions saved to {file_path}")


# مثال استفاده از کلاس
if __name__ == "__main__":
    time_steps = 672  # تنظیم تعداد time_steps
    num_features = 5  # ویژگی‌های Open, High, Low, Close, Volume

    model_manager = ModelManager("C:/AAA/CNN_BiLSTM_Attention_Model.h5", time_steps, num_features)
    model_manager.load_model()

    # بارگذاری داده‌های CSV
    input_data = model_manager.load_data_from_csv("C:/AAA/csv/EURUSD_Candlestick_1_Hour_ASK_01.01.2015-28.09.2024.csv")

    # انجام پیش‌بینی‌ها
    predictions = model_manager.predict(input_data)

    # ذخیره نتایج پیش‌بینی‌ها
    model_manager.save_predictions(predictions, "C:/AAA/predictions.csv")

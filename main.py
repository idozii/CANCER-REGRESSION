import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

# Load dữ liệu
cancer_df = pd.read_csv("data/cancer_reg.csv")
avg_household_df = pd.read_csv("data/avg-household-size.csv")

# Merge dữ liệu dựa trên cột chung (giả sử là 'countyfips')
df = cancer_df.merge(avg_household_df, on="countyfips", how="left")

# Xử lý giá trị NaN
df.fillna(df.mean(numeric_only=True), inplace=True)

# Xác định feature và target
feature_cols = df.select_dtypes(include=['number']).columns.drop("target_deathrate")
X = df[feature_cols]
y = df["target_deathrate"]

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo các mô hình
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Huấn luyện và đánh giá từng mô hình
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = mae

# In kết quả
for model, mae in results.items():
    print(f"MAE ({model}): {mae:.2f}")

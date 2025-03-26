import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import time
import os

os.makedirs("figure", exist_ok=True)

def remove_outliers(df, columns, threshold=2.5):
    df_clean = df.copy()
    outliers_removed = 0
    
    for col in columns:
        if df_clean[col].dtype in ['int64', 'float64']:
            z_scores = stats.zscore(df_clean[col], nan_policy='omit')
            outlier_mask = abs(z_scores) > threshold
            outliers_in_col = sum(outlier_mask)
            if outliers_in_col > 0:
                print(f"Removed {outliers_in_col} outliers from {col}")
                outliers_removed += outliers_in_col
            df_clean = df_clean[~outlier_mask]
    
    print(f"Total outliers removed: {outliers_removed}")
    print(f"Rows before outlier removal: {len(df)}")
    print(f"Rows after outlier removal: {len(df_clean)}")
    return df_clean

def create_interaction_terms(X):
    print("Creating interaction terms between top features...")
    feature_pairs = []
    top_5_features = list(X.columns)[:5] if len(X.columns) > 5 else list(X.columns)
    
    for i, f1 in enumerate(top_5_features):
        for f2 in top_5_features[i+1:]:
            feature_name = f"{f1}_x_{f2}"
            X[feature_name] = X[f1] * X[f2]
            feature_pairs.append(feature_name)
    
    for feature in top_5_features:
        feature_name = f"{feature}_squared"
        X[feature_name] = X[feature] ** 2
        feature_pairs.append(feature_name)
    
    print(f"Created {len(feature_pairs)} new features through interactions")
    return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    def print_gpu_utilization():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print_gpu_utilization()

sns.set_style('whitegrid')
sns.set_palette('colorblind')

avghouseholdsize_data = pd.read_csv('data/avg-household-size.csv')
cancereg_data = pd.read_csv('data/cancer_reg.csv')

numeric_cols = cancereg_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if numeric_cols:
    numeric_imputer = SimpleImputer(strategy='median')
    cancereg_data[numeric_cols] = numeric_imputer.fit_transform(cancereg_data[numeric_cols])

object_cols = cancereg_data.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cancereg_data[object_cols] = cat_imputer.fit_transform(cancereg_data[object_cols])

merged_data = pd.merge(avghouseholdsize_data, cancereg_data, on='geography', how='inner')
all_features = numeric_cols + object_cols
if 'target_deathrate' in all_features:
    all_features.remove('target_deathrate')

print("Applying outlier removal...")
cleaned_features = all_features.copy()
if 'geography' in cleaned_features:
    cleaned_features.remove('geography')

merged_data_clean = remove_outliers(merged_data, cleaned_features + ['target_deathrate'], threshold=2.5)

X = merged_data_clean[all_features]
if object_cols:
    X = pd.get_dummies(X, columns=object_cols, drop_first=True)
y = merged_data_clean['target_deathrate']

print("Performing feature selection...")
rf_for_selection = RandomForestRegressor(n_estimators=200, random_state=42)
rf_for_selection.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_for_selection.feature_importances_
}).sort_values('Importance', ascending=False)

top_features = feature_importance.head(20)['Feature']
X_top = X[top_features]

X_top_with_interactions = create_interaction_terms(X_top.copy())

X_train, X_test, y_train, y_test = train_test_split(
    X_top_with_interactions, y, test_size=0.2, random_state=42)

print(f"Training data shape after feature engineering: {X_train.shape}")
print(f"Testing data shape after feature engineering: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

class ImprovedNN(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.1)
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.1)
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.1)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = torch.nn.functional.leaky_relu(self.fc4(x), negative_slope=0.1)
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = self.fc5(x)
        return x

input_size = X_train_scaled.shape[1]
model = ImprovedNN(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

if torch.cuda.is_available():
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")
    print_gpu_utilization()

num_epochs = 1500
early_stop_patience = 30
best_loss = float('inf')
early_stop_counter = 0

train_losses, val_losses = [], []
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)

    if epoch % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_counter = 0
        best_model_state = model.state_dict()
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered")
            break

end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")

model.load_state_dict(best_model_state)

model.eval()
y_pred_list = []
y_actual_list = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        y_pred = model(batch_X)
        y_pred_list.append(y_pred.cpu().numpy())
        y_actual_list.append(batch_y.cpu().numpy())

y_pred_list = np.vstack(y_pred_list).flatten()
y_actual_list = np.vstack(y_actual_list).flatten()

mae_pytorch = mean_absolute_error(y_actual_list, y_pred_list)
mse_pytorch = mean_squared_error(y_actual_list, y_pred_list)
rmse_pytorch = np.sqrt(mse_pytorch)

print("\nImproved PyTorch Neural Network Regressor")
print(f'Mean Absolute Error: {mae_pytorch:.4f}')
print(f'Mean Squared Error: {mse_pytorch:.4f}')
print(f'Root Mean Squared Error: {rmse_pytorch:.4f}')

plt.figure(figsize=(10, 6))
plt.scatter(y_actual_list, y_pred_list, alpha=0.5)
plt.plot([min(y_actual_list), max(y_actual_list)], [min(y_actual_list), max(y_actual_list)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs Actual')
plt.savefig("figure/prediction_vs_actual.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig("figure/training_loss_plot.png")
plt.close()

residuals = y_actual_list - y_pred_list
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_list, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.savefig("figure/residual_plot.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.savefig("figure/residual_distribution.png")
plt.close()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nTraining a Gradient Boosting Regressor for comparison...")
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

mae_gb = mean_absolute_error(y_test, gb_preds)
mse_gb = mean_squared_error(y_test, gb_preds)
rmse_gb = np.sqrt(mse_gb)

print("Gradient Boosting Regressor")
print(f'Mean Absolute Error: {mae_gb:.4f}')
print(f'Mean Squared Error: {mse_gb:.4f}')
print(f'Root Mean Squared Error: {rmse_gb:.4f}')

print("\nTraining additional models for comparison...")

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, lr_preds)
mse_lr = mean_squared_error(y_test, lr_preds)
rmse_lr = np.sqrt(mse_lr)

print("Linear Regression")
print(f'Mean Absolute Error: {mae_lr:.4f}')
print(f'Mean Squared Error: {mse_lr:.4f}')
print(f'Root Mean Squared Error: {rmse_lr:.4f}')

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
mae_dt = mean_absolute_error(y_test, dt_preds)
mse_dt = mean_squared_error(y_test, dt_preds)
rmse_dt = np.sqrt(mse_dt)

print("Decision Tree Regressor")
print(f'Mean Absolute Error: {mae_dt:.4f}')
print(f'Mean Squared Error: {mse_dt:.4f}')
print(f'Root Mean Squared Error: {rmse_dt:.4f}')

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, rf_preds)
mse_rf = mean_squared_error(y_test, rf_preds)
rmse_rf = np.sqrt(mse_rf)

print("Random Forest Regressor")
print(f'Mean Absolute Error: {mae_rf:.4f}')
print(f'Mean Squared Error: {mse_rf:.4f}')
print(f'Root Mean Squared Error: {rmse_rf:.4f}')

models_data = {
    'Model': ['Gradient Boosting', 'Random Forest Regressor', 'Linear Regression', 
              'Neural Network Regressor', 'Decision Tree Regressor'],
    'Mean Absolute Error': [mae_gb, mae_rf, mae_lr, mae_pytorch, mae_dt],
    'Mean Squared Error': [mse_gb, mse_rf, mse_lr, mse_pytorch, mse_dt],
    'RMSE': [rmse_gb, rmse_rf, rmse_lr, rmse_pytorch, rmse_dt]
}

models_df = pd.DataFrame(models_data)
models_df = models_df.sort_values('Mean Squared Error')
models_df['Mean Absolute Error'] = models_df['Mean Absolute Error'].round(3)
models_df['Mean Squared Error'] = models_df['Mean Squared Error'].round(3)
models_df['RMSE'] = models_df['RMSE'].round(3)

print("\nModel Comparison:")
print(models_df)

models_df.to_csv("model_comparison_results.csv", index=False)

plt.figure(figsize=(14, 8))

plt.subplot(1, 3, 1)
bars = plt.bar(models_df['Model'], models_df['Mean Squared Error'])
plt.xticks(rotation=45, ha='right')
plt.title('Mean Squared Error by Model (Lower is Better)')
plt.ylabel('MSE')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{height:.2f}', ha='center', va='bottom')

plt.subplot(1, 3, 3)
bars = plt.bar(models_df['Model'], models_df['Mean Absolute Error'])
plt.xticks(rotation=45, ha='right')
plt.title('Mean Absolute Error by Model (Lower is Better)')
plt.ylabel('MAE')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("figure/Figure_7.png")
plt.close()

print("\nTable for README.md:")
print("| Model | Mean Absolute Error | Mean Squared Error | R² Score | RMSE |")
print("|-------|---------------------|--------------------|--------------------|---------|")
for _, row in models_df.iterrows():
    print(f"| {row['Model']} | {row['Mean Absolute Error']} | {row['Mean Squared Error']} |")

plt.figure(figsize=(12, 8))
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Gradient Boosting Feature Importance')
plt.tight_layout()
plt.savefig("figure/Figure_6.png")
plt.close()
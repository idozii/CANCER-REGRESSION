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
from sklearn.ensemble import RandomForestRegressor
import time
import os

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    
    def print_gpu_utilization():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print_gpu_utilization()
    
    torch.cuda.synchronize()

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

X = merged_data[all_features]
if object_cols:
    X = pd.get_dummies(X, columns=object_cols, drop_first=True)
y = merged_data['target_deathrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_for_selection = RandomForestRegressor(random_state=42)
rf_for_selection.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_for_selection.feature_importances_
}).sort_values('Importance', ascending=False)
top_features = feature_importance.head(20)['Feature']

scaler = StandardScaler()
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]
X_train_top_scaled = scaler.fit_transform(X_train_top)
X_test_top_scaled = scaler.transform(X_test_top)

X_train_tensor = torch.tensor(X_train_top_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_top_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

print(f"Input tensor device: {X_train_tensor.device}")
print(f"Target tensor device: {y_train_tensor.device}")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

input_size = X_train_top_scaled.shape[1]
model = NeuralNetwork(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")
    print_gpu_utilization()

num_epochs = 1000
early_stop_patience = 20
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

    if epoch % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if torch.cuda.is_available():
            print_gpu_utilization()
    
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
r2_pytorch = r2_score(y_actual_list, y_pred_list)

print("PyTorch Neural Network Regressor")
print(f'Mean Absolute Error: {mae_pytorch}')
print(f'Mean Squared Error: {mse_pytorch}')
print(f'R2 Score: {r2_pytorch}')

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.savefig("training_loss_plot.png")

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("Final GPU stats:")
    print_gpu_utilization()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats

cancer_regData = pd.read_csv('data/cancer_reg.csv')
avg_household_sizeData = pd.read_csv('data/avg-household-size.csv')

cancer_regData['geography_clean'] = cancer_regData['geography'].str.strip()
avg_household_sizeData['geography_clean'] = avg_household_sizeData['geography'].str.strip()

merged_df = pd.merge(
    cancer_regData,
    avg_household_sizeData[['geography_clean']],
    on='geography_clean',
    how='left'
)

merged_df.drop('pctsomecol18_24', inplace=True, axis=1)
for col in ['pctemployed16_over', 'pctprivatecoveragealone']:
    if col in merged_df.columns and merged_df[col].isnull().sum() > 0:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

object_columns = merged_df.select_dtypes(include=['object']).columns.tolist()
columns_to_drop = [col for col in object_columns if col != 'binnedinc']
merged_df = merged_df.drop(columns=columns_to_drop)

if 'binnedinc' in merged_df.columns:
    # Fix the regex warning
    merged_df['income_category'] = merged_df['binnedinc'].str.extract(r'(\d+)').astype(float)
    merged_df = merged_df.drop(columns=['binnedinc'])

numeric_df = merged_df.select_dtypes(include=[np.number])
numeric_df = numeric_df.fillna(numeric_df.median())

def remove_outliers_iqr(df, columns, multiplier=1.5):
    """More robust outlier removal using IQR"""
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    print(f"Removed {initial_rows - len(df_clean)} outliers using IQR method")
    return df_clean

# Only remove outliers from key variables
outlier_columns = ['target_deathrate', 'incidencerate', 'avganncount']
numeric_df_clean = remove_outliers_iqr(numeric_df, outlier_columns)

X = numeric_df_clean.drop('target_deathrate', axis=1)
y = numeric_df_clean['target_deathrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data for linear models only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models with appropriate data
models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'X_train': X_train_scaled,
        'X_test': X_test_scaled
    },
    'RandomForest': {
        'model': RandomForestRegressor(n_estimators=200, random_state=42),
        'X_train': X_train,  # Use unscaled data
        'X_test': X_test     # Use unscaled data
    }
}

results = {}

print("Model Comparison:")
print("-" * 60)
for name, model_config in models.items():
    model = model_config['model']
    X_train_data = model_config['X_train']
    X_test_data = model_config['X_test']
    
    # Train model
    model.fit(X_train_data, y_train)
    
    # Predict
    y_pred = model.predict(X_test_data)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R²': r2, 'MAE': mae, 'y_pred': y_pred}
    print(f"{name:20} - MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")

# Find best model for plotting
best_model_name = max(results, key=lambda x: results[x]['R²'])
y_pred_best = results[best_model_name]['y_pred']

print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['R²']:.4f}")

# Plot actual vs predicted for best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted Values - {best_model_name}')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Residual analysis for best model
residuals = y_test - y_pred_best
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'Residuals vs Predicted Values - {best_model_name}')

plt.subplot(1, 2, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot')

plt.tight_layout()
plt.savefig('optimized_diagnostics.png')
plt.close()

# Feature importance for RandomForest
if 'RandomForest' in models:
    rf_model = models['RandomForest']['model']
    if hasattr(rf_model, 'feature_importances_'):
        importance = rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nRandomForest Feature Importance:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('RandomForest Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

print("\nDiagnostic plots saved: 'actual_vs_predicted.png' and 'optimized_diagnostics.png'")
if 'RandomForest' in models:
    print("Feature importance plot saved: 'feature_importance.png'")
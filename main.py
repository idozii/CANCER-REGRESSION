import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
    avg_household_sizeData[['geography_clean', 'avghouseholdsize']],
    on='geography_clean',
    how='left'
)

for col in ['pctsomecol18_24', 'pctemployed16_over', 'pctprivatecoveragealone', 'avghouseholdsize']:
    if col in merged_df.columns and merged_df[col].isnull().sum() > 0:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

object_columns = merged_df.select_dtypes(include=['object']).columns.tolist()
columns_to_drop = [col for col in object_columns if col != 'binnedinc']
merged_df = merged_df.drop(columns=columns_to_drop)

if 'binnedinc' in merged_df.columns:
    merged_df['income_category'] = merged_df['binnedinc'].str.extract('(\d+)').astype(float)
    merged_df = merged_df.drop(columns=['binnedinc'])

numeric_df = merged_df.select_dtypes(include=[np.number])
numeric_df = numeric_df.fillna(numeric_df.median())

def remove_outliers(df, columns=None, threshold=3):
    df_clean = df.copy()
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns
    
    z_scores = pd.DataFrame()
    for col in columns:
        z_scores[col] = stats.zscore(df_clean[col])
    outlier_mask = (abs(z_scores) < threshold).all(axis=1)    
    return df_clean[outlier_mask]

numeric_df_no_outliers = remove_outliers(numeric_df, columns=numeric_df.columns, threshold=3)

X = numeric_df_no_outliers.drop('target_deathrate', axis=1)
y = numeric_df_no_outliers['target_deathrate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}

print("Model Comparison:")
print("-" * 60)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R²': r2, 'MAE': mae}
    print(f"{name:20} - MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")

best_model_name = max(results, key=lambda x: results[x]['R²'])
print(f"\nBest model: {best_model_name} with R² = {results[best_model_name]['R²']:.4f}")

if best_model_name == 'RandomForest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]  # Added parameter
    }
    best_model = RandomForestRegressor(random_state=42)

print("\nOptimizing best model with GridSearchCV...")
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
optimized_model = grid_search.best_estimator_
y_pred = optimized_model.predict(X_test_scaled)
optimized_r2 = r2_score(y_test, y_pred)
print(f"Optimized model R²: {optimized_r2:.4f}")

if best_model_name in ['RandomForest', 'GradientBoosting']:
    importance = optimized_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')

plt.subplot(1, 2, 2)
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot')

plt.tight_layout()
plt.savefig('optimized_diagnostics.png')
plt.close()

print("\nDiagnostic plots saved: 'actual_vs_predicted.png' and 'optimized_diagnostics.png'")
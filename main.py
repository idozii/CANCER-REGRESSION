import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, BatchNormalization
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

sns.set_style('whitegrid')
sns.set_palette('colorblind')

#* Load the data
avghouseholdsize_data = pd.read_csv('data/avg-household-size.csv')
cancereg_data = pd.read_csv('data/cancer_reg.csv')

#! Filter and clean the data
numeric_cols = cancereg_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'target_deathrate' in numeric_cols:
    numeric_cols.remove('target_deathrate')
if 'geography' in numeric_cols:
    numeric_cols.remove('geography')

numeric_imputer = SimpleImputer(strategy='median')
cancereg_data[numeric_cols] = numeric_imputer.fit_transform(cancereg_data[numeric_cols])

object_cols = cancereg_data.select_dtypes(include=['object']).columns.tolist()
if 'geography' in object_cols:
    object_cols.remove('geography')

if object_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cancereg_data[object_cols] = cat_imputer.fit_transform(cancereg_data[object_cols])

#* Merge the data
merged_data = pd.merge(avghouseholdsize_data, cancereg_data, on='geography', how='inner')
all_features = numeric_cols + object_cols

#! Prepare the data for modeling
X = merged_data[all_features]
y = merged_data['target_deathrate']

# One-hot encode categorical features
if object_cols:
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_data = one_hot_encoder.fit_transform(X[object_cols])
    encoded_feature_names = one_hot_encoder.get_feature_names_out(object_cols)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=X.index)
    X = X.drop(columns=object_cols).join(encoded_df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature importance analysis with Ridge and Lasso
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)

ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

ridge_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(ridge.coef_)
}).sort_values('Importance', ascending=False)

lasso_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(lasso.coef_)
}).sort_values('Importance', ascending=False)

# Feature importance analysis with ElasticNet
print("\nPerforming ElasticNet hyperparameter optimization...")

# Define parameter distributions for random search
param_distributions = {
    'alpha': uniform(0.001, 1.0),
    'l1_ratio': uniform(0.1, 0.9)
}

# Create base ElasticNet model
elastic_base = ElasticNet(max_iter=10000, random_state=42)

# Set up RandomizedSearchCV for faster tuning than GridSearchCV
random_search = RandomizedSearchCV(
    estimator=elastic_base,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best ElasticNet model
elastic = random_search.best_estimator_
print(f"Best ElasticNet parameters: {random_search.best_params_}")

# Calculate feature importance with the optimized ElasticNet
elastic_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(elastic.coef_)
}).sort_values('Importance', ascending=False)

# Display top features from ElasticNet
print("\nTop 20 features by ElasticNet importance:")
print(elastic_importance.head(20))

# Visualize ElasticNet feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=elastic_importance.head(20))
plt.title('Feature Importance (ElasticNet)')
plt.tight_layout()
plt.show()

# Select top features from all models
top_features_ridge = ridge_importance['Feature'].head(31).tolist()
top_features_lasso = lasso_importance['Feature'].head(31).tolist()
top_features_elastic = elastic_importance['Feature'].head(31).tolist()

# Combine top features from all three models
top_features = list(set(top_features_ridge + top_features_lasso + top_features_elastic))
print(f"\nUsing {len(top_features)} top features from all models: {top_features}")

# Continue with your existing code using these top features
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Scale the data
scaler = StandardScaler()
X_train_top_scaled = scaler.fit_transform(X_train_top)
X_test_top_scaled = scaler.transform(X_test_top)

# Neural Network
nn_model = Sequential()
early_stopping = EarlyStopping(
     min_delta=0.001, 
     patience=20, 
     restore_best_weights=True,     
)

nn_model.add(Dense(64, input_dim=X_train_top_scaled.shape[1], activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.2))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.2))
nn_model.add(Dense(16, activation='relu'))
nn_model.add(BatchNormalization())
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1))
nn_model.compile(
     optimizer=Adam(learning_rate=0.001), 
     loss='mean_squared_error',
)

history = nn_model.fit(
     X_train_top_scaled, y_train,
     validation_data=(X_test_top_scaled, y_test),
     batch_size=32,
     epochs=200,
     callbacks=[early_stopping],
     verbose=1,
)

# Plot training history
history_df = pd.DataFrame(history.history)
plt.figure(figsize=(10, 6))
history_df[['loss', 'val_loss']].plot()

# Evaluate the model
y_pred_nn = nn_model.predict(X_test_top_scaled)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network Regressor")
print(f'Mean Absolute Error: {mae_nn}')
print(f'Mean Squared Error: {mse_nn}')
print(f'R2 Score: {r2_nn}')

error_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_nn.flatten(),
    'Error': abs(y_test - y_pred_nn.flatten())
})
print("\nWorst predictions:")
print(error_df.sort_values('Error', ascending=False).head(5))

print("\nBest predictions:")
print(error_df.sort_values('Error', ascending=True).head(5))

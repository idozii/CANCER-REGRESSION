import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import BatchNormalization
from sklearn.impute import SimpleImputer

sns.set_style('whitegrid')
sns.set_palette('colorblind')

#! Load the data
avghouseholdsize_data = pd.read_csv('data/avg-household-size.csv')
cancereg_data = pd.read_csv('data/cancer_reg.csv')

#! Filter and clean the data
imputer = SimpleImputer(strategy='median')
cancereg_data[['pctsomecol18_24', 'pctemployed16_over', 'pctprivatecoveragealone']] = imputer.fit_transform(cancereg_data[['pctsomecol18_24', 'pctemployed16_over', 'pctprivatecoveragealone']])

#! Merge the data
merged_data = pd.merge(avghouseholdsize_data, cancereg_data, on='geography', how='inner')

#! Prepare the data for modeling using Random Forest
features = ['pctbachdeg25_over', 'incidencerate', 'medincome', 'pcths25_over', 'avgdeathsperyear']
X = merged_data[features]
y = merged_data['target_deathrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#! Train and evaluate the models
model_results = []

# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
model_results.append(('Gradient Boosting Regressor', mae_gbr, r2_gbr))

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
model_results.append(('Random Forest Regressor', mae_rf, r2_rf))

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, min_samples_split=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
model_results.append(('Decision Tree Regressor', mae_dt, r2_dt))

# Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
nn_model = Sequential()
early_stopping = EarlyStopping(
     min_delta=0.001, 
     patience=20, 
     restore_best_weights=True,     
)

nn_model.add(Dense(1024, input_dim=X_train_scaled.shape[1], activation='relu'))
nn_model.add(Dropout(0.3))
nn_model.add(BatchNormalization())
nn_model.add(Dense(1024, activation='relu'))
nn_model.add(Dropout(0.3))
nn_model.add(BatchNormalization())
nn_model.add(Dense(1024, activation='relu'))
nn_model.add(Dropout(0.3))
nn_model.add(BatchNormalization())
nn_model.add(Dense(1))

nn_model.compile(
     optimizer='adam', 
     loss='mean_absolute_error',
)
history = nn_model.fit(
     X_train_scaled, y_train,
     validation_data=(X_test_scaled, y_test),
     batch_size=64,
     epochs=100,
     callbacks=[early_stopping],
     verbose=1,
)
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()
plt.show()
y_pred_nn = nn_model.predict(X_test_scaled)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
model_results.append(('Neural Network Regressor', mae_nn, r2_nn))

# Plot the results
results_df = pd.DataFrame(model_results, columns=['Model', 'Mean Absolute Error', 'R2 Score'])
results_df.set_index('Model', inplace=True)
results_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
sns.set_style('whitegrid')
sns.set_palette('colorblind')

#! Load the data
avghouseholdsize_data = pd.read_csv('data/avg-household-size.csv')
cancereg_data = pd.read_csv('data/cancer_reg.csv')

#! Filter and clean the data
cancereg_data['pctsomecol18_24'] = cancereg_data['pctsomecol18_24'].fillna(cancereg_data['pctsomecol18_24'].mean())
cancereg_data['pctemployed16_over'] = cancereg_data['pctemployed16_over'].fillna(cancereg_data['pctemployed16_over'].mean())
cancereg_data['pctprivatecoveragealone'] = cancereg_data['pctprivatecoveragealone'].fillna(cancereg_data['pctprivatecoveragealone'].mean())

#! Merge the data
merged_data = pd.merge(avghouseholdsize_data, cancereg_data, on='geography', how='inner')

#! Plot the data
###Target deathrate
# # sns.histogram of target_deathrate
# sns.histplot(merged_data['target_deathrate'], bins=50, kde=True)
# plt.show()

# # sns.boxplot of target_deathrate
# sns.boxplot(data=merged_data, y='target_deathrate', palette='Set2')
# plt.show()

# # sns.boxplot of target_deathrate with binnedinc
# sns.boxplot(data=merged_data, x='binnedinc', y='target_deathrate', hue='binnedinc', palette='Set2', legend=False)
# plt.show()

# # sns.pairplot
# sns.pairplot(merged_data[['pctpubliccoveragealone', 'povertypercent', 'incidencerate', 'pctpubliccoverage', 'target_deathrate']])
# plt.show()

#! Prepare the data for modeling using Random Forest
features = ['pctbachdeg25_over', 'incidencerate', 'medincome', 'pcths25_over', 'avgdeathsperyear']
X = merged_data[features]
y = merged_data['target_deathrate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#! Train and evaluate the model
# Gradient Boosting Regressor
gbr_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
r2_gbr = r2_score(y_test, y_pred_gbr)
test_data_gbr = X_test.copy().iloc[[1]]
prediction = gbr_model.predict(test_data_gbr)
print("Gradient Boosting Regressor")
print(f'Mean Squared Error: {mse_gbr}')
print(f'R2 Score: {r2_gbr}')
print(f"The predicted value is {prediction}. The actual value is {y_test.iloc[1]}\n")

# Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
test_data = X_test_scaled[[1]]
prediction = lr_model.predict(test_data)
print("Linear Regression")
print(f'Mean Squared Error: {mse_lr}')
print(f'R2 Score: {r2_lr}')
print(f"The predicted value is {prediction}. The actual value is {y_test.iloc[1]}\n")

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
test_data_rf = X_test.copy().iloc[[1]]
prediction = rf_model.predict(test_data_rf)
print("Random Forest Regressor")
print(f'Mean Squared Error: {mse_rf}')
print(f'R2 Score: {r2_rf}')
print(f"The predicted value is {prediction}. The actual value is {y_test.iloc[1]}\n")

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, min_samples_split=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
test_data_dt = X_test.copy().iloc[[1]]
prediction = dt_model.predict(test_data_dt)
print("Decision Tree Regressor")
print(f'Mean Squared Error: {mse_dt}')
print(f'R2 Score: {r2_dt}')
print(f"The predicted value is {prediction}. The actual value is {y_test.iloc[1]}\n")

# Collect the results
models = ['Gradient Boosting', 'Linear Regression', 'Random Forest', 'Decision Tree']
mse_scores = [mse_gbr, mse_lr, mse_rf, mse_dt]
r2_scores = [r2_gbr, r2_lr, r2_rf, r2_dt]

# Plot Mean Squared Error
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['blue', 'green', 'red', 'purple'])
plt.title('Mean Squared Error Comparison')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')

# Plot R2 Score
plt.subplot(1, 2, 2)
plt.bar(models, r2_scores, color=['blue', 'green', 'red', 'purple'])
plt.title('R2 Score Comparison')
plt.xlabel('Models')
plt.ylabel('R2 Score')

plt.tight_layout()
plt.show()
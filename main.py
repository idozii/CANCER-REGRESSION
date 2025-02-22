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
y_pred = gbr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions['Difference'] = predictions['Actual'] - predictions['Predicted']
print(predictions)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred = lr_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions['Difference'] = predictions['Actual'] - predictions['Predicted']
print(predictions)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions['Difference'] = predictions['Actual'] - predictions['Predicted']
print(predictions)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

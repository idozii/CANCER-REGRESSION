import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
sns.set_style('whitegrid')
sns.set_palette('colorblind')

#! Load the data
avghouseholdsize_data = pd.read_csv('data/avg-household-size.csv')
cancereg_data = pd.read_csv('data/cancer_reg.csv')

#! Filter and clean the data
avghouseholdsize_data = avghouseholdsize_data.drop_duplicates()
avghouseholdsize_data = avghouseholdsize_data.dropna()
cancereg_data = cancereg_data.drop_duplicates()
cancereg_data = cancereg_data.dropna()
avghouseholdsize_data = avghouseholdsize_data.ffill()
cancereg_data = cancereg_data.ffill()

#! Merge the data
merged_data = pd.merge(avghouseholdsize_data, cancereg_data, on='geography', how='inner')
merged_data = merged_data.drop_duplicates()
merged_data = merged_data.dropna()

#! Plot the data
##Target deathrate
# sns.histogram
# sns.histplot(merged_data['target_deathrate'], bins=50, kde=True)
# plt.show()

# # sns.boxplot
# sns.boxplot(data=merged_data, x='binnedinc', y='target_deathrate', hue='binnedinc', palette='Set2', legend=False)
# plt.show()

# # sns.pairplot
# sns.pairplot(merged_data[['avgdeathsperyear', 'binnedinc', 'incidencerate', 'medincome' , 'medianage', 'geography', 'target_deathrate']])
# plt.show()

#!Calculate correlation matrix
# Select only numeric columns for correlation matrix
numeric_columns = merged_data.select_dtypes(include=['number']).columns
correlation_matrix = merged_data[numeric_columns].corr()

# Select features with high correlation to target_deathrate
correlation_threshold = 0.1
correlated_features = correlation_matrix['target_deathrate'][abs(correlation_matrix['target_deathrate']) > correlation_threshold].index.tolist()
correlated_features.remove('target_deathrate')

#! Build model for predicting
X = merged_data[correlated_features] 
Y = merged_data['target_deathrate']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
y_pred1 = lr_model.predict(X_train)
test_data = X_test.iloc[[1]]
prediction = lr_model.predict(test_data)
print(f"The predicted value is {prediction}. The actual value is {Y_test.iloc[1]}")
print('Root Mean Squared Error of Linear Regression: ', np.sqrt(mean_squared_error(y_pred1, Y_train)))

# Random Forest Regressor model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, Y_train)
y_pred2 = rf_model.predict(X_train)
test_data = X_test.iloc[[1]]
prediction = rf_model.predict(test_data)
print(f"The predicted value is {prediction}. The actual value is {Y_test.iloc[1]}")
print('Root Mean Squared Error of Random Forest Regressor: ', np.sqrt(mean_squared_error(y_pred2, Y_train)))

# Decision Tree Regressor model
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, Y_train)
y_pred3 = dt_model.predict(X_train)
test_data = X_test.iloc[[1]]
prediction = dt_model.predict(test_data)
print(f"The predicted value is {prediction}. The actual value is {Y_test.iloc[1]}")
print('Root Mean Squared Error of Decision Tree Regressor: ', np.sqrt(mean_squared_error(y_pred3, Y_train)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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
sns.histplot(merged_data['target_deathrate'], bins=50, kde=True)
plt.show()

# sns.boxplot
sns.boxplot(data=merged_data, x='binnedinc', y='target_deathrate', hue='binnedinc', palette='Set2', legend=False)
plt.show()

# sns.pairplot
sns.pairplot(merged_data[['avgdeathsperyear', 'binnedinc', 'incidencerate', 'medincome' , 'medianage', 'geography', 'target_deathrate']])
plt.show()

#! Build model for predicting
X = merged_data.drop(columns=['target_deathrate','binnedinc', 'geography'])
Y = merged_data['target_deathrate']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
y_pred1 = lr_model.predict(X_test)
r2_score(Y_test, y_pred1)
print('R2 Score of Linear Regression: ', r2_score(Y_test, y_pred1))

rf_model = RandomForestRegressor(random_state=42, n_estimators=50)
rf_model.fit(X_train, Y_train)
y_pred2 = rf_model.predict(X_test)
r2_score(Y_test, y_pred2)
print('R2 Score of Random Forest Regressor: ', r2_score(Y_test, y_pred2))
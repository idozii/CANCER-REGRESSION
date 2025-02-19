import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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
sns.boxplot(data=merged_data, x='binnedinc', y='target_deathrate', palette='Set2')
plt.show()

# sns.pairplot
sns.pairplot(merged_data[['avgdeathsperyear', 'binnedinc', 'incidencerate', 'medincome' , 'medianage', 'geography', 'target_deathrate']])
plt.show()

#! Build model for predicting
X = merged_data[['avganncount']]
Y = merged_data['avgdeathsperyear']

model = RandomForestRegressor(random_state=42, n_estimators=50)
model.fit(X, Y)
y_pred = model.predict(X)
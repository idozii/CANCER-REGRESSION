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
cancereg_data['pctsomecol18_24'] = cancereg_data['pctsomecol18_24'].fillna(cancereg_data['pctsomecol18_24'].mean())
cancereg_data['pctemployed16_over'] = cancereg_data['pctemployed16_over'].fillna(cancereg_data['pctemployed16_over'].mean())
cancereg_data['pctprivatecoveragealone'] = cancereg_data['pctprivatecoveragealone'].fillna(cancereg_data['pctprivatecoveragealone'].mean())

#! Merge the data
merged_data = pd.merge(avghouseholdsize_data, cancereg_data, on='geography', how='inner')

#! Plot the data
##Target deathrate
# sns.histogram
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


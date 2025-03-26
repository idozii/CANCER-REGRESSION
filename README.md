# CANCER REGRESSION ANALYSIS PROJECT

## Overview

This project analyzes and predicts cancer mortality rates across various regions using machine learning models. The analysis leverages demographic, socioeconomic, and health-related data to identify key predictors of cancer death rates.

## Dataset Description

The analysis uses two primary datasets:

### 1. Cancer Regression Data (cancer_reg.csv): Contains comprehensive information about:

- Cancer statistics (incidence rate, deaths per year)
- Economic indicators (median income, poverty percentages)
- Demographic information (age distribution, education levels)
- Healthcare coverage metrics (public and private insurance)
- Population characteristics (race, marital status)

### 2. Average Household Size Data (avg-household-size.csv): Contains regional household size statistics

The primary target variable is target_deathrate, representing the death rate due to cancer in each region.

| Column                   | Description                                                   |
|--------------------------|---------------------------------------------------------------|
| avganncount              | Average number of cancer cases diagnosed annually             |
| avgdeathsperyear         | Average number of deaths due to cancer per year               |
| target_deathrate         | Target death rate due to cancer                               |
| incidencerate            | Incidence rate of cancer                                      |
| medincome                | Median income in the region                                   |
| popest2015               | Estimated population in 2015                                  |
| povertypercent           | Percentage of population below the poverty line               |
| studypercap              | Per capita number of cancer-related clinical trials conducted |
| binnedinc                | Binned median income                                          |
| medianage                | Median age in the region                                      |
| pctprivatecoveragealone  | Percentage of population covered by private health insurance alone |
| pctempprivcoverage       | Percentage of population covered by employee-provided private health insurance |
| pctpubliccoverage        | Percentage of population covered by public health insurance   |
| pctpubliccoveragealone   | Percentage of population covered by public health insurance only |
| pctwhite                 | Percentage of White population                                |
| pctblack                 | Percentage of Black population                                |
| pctasian                 | Percentage of Asian population                                |
| pctotherrace             | Percentage of population belonging to other races             |
| pctmarriedhouseholds     | Percentage of married households                              |
| birthrate                | Birth rate in the region                                      |

## Data Preparation and Cleaning

### Initial Data Assessment

Both datasets were examined for data quality issues:

- **Household Size Data**: 3,220 records with complete data for statefips, countyfips, avghouseholdsize, and geography
- **Cancer Regression Data**: 3,047 records with most variables complete, though some columns had missing values:
  - `pctsomecol18_24`: 762 non-null values
  - `pctemployed16_over`: 2,895 non-null values
  - `pctprivatecoveragealone`: 2,438 non-null values

### Non-null checking (avghouseholdsize_data)

| Column            | Non-Null Count | Dtype   |
|-------------------|----------------|---------|
| statefips         | 3220 non-null  | int64   |
| countyfips        | 3220 non-null  | int64   |
| avghouseholdsize  | 3220 non-null  | float64 |
| geography         | 3220 non-null  | object  |

### Non-null checking (cancereg_data)

| Column                   | Non-Null Count | Dtype   |
|--------------------------|----------------|---------|
| avganncount              | 3047 non-null  | float64 |
| avgdeathsperyear         | 3047 non-null  | int64   |
| target_deathrate         | 3047 non-null  | float64 |
| incidencerate            | 3047 non-null  | float64 |
| medincome                | 3047 non-null  | int64   |
| popest2015               | 3047 non-null  | int64   |
| povertypercent           | 3047 non-null  | float64 |
| studypercap              | 3047 non-null  | float64 |
| binnedinc                | 3047 non-null  | object  |
| medianage                | 3047 non-null  | float64 |
| medianagemale            | 3047 non-null  | float64 |
| medianagefemale          | 3047 non-null  | float64 |
| geography                | 3047 non-null  | object  |
| percentmarried           | 3047 non-null  | float64 |
| pctnohs18_24             | 3047 non-null  | float64 |
| pcths18_24               | 3047 non-null  | float64 |
| pctsomecol18_24          | 762 non-null   | float64 |
| pctbachdeg18_24          | 3047 non-null  | float64 |
| pcths25_over             | 3047 non-null  | float64 |
| pctbachdeg25_over        | 3047 non-null  | float64 |
| pctemployed16_over       | 2895 non-null  | float64 |
| pctunemployed16_over     | 3047 non-null  | float64 |
| pctprivatecoverage       | 3047 non-null  | float64 |
| pctprivatecoveragealone  | 2438 non-null  | float64 |
| pctempprivcoverage       | 3047 non-null  | float64 |
| pctpubliccoverage        | 3047 non-null  | float64 |
| pctpubliccoveragealone   | 3047 non-null  | float64 |
| pctwhite                 | 3047 non-null  | float64 |
| pctblack                 | 3047 non-null  | float64 |
| pctasian                 | 3047 non-null  | float64 |
| pctotherrace             | 3047 non-null  | float64 |
| pctmarriedhouseholds     | 3047 non-null  | float64 |
| birthrate                | 3047 non-null  | float64 |

### Data Cleaning Process

The cleaning process included:

- Handling missing values using median imputation
- Handling missing values for categorical columns
- Checking and removing duplicate entries
- Merging datasets on the common `geography` column
- Feature engineering to create interaction terms between top predictors
- Additionally, outlier removal was performed to improve model quality

After merging, the final dataset contained 591 complete records with 36 features.

## Exploratory Data Analysis

The distribution and relationships in the data were visualized:

### Target Variable Analysis

![Histogram of target_deathrate](/figure/Figure_1.png)

- The target death rate shows a roughly normal distribution with values primarily between 150-200 deaths per 100,000 population

### Statistical Distribution Analysis

![Boxplot of target_deathrate](/figure/Figure_2.png)

- Boxplot analysis revealed moderate outliers in the death rate data, which were addressed in data preprocessing

### Income Level Relationship

![Boxplot by Income Brackets](/figure/Figure_3.png)

- Clear relationship between income brackets and cancer death rates
- Lower income areas generally show higher cancer death rates

### Feature Relationships

![Feature Pairplot](/figure/Figure_4.png)

- Strong negative correlation between education level and death rate
- Positive correlation between cancer incidence rate and death rate
- Significant relationship between health insurance coverage and death rate

## Feature Selection

A Random Forest Regressor was used to identify the most important predictors:

| Rank | Feature                   | Importance |
|------|---------------------------|------------|
| 1    | pctbachdeg25_over         | 0.204280   |
| 2    | incidencerate             | 0.190924   |
| 3    | medincome                 | 0.053300   |
| 4    | pcths25_over              | 0.044926   |
| 5    | avgdeathsperyear          | 0.041757   |
| 6    | pctpubliccoveragealone    | 0.035952   |
| 7    | pctprivatecoverage        | 0.034321   |
| 8    | avghouseholdsize          | 0.027914   |
| 9    | povertypercent            | 0.026800   |
| 10   | pctblack                  | 0.024291   |

![Feature Importance](/figure/Figure_5.png)

The feature importance analysis reveals:

- Education level (`pctbachdeg25_over`) is the strongest predictor
- Cancer incidence rate is the second most important factor
- Economic factors (`medincome`) play a significant role
- High school education levels (`pcths25_over`) are also influential

## Model Development and Optimization

### Gradient Boosting Model Optimization

Grid Search was used to find optimal hyperparameters:

```python
{'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}

![Gradient Boosting Feature Importance](/figure/Figure_6.png)

### Neural Network Implementation

A custom deep neural network was implemented with:
- Multiple dense layers with batch normalization
- Dropout layers for regularization
- Early stopping to prevent overfitting
- Learning rate scheduling

```python
class ImprovedNN(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc5 = nn.Linear(32, 1)


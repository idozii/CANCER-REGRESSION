
# CANCER REGRESSION ASSIGNMENT

This dataset contains information about cancer cases, deaths, and related demographic data for various regions.

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

## FIRST CHECK AFTER CLEANING

This part illustrates filtered datasets after processing through dropna(), dropduplicates() and ffill().

### *avghouseholdsize_data*

| statefips | countyfips | avghouseholdsize | geography                             |
|-----------|------------|------------------|---------------------------------------|
| 2         | 13         | 2.43             | Aleutians East Borough, Alaska        |
| 2         | 16         | 3.59             | Aleutians West Census Area, Alaska    |
| 2         | 20         | 2.77             | Anchorage Municipality, Alaska        |
| 2         | 50         | 3.86             | Bethel Census Area, Alaska            |
| 2         | 60         | 2.50             | Bristol Bay Borough, Alaska           |

### Non-null checking (avghouseholdsize_data)

| Column            | Non-Null Count | Dtype   |
|-------------------|----------------|---------|
| statefips         | 3220 non-null  | int64   |
| countyfips        | 3220 non-null  | int64   |
| avghouseholdsize  | 3220 non-null  | float64 |
| geography         | 3220 non-null  | object  |

### Missing values and duplicate checking (avghouseholdsize_data)

| Column            | Missing Values |
|-------------------|----------------|
| statefips         | 0              |
| countyfips        | 0              |
| avghouseholdsize  | 0              |
| geography         | 0              |
| dtype             | int64          |

### *cancereg_data*

| avganncount | avgdeathsperyear | target_deathrate | incidencerate | medincome | popest2015 | pctwhite  | pctblack | pctasian | pctotherrace | pctmarriedhouseholds | birthrate |
|-------------|------------------|------------------|---------------|-----------|------------|-----------|----------|----------|--------------|----------------------|-----------|
| 1397.0      | 469              | 164.9            | 489.8         | 61898     | 260131     | 81.780529 | 2.594728 | 4.821857 | 1.843479     | 52.856076             | 6.118831  |
| 173.0       | 70               | 161.3            | 411.6         | 48127     | 43269      | 89.228509 | 0.969102 | 2.246233 | 3.741352     | 45.372500             | 4.333096  |
| 102.0       | 50               | 174.7            | 349.7         | 49348     | 21026      | 90.922190 | 0.739673 | 0.465898 | 2.747358     | 54.444868             | 3.729488  |
| 427.0       | 202              | 194.8            | 430.4         | 44243     | 75882      | 91.744686 | 0.782626 | 1.161359 | 1.362643     | 51.021514             | 4.603841  |
| 57.0        | 26               | 144.4            | 350.1         | 49955     | 10321      | 94.104024 | 0.270192 | 0.665830 | 0.492135     | 54.027460             | 6.796657  |

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

### Missing values and duplicate checking (cancereg_data)

| Column                   | Missing Values |
|--------------------------|----------------|
| avganncount              | 0              |
| avgdeathsperyear         | 0              |
| target_deathrate         | 0              |
| incidencerate            | 0              |
| medincome                | 0              |
| popest2015               | 0              |
| povertypercent           | 0              |
| studypercap              | 0              |
| binnedinc                | 0              |
| medianage                | 0              |
| medianagemale            | 0              |
| medianagefemale          | 0              |
| geography                | 0              |
| percentmarried           | 0              |
| pctnohs18_24             | 0              |
| pcths18_24               | 0              |
| pctsomecol18_24          | 0              |
| pctbachdeg18_24          | 0              |
| pcths25_over             | 0              |
| pctbachdeg25_over        | 0              |
| pctemployed16_over       | 0              |
| pctunemployed16_over     | 0              |
| pctprivatecoverage       | 0              |
| pctprivatecoveragealone  | 0              |
| pctempprivcoverage       | 0              |
| pctpubliccoverage        | 0              |
| pctpubliccoveragealone   | 0              |
| pctwhite                 | 0              |
| pctblack                 | 0              |
| pctasian                 | 0              |
| pctotherrace             | 0              |
| pctmarriedhouseholds     | 0              |
| birthrate                | 0              |

### Evaluation (CLEANING_PART)

2 datasets have 1 common column: geography.

2 datasets don't need to clean anymore because they don't have any missing values and duplicates.

Approved to merge for plotting and predicting.

## MERGING DATA BASED ON GEOGRAPHY

This part provides information about merging data and how I clean these datasets within duplicates, missing values cases.

### *merged_data*

| statefips | countyfips | avghouseholdsize | geography                          | avganncount | pctblack | pctasian | pctotherrace | pctmarriedhouseholds | birthrate |
|-----------|-------------|------------------|------------------------------------|-------------|----------|----------|--------------|----------------------|-----------|
| 2         | 100         | 2.12             | Haines Borough, Alaska             | 13.0        | 0.039062 | 3.007812 | 0.507812     | 47.789116             | 5.374280  |
| 2         | 122         | 2.57             | Kenai Peninsula Borough, Alaska    | 266.0       | 0.634382 | 1.279251 | 0.559235     | 51.021643             | 5.680094  |
| 2         | 130         | 2.55             | Ketchikan Gateway Borough, Alaska  | 63.0        | 0.423389 | 7.307103 | 0.656982     | 46.667932             | 4.936668  |
| 2         | 290         | 2.81             | Yukon-Koyukuk Census Area, Alaska  | 27.0        | 0.301205 | 0.460666 | 0.212615     | 36.377397             | 6.744604  |
| 1         | 19          | 2.28             | Cherokee County, Alabama           | 158.0       | 4.925408 | 0.338357 | 0.065365     | 57.173258             | 4.687790  |

### Non-null checking

| Column                   | Non-Null Count | Dtype   |
|--------------------------|----------------|---------|
| statefips                | 591            | int64   |
| countyfips               | 591            | int64   |
| avghouseholdsize         | 591            | float64 |
| geography                | 591            | object  |
| avganncount              | 591            | float64 |
| avgdeathsperyear         | 591            | int64   |
| target_deathrate         | 591            | float64 |
| incidencerate            | 591            | float64 |
| medincome                | 591            | int64   |
| popest2015               | 591            | int64   |
| povertypercent           | 591            | float64 |
| studypercap              | 591            | float64 |
| binnedinc                | 591            | object  |
| medianage                | 591            | float64 |
| medianagemale            | 591            | float64 |
| medianagefemale          | 591            | float64 |
| percentmarried           | 591            | float64 |
| pctnohs18_24             | 591            | float64 |
| pcths18_24               | 591            | float64 |
| pctsomecol18_24          | 591            | float64 |
| pctbachdeg18_24          | 591            | float64 |
| pcths25_over             | 591            | float64 |
| pctbachdeg25_over        | 591            | float64 |
| pctemployed16_over       | 591            | float64 |
| pctunemployed16_over     | 591            | float64 |
| pctprivatecoverage       | 591            | float64 |
| pctprivatecoveragealone  | 591            | float64 |
| pctempprivcoverage       | 591            | float64 |
| pctpubliccoverage        | 591            | float64 |
| pctpubliccoveragealone   | 591            | float64 |
| pctwhite                 | 591            | float64 |
| pctblack                 | 591            | float64 |
| pctasian                 | 591            | float64 |
| pctotherrace             | 591            | float64 |
| pctmarriedhouseholds     | 591            | float64 |
| birthrate                | 591            | float64 |

### Missing values and duplicate checking

| Column                   | Missing Values |
|--------------------------|----------------|
| statefips                | 0              |
| countyfips               | 0              |
| avghouseholdsize         | 0              |
| geography                | 0              |
| avganncount              | 0              |
| avgdeathsperyear         | 0              |
| target_deathrate         | 0              |
| incidencerate            | 0              |
| medincome                | 0              |
| popest2015               | 0              |
| povertypercent           | 0              |
| studypercap              | 0              |
| binnedinc                | 0              |
| medianage                | 0              |
| medianagemale            | 0              |
| medianagefemale          | 0              |
| percentmarried           | 0              |
| pctnohs18_24             | 0              |
| pcths18_24               | 0              |
| pctsomecol18_24          | 0              |
| pctbachdeg18_24          | 0              |
| pcths25_over             | 0              |
| pctbachdeg25_over        | 0              |
| pctemployed16_over       | 0              |
| pctunemployed16_over     | 0              |
| pctprivatecoverage       | 0              |
| pctprivatecoveragealone  | 0              |
| pctempprivcoverage       | 0              |
| pctpubliccoverage        | 0              |
| pctpubliccoveragealone   | 0              |
| pctwhite                 | 0              |
| pctblack                 | 0              |
| pctasian                 | 0              |
| pctotherrace             | 0              |
| pctmarriedhouseholds     | 0              |
| birthrate                | 0              |

### Evaluation (MERGING_PART)

Noted that target death rate is the most important feature to predict.

Plot for further validation.

## PLOTTING

This part describes the visualization of the data.

### Graph 1: Histogram of target_deathrate

![Histogram of target_deathrate](/figure/Figure_1.png)

### Graph 2: Boxplot of target_deathrate with bennedInc

![Boxplot of target_deathrate with bennedInc](/figure/Figure_2.png)

### Graph 3: Pairplot of target_deathrate with other important features

![Pairplot of target_deathrate with other important features](/figure/Figure_3.png)

### Evaluation (PLOTTING_PART)

Based on the plots, we can see that the target death rate is normally distributed and has a strong correlation with other features.

## PREDICTING

This part will predict the target death rate based on the features.

### Linear Regression

### Random Forest Regression

### Evaluation (PREDICTING_PART)

The Random Forest Regression model has a better performance than the Linear Regression model.

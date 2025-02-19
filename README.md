
# Project Title

This dataset contains information about cancer cases, deaths, and related demographic data for various regions

avganncount: Average number of cancer cases diagnosed annually.
avgdeathsperyear: Average number of deaths due to cancer per year.
target_deathrate: Target death rate due to cancer.
incidencerate: Incidence rate of cancer.
medincome: Median income in the region.
popest2015: Estimated population in 2015.
povertypercent: Percentage of population below the poverty line.
studypercap: Per capita number of cancer-related clinical trials conducted.
binnedinc: Binned median income.
medianage: Median age in the region.
pctprivatecoveragealone: Percentage of population covered by private health insurance alone.
pctempprivcoverage: Percentage of population covered by employee-provided private health insurance.
pctpubliccoverage: Percentage of population covered by public health insurance.
pctpubliccoveragealone: Percentage of population covered by public health insurance only.
pctwhite: Percentage of White population.
pctblack: Percentage of Black population.
pctasian: Percentage of Asian population.
pctotherrace: Percentage of population belonging to other races.
pctmarriedhouseholds: Percentage of married households.
birthrate: Birth rate in the region.

!!FIRST CHECK AFTER CLEANING
*avghouseholdsize_data*
statefips  countyfips  avghouseholdsize                           geography
0          2          13              2.43      Aleutians East Borough, Alaska
1          2          16              3.59  Aleutians West Census Area, Alaska
2          2          20              2.77      Anchorage Municipality, Alaska
3          2          50              3.86          Bethel Census Area, Alaska
4          2          60              2.50         Bristol Bay Borough, Alaska

## Non-null checking

# Column            Non-Null Count  Dtype

---  ------            --------------  -----  
 0   statefips         3220 non-null   int64  
 1   countyfips        3220 non-null   int64  
 2   avghouseholdsize  3220 non-null   float64
3   geography         3220 non-null   object
dtypes: float64(1), int64(2), object(1)   

*Missing values and duplicate checking*
statefips           0
countyfips          0
avghouseholdsize    0
geography           0
dtype: int64

*cancereg_data*
avganncount  avgdeathsperyear  target_deathrate  incidencerate  medincome  popest2015  ...   pctwhite  pctblack  pctasian  pctotherrace  pctmarriedhouseholds  birthrate
0       1397.0               469             164.9          489.8      61898      260131  ...  81.780529  2.594728  4.821857      1.843479             52.856076   6.118831
1        173.0                70             161.3          411.6      48127       43269  ...  89.228509  0.969102  2.246233      3.741352             45.372500   4.333096
2        102.0                50             174.7          349.7      49348       21026  ...  90.922190  0.739673  0.465898      2.747358             54.444868   3.729488
3        427.0               202             194.8          430.4      44243       75882  ...  91.744686  0.782626  1.161359      1.362643             51.021514   4.603841
4         57.0                26             144.4          350.1      49955       10321  ...  94.104024  0.270192  0.665830      0.492135             54.027460   6.796657

### Non-null checking

#   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   avganncount              3047 non-null   float64
 1   avgdeathsperyear         3047 non-null   int64
 2   target_deathrate         3047 non-null   float64
 3   incidencerate            3047 non-null   float64
 4   medincome                3047 non-null   int64
 5   popest2015               3047 non-null   int64
 6   povertypercent           3047 non-null   float64
 7   studypercap              3047 non-null   float64
 8   binnedinc                3047 non-null   object
 9   medianage                3047 non-null   float64
 10  medianagemale            3047 non-null   float64
 11  medianagefemale          3047 non-null   float64
 12  geography                3047 non-null   object
 13  percentmarried           3047 non-null   float64
 14  pctnohs18_24             3047 non-null   float64
 15  pcths18_24               3047 non-null   float64
 16  pctsomecol18_24          762 non-null    float64
 17  pctbachdeg18_24          3047 non-null   float64
 18  pcths25_over             3047 non-null   float64
 19  pctbachdeg25_over        3047 non-null   float64
 20  pctemployed16_over       2895 non-null   float64
 21  pctunemployed16_over     3047 non-null   float64
 22  pctprivatecoverage       3047 non-null   float64
 23  pctprivatecoveragealone  2438 non-null   float64
 24  pctempprivcoverage       3047 non-null   float64
 25  pctpubliccoverage        3047 non-null   float64
 26  pctpubliccoveragealone   3047 non-null   float64
 27  pctwhite                 3047 non-null   float64
 28  pctblack                 3047 non-null   float64
 29  pctasian                 3047 non-null   float64
 30  pctotherrace             3047 non-null   float64
 31  pctmarriedhouseholds     3047 non-null   float64
 32  birthrate                3047 non-null   float64
dtypes: float64(28), int64(3), object(2)

*Missing values and duplicate checking*
avganncount                0
avgdeathsperyear           0
target_deathrate           0
incidencerate              0
medincome                  0
popest2015                 0
povertypercent             0
studypercap                0
binnedinc                  0
medianage                  0
medianagemale              0
medianagefemale            0
geography                  0
percentmarried             0
pctnohs18_24               0
pcths18_24                 0
pctsomecol18_24            0
pctbachdeg18_24            0
pcths25_over               0
pctbachdeg25_over          0
pctemployed16_over         0
pctunemployed16_over       0
pctprivatecoverage         0
pctprivatecoveragealone    0
pctempprivcoverage         0
pctpubliccoverage          0
pctpubliccoveragealone     0
pctwhite                   0
pctblack                   0
pctasian                   0
pctotherrace               0
pctmarriedhouseholds       0
birthrate                  0
dtype: int64

??=> 2 datasets have 1 common column: geography
??=> 2 datasets dont need to clean anymore
??=> merge for plotting and predicting

!!Merged_data
*merged_data*
statefips  countyfips  avghouseholdsize                          geography  avganncount  ...  pctblack  pctasian  pctotherrace  pctmarriedhouseholds  birthrate
0          2         100              2.12             Haines Borough, Alaska         13.0  ...  0.039062  3.007812      0.507812             47.789116   5.374280
1          2         122              2.57    Kenai Peninsula Borough, Alaska        266.0  ...  0.634382  1.279251      0.559235             51.021643   5.680094
2          2         130              2.55  Ketchikan Gateway Borough, Alaska         63.0  ...  0.423389  7.307103      0.656982             46.667932   4.936668
3          2         290              2.81  Yukon-Koyukuk Census Area, Alaska         27.0  ...  0.301205  0.460666      0.212615             36.377397   6.744604
4          1          19              2.28           Cherokee County, Alabama        158.0  ...  4.925408  0.338357      0.065365             57.173258   4.687790
[5 rows x 36 columns]

*Non-null checking*
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   statefips                591 non-null    int64  
 1   countyfips               591 non-null    int64  
 2   avghouseholdsize         591 non-null    float64
 3   geography                591 non-null    object 
 4   avganncount              591 non-null    float64
 5   avgdeathsperyear         591 non-null    int64
 6   target_deathrate         591 non-null    float64
 7   incidencerate            591 non-null    float64
 8   medincome                591 non-null    int64
 9   popest2015               591 non-null    int64
 10  povertypercent           591 non-null    float64
 11  studypercap              591 non-null    float64
 12  binnedinc                591 non-null    object
 13  medianage                591 non-null    float64
 14  medianagemale            591 non-null    float64
 15  medianagefemale          591 non-null    float64
 16  percentmarried           591 non-null    float64
 17  pctnohs18_24             591 non-null    float64
 18  pcths18_24               591 non-null    float64
 19  pctsomecol18_24          591 non-null    float64
 20  pctbachdeg18_24          591 non-null    float64
 21  pcths25_over             591 non-null    float64
 22  pctbachdeg25_over        591 non-null    float64
 23  pctemployed16_over       591 non-null    float64
 24  pctunemployed16_over     591 non-null    float64
 25  pctprivatecoverage       591 non-null    float64
 26  pctprivatecoveragealone  591 non-null    float64
 27  pctempprivcoverage       591 non-null    float64
 28  pctpubliccoverage        591 non-null    float64
 29  pctpubliccoveragealone   591 non-null    float64
 30  pctwhite                 591 non-null    float64
 31  pctblack                 591 non-null    float64
 32  pctasian                 591 non-null    float64
 33  pctotherrace             591 non-null    float64
 34  pctmarriedhouseholds     591 non-null    float64
 35  birthrate                591 non-null    float64
dtypes: float64(29), int64(5), object(2)

*Missing values and duplicate checking*
statefips                  0
countyfips                 0
avghouseholdsize           0
geography                  0
avganncount                0
avgdeathsperyear           0
target_deathrate           0
incidencerate              0
medincome                  0
popest2015                 0
povertypercent             0
studypercap                0
binnedinc                  0
medianage                  0
medianagemale              0
medianagefemale            0
percentmarried             0
pctnohs18_24               0
pcths18_24                 0
pctsomecol18_24            0
pctbachdeg18_24            0
pcths25_over               0
pctbachdeg25_over          0
pctemployed16_over         0
pctunemployed16_over       0
pctprivatecoverage         0
pctprivatecoveragealone    0
pctempprivcoverage         0
pctpubliccoverage          0
pctpubliccoveragealone     0
pctwhite                   0
pctblack                   0
pctasian                   0
pctotherrace               0
pctmarriedhouseholds       0
birthrate                  0
dtype: int64


??=> Noted that target death rate is the most important feature to predict
??=> Plot for further validation

!!Plot for target death rate (3 plots based on teacher's requirement)
*Graph1: histogram of target_deathrate*
Figure_1.png
*Graph2: boxplot of target_deathrate*
Figure_2.png
*Graph3: pairplot of target_deathrate with other important features*
Figure_3.png

??=> Based on the plots, we can see that the target death rate is normally distributed and has a strong correlation with other features

!!Predicting target death rate using random forest regressor

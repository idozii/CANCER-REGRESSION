# Cancer Regression Analysis

Production-ready R pipeline for exploratory analysis, statistical testing, and regression modeling on U.S. county-level cancer data.

## Overview

This project analyzes county-level cancer mortality (`target_deathrate`) and compares two supervised regression approaches:

- Multiple Linear Regression
- Random Forest Regression

The script also performs:

- Missing-data profiling and visualization
- Feature engineering (`binnedinc` midpoint, encoded `geography`)
- Correlation and distribution visual diagnostics
- Two one-tailed z-tests
- One-way ANOVA across states

## Project Structure

```text
.
├── CANCER-REGRESSION.Rproj
├── main.R
├── data/
│   ├── avg-household-size.csv
│   └── cancer_reg.csv
└── figure/
    └── visualize/
        ├── eda/
        └── model/
```

## Requirements

- R 4.1+
- Required R packages:
  - `caret`
  - `corrplot`
  - `randomForest`

Install dependencies:

```r
install.packages(c("caret", "corrplot", "randomForest"))
```

## How To Run

From the project root:

```bash
Rscript main.R
```

Or open `CANCER-REGRESSION.Rproj` and run `main.R` in RStudio.

## Input Data

- Main input file: `data/cancer_reg.csv`
- The script validates required columns at startup and stops early if any are missing.

## Outputs

The pipeline writes artifacts under `figure/visualize/`.

EDA outputs:

- `figure/visualize/eda/missing_values_percent.png`
- `figure/visualize/eda/correlation_matrix_clean.png`
- `figure/visualize/eda/<feature>_hist_box_clean.png` for selected features and target

Model outputs:

- `figure/visualize/model/optimized_diagnostics.png`
- `figure/visualize/model/model_testing_scatter_full.png`
- `figure/visualize/model/model_testing_comparison.csv`

## Reproducibility

The script uses a fixed random seed (`2025`) for deterministic train/test partitioning and sampling.

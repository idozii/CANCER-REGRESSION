# Load required libraries
library(randomForest)
library(caret)
library(ggplot2)
library(corrplot)
library(stringr)

# Create directories for outputs
dir.create("figure/visualize/eda", recursive = TRUE, showWarnings = FALSE)
dir.create("figure/visualize/model", recursive = TRUE, showWarnings = FALSE)

# Load data
cancer_regData <- read.csv('data/cancer_reg.csv')
avg_household_sizeData <- read.csv('data/avg-household-size.csv')

cat("Cancer registry data dimensions:", dim(cancer_regData)[1], "rows,", dim(cancer_regData)[2], "columns\n")
cat("Household size data dimensions:", dim(avg_household_sizeData)[1], "rows,", dim(avg_household_sizeData)[2], "columns\n")

# 1. INITIAL ASSESSMENT - MISSING VALUES VISUALIZATION
# Check for missing values in cancer data
missing_counts <- colSums(is.na(cancer_regData))
missing_vars <- missing_counts[missing_counts > 0]

# Print missing value counts for reference
cat("\nMissing values assessment before preprocessing:\n")
print(missing_vars)

# Save visualization to PNG file
png("figure/visualize/eda/01_missing_values_raw.png", width = 1000, height = 600)

# Create missing values plot with improved formatting
if(length(missing_vars) > 0) {
  missing_data <- as.data.frame(missing_vars)
  missing_data$Variable <- rownames(missing_data)
  names(missing_data)[1] <- "Missing_Count"
  missing_data <- missing_data[order(-missing_data$Missing_Count),]
  
  # Adjust margins to ensure variable names are fully visible
  par(mar=c(10, 4, 4, 2) + 0.1)
  
  barplot(missing_data$Missing_Count,
          names.arg = missing_data$Variable,
          main = "Missing Values by Variable (Raw Data)",
          ylab = "Number of Missing Values",
          col = "orange",
          las = 2,  # Rotate labels 90 degrees
          cex.names = 0.9,
          ylim = c(0, max(missing_data$Missing_Count) * 1.1))  # Add some space at top
  
  # Add count labels above bars
  text(x = seq_along(missing_data$Missing_Count) * 1.2 - 0.5, 
       y = missing_data$Missing_Count + max(missing_data$Missing_Count) * 0.02,
       labels = missing_data$Missing_Count,
       cex = 0.8)
} else {
  plot(1, type="n", axes=F, xlab="", ylab="", main="No Missing Values Found in Raw Data")
}

dev.off()

# 2. DATA PREPROCESSING

# Clean geography for merging
cancer_regData$geography_clean <- trimws(cancer_regData$geography)
avg_household_sizeData$geography_clean <- trimws(avg_household_sizeData$geography)

# Merge datasets 
merged_df <- merge(cancer_regData, 
                   avg_household_sizeData[c('geography_clean', 'avghouseholdsize')], 
                   by = 'geography_clean', 
                   all.x = TRUE)

cat("Data merged with household size information.\n")
cat("Merged data dimensions:", dim(merged_df)[1], "rows,", dim(merged_df)[2], "columns\n")

# Drop pctsomecol18_24 column due to excessive missing values
merged_df$pctsomecol18_24 <- NULL
cat("Removed pctsomecol18_24 column due to excessive missing values.\n")

# Handle missing values for specific columns
cols_to_fill <- c('pctemployed16_over', 'pctprivatecoveragealone')
for (col in cols_to_fill) {
  if (col %in% names(merged_df) && sum(is.na(merged_df[[col]])) > 0) {
    cat("Imputing missing values in", col, "with median value.\n")
    merged_df[[col]][is.na(merged_df[[col]])] <- median(merged_df[[col]], na.rm = TRUE)
  }
}

# Keep only binnedinc from object columns, drop others
object_columns <- names(merged_df)[sapply(merged_df, function(x) is.character(x) | is.factor(x))]
columns_to_drop <- setdiff(object_columns, 'binnedinc')
merged_df <- merged_df[, !names(merged_df) %in% columns_to_drop]

cat("Removed non-numeric categorical columns:", paste(head(columns_to_drop, 5), collapse=", "), "and others.\n")

# Convert binnedinc to numeric (extract first number)
if ('binnedinc' %in% names(merged_df)) {
  merged_df$income_category <- as.numeric(str_extract(merged_df$binnedinc, "\\d+"))
  merged_df$binnedinc <- NULL
  cat("Converted binnedinc to numeric income_category.\n")
}

# Select only numeric columns and fill missing values
numeric_df <- merged_df[sapply(merged_df, is.numeric)]
for (col in names(numeric_df)) {
  if (sum(is.na(numeric_df[[col]])) > 0) {
    cat("Imputing remaining missing values in", col, "with median.\n")
    numeric_df[[col]][is.na(numeric_df[[col]])] <- median(numeric_df[[col]], na.rm = TRUE)
  }
}

# Check for any remaining missing values
missing_after <- sum(is.na(numeric_df))
cat("After imputation, remaining missing values:", missing_after, "\n")

# 3. OUTLIER REMOVAL
# IQR-based outlier removal function
remove_outliers_iqr <- function(df, columns, multiplier = 1.5) {
  df_clean <- df
  initial_rows <- nrow(df_clean)
  
  for (col in columns) {
    if (col %in% names(df_clean)) {
      Q1 <- quantile(df_clean[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(df_clean[[col]], 0.75, na.rm = TRUE)
      IQR_val <- Q3 - Q1
      
      lower_bound <- Q1 - multiplier * IQR_val
      upper_bound <- Q3 + multiplier * IQR_val
      
      df_clean <- df_clean[df_clean[[col]] >= lower_bound & df_clean[[col]] <= upper_bound, ]
    }
  }
  
  cat(sprintf("Removed %d outliers using IQR method\n", initial_rows - nrow(df_clean)))
  return(df_clean)
}

# Remove outliers only from key variables
outlier_columns <- c('target_deathrate', 'incidencerate', 'avganncount')
numeric_df_clean <- remove_outliers_iqr(numeric_df, outlier_columns)
cat("Final cleaned dataset has", nrow(numeric_df_clean), "observations and", ncol(numeric_df_clean), "variables.\n")

# 4. DESCRIPTIVE STATISTICS AND VISUALIZATIONS (on cleaned data)

# Define key variables for detailed analysis
key_vars <- c("target_deathrate", "incidencerate", "avganncount", "medincome", 
              "povertypercent", "pctbachdeg25_over", "pctprivatecoverage", "avghouseholdsize")

# Filter to existing variables
key_vars <- key_vars[key_vars %in% names(numeric_df_clean)]

# Summary statistics for processed data
cat("\nSummary Statistics for Processed Data:\n")
print(summary(numeric_df_clean[, key_vars]))

# TARGET VARIABLE ANALYSIS
png("figure/visualize/eda/02_target_distribution_clean.png", width = 1200, height = 600)
par(mfrow = c(1, 2))

# Histogram of target variable
hist(numeric_df_clean$target_deathrate, 
     main = "Distribution of Cancer Death Rate (Target) - Clean Data",
     xlab = "Death Rate per 100k",
     col = "lightcoral",
     breaks = 30,
     border = "black")

# Boxplot of target variable
boxplot(numeric_df_clean$target_deathrate,
        main = "Boxplot of Cancer Death Rate - Clean Data",
        ylab = "Death Rate per 100k",
        col = "lightcoral")

dev.off()

# CORRELATION MATRIX OF KEY NUMERIC VARIABLES
png("figure/visualize/eda/03_correlation_matrix_clean.png", width = 1000, height = 1000)

# Calculate correlation matrix for key variables
key_numeric <- numeric_df_clean[, key_vars]
corrplot(cor(key_numeric, use = "pairwise.complete.obs"),
         method = "color",
         type = "upper",
         tl.cex = 0.8,
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         order = "hclust",
         title = "Correlation Matrix of Key Variables (Clean Data)")

dev.off()

cat("Correlation matrix created with", ncol(key_numeric), "key variables.\n")

# DISTRIBUTION OF KEY VARIABLES
png("figure/visualize/eda/04_key_variables_distributions_clean.png", width = 1200, height = 800)
par(mfrow = c(2, 4))

for(var in key_vars) {
  if(is.numeric(numeric_df_clean[[var]])) {
    hist(numeric_df_clean[[var]], 
         main = paste("Distribution of", var),
         xlab = var,
         col = "lightblue",
         breaks = 20,
         border = "black",
         cex.main = 0.8)
  }
}

dev.off()

# BOXPLOTS FOR KEY VARIABLES
png("figure/visualize/eda/05_key_variables_boxplots_clean.png", width = 1200, height = 800)
par(mfrow = c(2, 4))

for(var in key_vars) {
  if(is.numeric(numeric_df_clean[[var]])) {
    boxplot(numeric_df_clean[[var]],
            main = paste("Boxplot of", var),
            ylab = var,
            col = "lightgreen",
            cex.main = 0.8)
  }
}

dev.off()

# SCATTER PLOTS OF KEY VARIABLES VS TARGET
png("figure/visualize/eda/06_scatter_vs_target_clean.png", width = 1200, height = 800)
par(mfrow = c(2, 3))

# Remove target from key_vars for scatter plots
key_vars_no_target <- key_vars[key_vars != "target_deathrate"]

for(var in key_vars_no_target[1:6]) {  # Show first 6 variables
  if(is.numeric(numeric_df_clean[[var]])) {
    plot(numeric_df_clean[[var]], numeric_df_clean$target_deathrate,
         xlab = var,
         ylab = "Target Death Rate",
         main = paste(var, "vs Target Death Rate"),
         pch = 19,
         col = rgb(0, 0, 1, 0.6),
         cex = 0.5,
         cex.main = 0.8)
    
    # Add correlation coefficient and regression line
    if(sum(!is.na(numeric_df_clean[[var]]) & !is.na(numeric_df_clean$target_deathrate)) > 0) {
      cor_val <- cor(numeric_df_clean[[var]], numeric_df_clean$target_deathrate, use = "complete.obs")
      mtext(paste("r =", round(cor_val, 3)), side = 3, line = 0, cex = 0.7)
      
      # Add regression line
      abline(lm(target_deathrate ~ get(var), data = numeric_df_clean), col = "red", lwd = 2)
    }
  }
}

dev.off()

# SUMMARY STATISTICS TABLE
summary_stats_processed <- data.frame(
  Variable = names(numeric_df_clean),
  Mean = sapply(numeric_df_clean, mean, na.rm = TRUE),
  Median = sapply(numeric_df_clean, median, na.rm = TRUE),
  SD = sapply(numeric_df_clean, sd, na.rm = TRUE),
  Min = sapply(numeric_df_clean, min, na.rm = TRUE),
  Max = sapply(numeric_df_clean, max, na.rm = TRUE),
  stringsAsFactors = FALSE
)

write.csv(summary_stats_processed, "figure/visualize/eda/processed_data_summary_statistics.csv", row.names = FALSE)

# 5. MODEL BUILDING AND EVALUATION
# Split features and target
X <- numeric_df_clean[, !names(numeric_df_clean) %in% 'target_deathrate']
y <- numeric_df_clean$target_deathrate

# Train/test split (70/30 to match Python's 0.3 test size)
set.seed(42)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Check for zero variance predictors and remove them
near_zero_var <- nearZeroVar(X_train)
if(length(near_zero_var) > 0) {
  cat("Removing", length(near_zero_var), "near zero variance predictors\n")
  X_train <- X_train[, -near_zero_var]
  X_test <- X_test[, -near_zero_var]
}

# Handle NA values in training and test data
cat("\nChecking for NA values in features:\n")
na_count_X_train <- sum(is.na(X_train))
na_count_X_test <- sum(is.na(X_test))
cat("NA values in training features:", na_count_X_train, "\n")
cat("NA values in test features:", na_count_X_test, "\n")

# Identify and impute any NA values that might still exist
if(na_count_X_train > 0 || na_count_X_test > 0) {
  for(col in names(X_train)) {
    if(sum(is.na(X_train[[col]])) > 0) {
      col_median <- median(X_train[[col]], na.rm = TRUE)
      X_train[[col]][is.na(X_train[[col]])] <- col_median
      cat("Imputing NA values in training column", col, "\n")
    }
    
    if(sum(is.na(X_test[[col]])) > 0) {
      # Use training set median for test set to prevent data leakage
      col_median <- median(X_train[[col]], na.rm = TRUE)
      X_test[[col]][is.na(X_test[[col]])] <- col_median
      cat("Imputing NA values in test column", col, "\n")
    }
  }
}

# Verify no more NA values
cat("After final imputation - NA values in X_train:", sum(is.na(X_train)), 
    ", NA values in X_test:", sum(is.na(X_test)), "\n")

# Scale data for linear regression
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                     scale = attr(X_train_scaled, "scaled:scale"))

# Handle potential NAs introduced by scaling (e.g., due to constant columns)
if(sum(is.na(X_train_scaled)) > 0 || sum(is.na(X_test_scaled)) > 0) {
  cat("Scaling introduced NA values. Identifying problematic columns...\n")
  
  # Find columns with NAs after scaling
  na_cols_train <- colSums(is.na(X_train_scaled)) > 0
  na_cols_test <- colSums(is.na(X_test_scaled)) > 0
  problem_cols <- unique(c(names(X_train)[na_cols_train], names(X_test)[na_cols_test]))
  
  cat("Problem columns:", paste(problem_cols, collapse=", "), "\n")
  
  # Remove problematic columns from both original and scaled datasets
  X_train <- X_train[, !names(X_train) %in% problem_cols]
  X_test <- X_test[, !names(X_test) %in% problem_cols]
  
  # Re-scale after removing problematic columns
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), 
                       scale = attr(X_train_scaled, "scaled:scale"))
}

# Convert to data frames for modeling
X_train_scaled <- as.data.frame(X_train_scaled)
X_test_scaled <- as.data.frame(X_test_scaled)

# Linear Regression (using scaled data)
cat("\nTraining Linear Regression model...\n")
lm_model <- lm(y_train ~ ., data = X_train_scaled)
lm_preds <- predict(lm_model, X_test_scaled)
lm_mse <- mean((y_test - lm_preds)^2)
lm_r2 <- cor(y_test, lm_preds)^2
lm_mae <- mean(abs(y_test - lm_preds))

# Random Forest (using unscaled data)
cat("Training Random Forest model...\n")
set.seed(42)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 200, importance = TRUE)
rf_preds <- predict(rf_model, X_test)
rf_mse <- mean((y_test - rf_preds)^2)
rf_r2 <- cor(y_test, rf_preds)^2
rf_mae <- mean(abs(y_test - rf_preds))

# Model comparison results
cat("\nModel Comparison:\n")
cat("------------------------------------------------------------\n")
cat(sprintf("Multiple Linear Regression - MSE: %.2f, R²: %.4f, MAE: %.2f\n", lm_mse, lm_r2, lm_mae))
cat(sprintf("Random Forest          - MSE: %.2f, R²: %.4f, MAE: %.2f\n", rf_mse, rf_r2, rf_mae))

# Find best model
results <- data.frame(
  Model = c("LinearRegression", "RandomForest"),
  MSE = c(lm_mse, rf_mse),
  R2 = c(lm_r2, rf_r2),
  MAE = c(lm_mae, rf_mae),
  stringsAsFactors = FALSE
)

best_model_idx <- which.max(results$R2)
best_model_name <- results$Model[best_model_idx]
best_r2 <- results$R2[best_model_idx]

cat(sprintf("\nBest model: %s with R² = %.4f\n", best_model_name, best_r2))

# Get predictions from best model
if (best_model_name == "LinearRegression") {
  y_pred_best <- lm_preds
} else {
  y_pred_best <- rf_preds
}

# Plot actual vs predicted for best model
png("figure/visualize/model/actual_vs_predicted.png", width = 800, height = 600)
plot(y_test, y_pred_best, 
     xlab = "Actual", ylab = "Predicted",
     main = paste("Actual vs Predicted Values -", best_model_name),
     pch = 19, col = rgb(0, 0, 1, 0.5))
abline(a = 0, b = 1, col = "red", lty = 2, lwd = 2)
dev.off()

# Residual analysis
residuals <- y_test - y_pred_best

png("figure/visualize/model/optimized_diagnostics.png", width = 1200, height = 600)
par(mfrow = c(1, 2))

# Residuals vs predicted
plot(y_pred_best, residuals,
     xlab = "Predicted Values", ylab = "Residuals",
     main = paste("Residuals vs Predicted Values -", best_model_name),
     pch = 19, col = rgb(0, 0, 1, 0.5))
abline(h = 0, col = "red", lty = 1)

# Normal Q-Q plot
qqnorm(residuals, main = "Normal Q-Q Plot")
qqline(residuals, col = "red")

dev.off()

# 6. MODEL TESTING
cat("\n6. MODEL TESTING\n")
cat("In this section, we will compare the predicted values using 2 models and the real values of the test set.\n")

# Extract a sample of test data points for comparison table
set.seed(123) # For reproducible sample
sample_size <- 20
sample_indices <- sample(1:length(y_test), sample_size)

# Create dataframe with actual and predicted values
test_comparison <- data.frame(
  "Real_Values" = round(y_test[sample_indices], 1),
  "Multiple_Linear_Regression" = round(lm_preds[sample_indices], 1),
  "Random_Forest_Regression" = round(rf_preds[sample_indices], 1)
)

# Save comparison to CSV
write.csv(test_comparison, "figure/visualize/model/model_testing_comparison.csv", row.names = FALSE)

# Create visualization showing model comparisons
png("figure/visualize/model/model_testing_scatter.png", width = 1200, height = 900)
par(mfrow = c(2, 2))

# 1. Linear Regression scatter plot
plot(test_comparison$Real_Values, test_comparison$Multiple_Linear_Regression,
     xlab = "Real Values", ylab = "Predicted Values",
     main = "(a) Multiple Linear Regression",
     pch = 19, col = rgb(0, 0, 1, 0.7),
     xlim = range(test_comparison$Real_Values),
     ylim = range(c(test_comparison$Multiple_Linear_Regression, test_comparison$Random_Forest_Regression)))
abline(0, 1, col = "red", lty = 2)

# 2. Random Forest scatter plot
plot(test_comparison$Real_Values, test_comparison$Random_Forest_Regression,
     xlab = "Real Values", ylab = "Predicted Values",
     main = "(b) Random Forest Regression",
     pch = 19, col = rgb(0, 0, 1, 0.7),
     xlim = range(test_comparison$Real_Values),
     ylim = range(c(test_comparison$Multiple_Linear_Regression, test_comparison$Random_Forest_Regression)))
abline(0, 1, col = "red", lty = 2)

# 3. Combined plot showing both models
plot(test_comparison$Real_Values, test_comparison$Multiple_Linear_Regression,
     xlab = "Real Values", ylab = "Predicted Values",
     main = "(c) Combination of 2 models",
     pch = 19, col = rgb(1, 0.3, 0.3, 0.7),
     xlim = range(test_comparison$Real_Values),
     ylim = range(c(test_comparison$Multiple_Linear_Regression, test_comparison$Random_Forest_Regression)))
points(test_comparison$Real_Values, test_comparison$Random_Forest_Regression, 
       pch = 17, col = rgb(0, 0.7, 0.7, 0.7))
abline(0, 1, col = "blue", lty = 2)

# Add legend
legend("topleft", 
       legend = c("Linear Regression", "Random Forest"),
       col = c(rgb(1, 0.3, 0.3, 0.7), rgb(0, 0.7, 0.7, 0.7)),
       pch = c(19, 17),
       cex = 0.8)

dev.off()

# Feature importance for Random Forest
if (!is.null(rf_model)) {
  # Extract variable importance
  importance_vals <- importance(rf_model)
  feature_importance <- data.frame(
    Feature = rownames(importance_vals),
    Importance = importance_vals[, "%IncMSE"],
    stringsAsFactors = FALSE
  )
  feature_importance <- feature_importance[order(-feature_importance$Importance), ]
  
  # Create feature importance plot
  png("figure/visualize/model/feature_importance.png", width = 1000, height = 800)
  par(mar = c(5, 10, 4, 2))  # Adjust margins for long feature names
  barplot(feature_importance$Importance[10:1], 
          names.arg = feature_importance$Feature[10:1],
          horiz = TRUE, las = 1, 
          main = "Top 10 Features by Importance",
          xlab = "% Increase in MSE")
  dev.off()
  
  # Save feature importance to CSV
  write.csv(feature_importance, "figure/visualize/model/feature_importance.csv", row.names = FALSE)
  
  # Display top 10 features
  cat("\nTop 10 most important features in Random Forest model:\n")
  print(head(feature_importance, 10))
}

# Save model comparison results to CSV
write.csv(results, "figure/visualize/model/model_comparison_results.csv", row.names = FALSE)
# Load required libraries
library(randomForest)
library(caret)
library(ggplot2)
library(corrplot)
library(stringr)

# Create directories for outputs
dir.create("figure/visualize/eda", recursive = TRUE, showWarnings = FALSE)
dir.create("figure/visualize/model", recursive = TRUE, showWarnings = FALSE)
dir.create("figure/visualize/inference", recursive = TRUE, showWarnings = FALSE)

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

# After outlier removal (line ~142)
# 4. DESCRIPTIVE STATISTICS AND VISUALIZATIONS (on cleaned data)

# Generate and save comprehensive summary of cleaned dataset
cat("\nDetailed Summary Statistics of Processed Data:\n")
data_summary <- summary(numeric_df_clean)
print(data_summary)

# Save full summary statistics to file
sink("figure/visualize/eda/full_data_summary.txt")
cat("SUMMARY STATISTICS FOR ALL VARIABLES AFTER PREPROCESSING\n")
cat("========================================================\n\n")
print(data_summary)
sink()

# Generate five-number summary plus mean for key variables
cat("\nFive-Number Summary for Key Variables:\n")
key_vars <- c("target_deathrate", "incidencerate", "avganncount", "medincome", 
              "povertypercent", "pctbachdeg25_over", "pctprivatecoverage", "avghouseholdsize")

# Filter to existing variables
key_vars <- key_vars[key_vars %in% names(numeric_df_clean)]

# Create detailed summary for key variables
key_summary <- summary(numeric_df_clean[, key_vars])
print(key_summary)

# Generate additional descriptive statistics for key variables
desc_stats <- data.frame(
  Variable = key_vars,
  Mean = sapply(numeric_df_clean[, key_vars], mean),
  SD = sapply(numeric_df_clean[, key_vars], sd),
  CV = sapply(numeric_df_clean[, key_vars], function(x) sd(x)/mean(x)*100),
  Skewness = sapply(numeric_df_clean[, key_vars], function(x) {
    m3 <- sum((x - mean(x))^3)/length(x)
    s3 <- (sum((x - mean(x))^2)/length(x))^(3/2)
    m3/s3  # Skewness formula
  }),
  stringsAsFactors = FALSE
)

# Display additional statistics
cat("\nAdditional Descriptive Statistics for Key Variables:\n")
print(desc_stats, digits = 3)

# Save comprehensive descriptive statistics to CSV
write.csv(desc_stats, "figure/visualize/eda/key_variables_descriptive_stats.csv", row.names = FALSE)

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

# Create visualization showing model comparisons with full test dataset
png("figure/visualize/model/model_testing_scatter_full.png", width = 1200, height = 900)
par(mfrow = c(2, 2))

# 1. Linear Regression scatter plot (using full test set)
plot(y_test, lm_preds,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "(a) Multiple Linear Regression",
     pch = 19, col = rgb(0, 0, 1, 0.4),
     xlim = range(y_test),
     ylim = range(c(lm_preds, rf_preds)))
abline(0, 1, col = "red", lty = 2)

# 2. Random Forest scatter plot (using full test set)
plot(y_test, rf_preds,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "(b) Random Forest Regression",
     pch = 19, col = rgb(0, 0, 1, 0.4),
     xlim = range(y_test),
     ylim = range(c(lm_preds, rf_preds)))
abline(0, 1, col = "red", lty = 2)

# 3. Combined plot showing both models (using full test set)
plot(y_test, lm_preds,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "(c) Combination of 2 models",
     pch = 19, col = rgb(1, 0.3, 0.3, 0.5),
     xlim = range(y_test),
     ylim = range(c(lm_preds, rf_preds)))
points(y_test, rf_preds, 
       pch = 17, col = rgb(0, 0.7, 0.7, 0.5))
abline(0, 1, col = "blue", lty = 2)

# Add legend
legend("topleft", 
       legend = c("Linear Regression", "Random Forest"),
       col = c(rgb(1, 0.3, 0.3, 0.7), rgb(0, 0.7, 0.7, 0.7)),
       pch = c(19, 17),
       cex = 0.8)

# 4. Error distribution plot (showing residuals for both models)
residuals_lm <- y_test - lm_preds
residuals_rf <- y_test - rf_preds
hist_range <- range(c(residuals_lm, residuals_rf))

# Create overlaid histograms of residuals
hist(residuals_lm, 
     xlim = hist_range,
     col = rgb(1, 0.3, 0.3, 0.5),
     border = "red",
     breaks = 20,
     main = "(d) Error Distribution",
     xlab = "Residuals (Actual - Predicted)")
hist(residuals_rf, 
     add = TRUE, 
     col = rgb(0, 0.7, 0.7, 0.5),
     border = "cyan",
     breaks = 20)

# Add vertical lines for mean errors
abline(v = mean(residuals_lm), col = "red", lwd = 2, lty = 2)
abline(v = mean(residuals_rf), col = "cyan", lwd = 2, lty = 2)

# Add legend
legend("topright", 
       legend = c("Linear Regression", "Random Forest"),
       col = c("red", "cyan"),
       lwd = 2,
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

# 7. STATISTICAL INFERENCE
cat("\n7. STATISTICAL INFERENCE\n")
cat("In this section, we will analyze the statistical significance of our models and findings.\n")

# 7.1 Linear Regression Statistical Analysis
cat("\n7.1 Linear Regression Statistical Analysis\n")
lm_summary <- summary(lm_model)
cat("Linear model summary statistics:\n")
print(lm_summary)

# Extract significant predictors
lm_coefs <- as.data.frame(lm_summary$coefficients)
lm_coefs$Variable <- rownames(lm_coefs)
colnames(lm_coefs)[1:4] <- c("Estimate", "Std_Error", "t_value", "p_value")
lm_coefs$Significant <- ifelse(lm_coefs$p_value < 0.05, "Yes", "No")

# Sort by significance and absolute effect size
lm_coefs <- lm_coefs[order(lm_coefs$p_value, -abs(lm_coefs$Estimate)),]

# Get significant predictors
lm_significant <- lm_coefs[lm_coefs$p_value < 0.05, ]

# Save to CSV
write.csv(lm_coefs, "figure/visualize/inference/lm_coefficients.csv", row.names = FALSE)

# Create visualization of most significant coefficients
png("figure/visualize/inference/significant_predictors.png", width = 1000, height = 800)
par(mar = c(5, 12, 4, 2))  # Increase margin for variable names

# Select top 15 significant predictors by p-value
top_n <- min(15, nrow(lm_significant))
top_vars <- lm_significant[1:top_n, ]

# Create horizontal barplot
barplot(top_vars$Estimate, 
        names.arg = top_vars$Variable,
        horiz = TRUE, 
        las = 1,  # Make labels horizontal
        col = ifelse(top_vars$Estimate > 0, "steelblue", "firebrick"),
        main = "Significant Predictors of Cancer Mortality (p < 0.05)",
        xlab = "Standardized Coefficient",
        xlim = c(min(top_vars$Estimate) * 1.2, max(top_vars$Estimate) * 1.2))

# Add reference line at zero
abline(v = 0, col = "black", lty = 2)

# Add significance markers
sig_markers <- ifelse(top_vars$p_value < 0.001, "***", 
                     ifelse(top_vars$p_value < 0.01, "**", 
                           ifelse(top_vars$p_value < 0.05, "*", "")))

# Add the markers to the plot
text(x = ifelse(top_vars$Estimate > 0, 
               top_vars$Estimate + max(abs(top_vars$Estimate)) * 0.05, 
               top_vars$Estimate - max(abs(top_vars$Estimate)) * 0.05),
     y = seq(0.7, length(top_vars$Estimate) * 1.2 - 0.5, by = 1.2),
     labels = sig_markers,
     col = "black",
     cex = 1.2)

# Add legend
legend("bottomright", 
       legend = c("Positive effect", "Negative effect", "*** p<0.001", "** p<0.01", "* p<0.05"),
       fill = c("steelblue", "firebrick", NA, NA, NA),
       border = c("black", "black", NA, NA, NA),
       pch = c(NA, NA, "*", "*", "*"),
       col = c(NA, NA, "black", "black", "black"),
       cex = 0.8)

dev.off()

# 7.2 Statistical Hypotheses Testing
cat("\n7.2 Statistical Hypotheses Testing\n")

# 7.2.1 General Principles of Statistical Hypothesis Testing
cat("\n7.2.1 General Principles of Statistical Hypothesis Testing\n")

# Define function to test hypotheses about relationships between variables and cancer mortality
cat("Testing key hypotheses about factors affecting cancer mortality rates:\n")

# Function to perform hypothesis test for a single variable
test_hypothesis <- function(variable, data, alpha = 0.05) {
  # Create formula
  formula <- as.formula(paste("target_deathrate ~", variable))
  
  # Fit simple linear model
  model <- lm(formula, data = data)
  
  # Get model summary
  summary_stats <- summary(model)
  
  # Extract p-value for variable coefficient
  p_value <- summary_stats$coefficients[2, 4]
  
  # Extract coefficient
  coefficient <- summary_stats$coefficients[2, 1]
  
  # Extract R-squared
  r_squared <- summary_stats$r.squared
  
  # Get direction of effect
  direction <- ifelse(coefficient > 0, "positive", "negative")
  
  # Test significance
  significant <- p_value < alpha
  
  # Create results
  result <- list(
    variable = variable,
    null_hypothesis = paste0("H0: There is no relationship between ", variable, " and cancer mortality rate"),
    alt_hypothesis = paste0("H1: There is a relationship between ", variable, " and cancer mortality rate"),
    coefficient = coefficient,
    direction = direction,
    p_value = p_value,
    r_squared = r_squared,
    significant = significant,
    conclusion = ifelse(significant, 
                       paste0("Reject H0: ", variable, " has a statistically significant ", direction, 
                              " relationship with cancer mortality (p = ", sprintf("%.4f", p_value), ")"),
                       paste0("Fail to reject H0: No statistically significant relationship between ", 
                              variable, " and cancer mortality (p = ", sprintf("%.4f", p_value), ")"))
  )
  
  return(result)
}

# Select key variables to test
key_test_variables <- c("incidencerate", "medincome", "povertypercent", 
                      "pctbachdeg25_over", "pctprivatecoverage", "pctpubliccoverage", 
                      "pctunemployed16_over", "pctwhite", "pctblack")

# Test each hypothesis
hypothesis_results <- list()
for (var in key_test_variables) {
  if (var %in% colnames(numeric_df_clean)) {
    cat("\nTesting hypothesis for variable:", var, "\n")
    result <- test_hypothesis(var, numeric_df_clean)
    hypothesis_results[[var]] <- result
    cat(result$conclusion, "\n")
  }
}

# Convert hypothesis results to dataframe for visualization
hyp_df <- data.frame(
  Variable = sapply(hypothesis_results, function(x) x$variable),
  P_Value = sapply(hypothesis_results, function(x) x$p_value),
  Coefficient = sapply(hypothesis_results, function(x) x$coefficient),
  Direction = sapply(hypothesis_results, function(x) x$direction),
  R_Squared = sapply(hypothesis_results, function(x) x$r_squared),
  Significant = sapply(hypothesis_results, function(x) x$significant),
  stringsAsFactors = FALSE
)

# Sort by significance
hyp_df <- hyp_df[order(hyp_df$P_Value), ]

# Save results to file
write.csv(hyp_df, "figure/visualize/inference/hypothesis_tests.csv", row.names = FALSE)

# Create visualization of hypothesis test results
png("figure/visualize/inference/hypothesis_tests_visualization.png", width = 1000, height = 800)
par(mfrow = c(1, 2))

# 1. P-values plot
par(mar = c(10, 4, 4, 2))
barplot(-log10(hyp_df$P_Value),
        names.arg = hyp_df$Variable,
        col = ifelse(hyp_df$Significant, "darkgreen", "gray70"),
        main = "Statistical Significance of Variables",
        ylab = "-log10(p-value)",
        las = 2)
abline(h = -log10(0.05), col = "red", lty = 2, lwd = 2)
text(x = 0.5, y = -log10(0.05) + 0.3, labels = "Significance threshold (p = 0.05)", col = "red", pos = 4)

# 2. R-squared plot
par(mar = c(10, 4, 4, 2))
barplot(hyp_df$R_Squared,
        names.arg = hyp_df$Variable,
        col = ifelse(hyp_df$Direction == "positive", "steelblue", "firebrick"),
        main = "Explained Variance by Variable",
        ylab = "R²",
        las = 2)

# Add direction indicators
for(i in 1:nrow(hyp_df)) {
  text(x = i - 0.5, 
       y = hyp_df$R_Squared[i] + max(hyp_df$R_Squared) * 0.05,
       labels = ifelse(hyp_df$Direction[i] == "positive", "+", "-"),
       col = ifelse(hyp_df$Direction[i] == "positive", "darkblue", "darkred"),
       cex = 1.5,
       font = 2)
}

dev.off()

# 7.3 ANOVA Analysis - Comparing Models
cat("\n7.3 Comparing Models with Analysis of Variance (ANOVA)\n")

# Create reduced model with top predictors
if (nrow(lm_significant) >= 5) {
  # Select top 5 significant predictors
  top5_vars <- lm_significant$Variable[1:5]
  top5_vars <- top5_vars[top5_vars != "(Intercept)"]
  
  # Create formula for reduced model
  reduced_formula <- paste("y_train ~", paste(top5_vars, collapse = " + "))
  cat("Reduced model formula:", reduced_formula, "\n")
  
  # Fit reduced model
  reduced_model <- lm(as.formula(reduced_formula), data = X_train_scaled)
  
  # Compare models with ANOVA
  anova_result <- anova(reduced_model, lm_model)
  cat("\nANOVA Comparison of Reduced vs Full Model:\n")
  print(anova_result)
  
  # Interpret ANOVA result
  is_full_better <- anova_result[2, "Pr(>F)"] < 0.05
  anova_conclusion <- ifelse(is_full_better,
                            "The full model significantly improves fit compared to the reduced model",
                            "The reduced model is not significantly different from the full model")
  cat("\nANOVA Conclusion:", anova_conclusion, "\n")
  
  # Save ANOVA results
  capture.output(anova_result, file = "figure/visualize/inference/anova_results.txt")
  
  # Save ANOVA conclusion
  writeLines(anova_conclusion, "figure/visualize/inference/anova_conclusion.txt")
  
  # Create visual comparison of models
  png("figure/visualize/inference/model_comparison_r2.png", width = 800, height = 600)
  
  # Get R-squared values
  reduced_r2 <- summary(reduced_model)$r.squared
  reduced_adj_r2 <- summary(reduced_model)$adj.r.squared
  full_r2 <- summary(lm_model)$r.squared
  full_adj_r2 <- summary(lm_model)$adj.r.squared
  
  # Create barplot comparing R² and adjusted R²
  bar_heights <- c(reduced_r2, reduced_adj_r2, full_r2, full_adj_r2)
  bar_colors <- c("orange", "darkorange", "steelblue", "darkblue")
  bar_names <- c("Reduced Model\nR²", "Reduced Model\nAdj. R²", 
                "Full Model\nR²", "Full Model\nAdj. R²")
  
  barplot(bar_heights,
          names.arg = bar_names,
          col = bar_colors,
          main = "Model Comparison: Explained Variance",
          ylab = "R-squared / Adjusted R-squared",
          ylim = c(0, max(bar_heights) * 1.1))
  
  # Add text labels
  text(x = c(0.7, 1.9, 3.1, 4.3),
       y = bar_heights * 1.02,
       labels = sprintf("%.3f", bar_heights),
       cex = 1.2)
  
  # Add significance indicator if applicable
  if (is_full_better) {
    text(x = 3.6, y = max(bar_heights) * 1.05,
         labels = "Significantly better (p < 0.05)",
         cex = 0.9, col = "darkblue")
  }
  
  dev.off()
  
  # Create F-test visualization
  png("figure/visualize/inference/f_test_visualization.png", width = 800, height = 600)
  
  # Get F-statistic and degrees of freedom
  f_statistic <- anova_result[2, "F"]
  df1 <- anova_result[2, "Df"]
  df2 <- anova_result[2, "Res.Df"][2]
  
  # Create a sequence for the F distribution
  x <- seq(0, qf(0.999, df1, df2), length.out = 1000)
  y <- df(x, df1, df2)
  
  # Plot the F distribution
  plot(x, y, type = "l", lwd = 2, col = "blue",
       main = "F-Test Visualization for Model Comparison",
       xlab = "F-statistic",
       ylab = "Density")
  
  # Add critical F value
  f_crit <- qf(0.95, df1, df2)
  abline(v = f_crit, col = "red", lty = 2, lwd = 2)
  text(f_crit + 1, max(y) * 0.8, "Critical F", col = "red", pos = 4)
  
  # Add observed F value
  abline(v = f_statistic, col = "green", lwd = 2)
  text(f_statistic, max(y) * 0.6, 
       sprintf("Observed F = %.2f", f_statistic), 
       col = "green", pos = ifelse(f_statistic > f_crit, 4, 2))
  
  # Shade the rejection region
  rejection_x <- x[x > f_crit]
  rejection_y <- df(rejection_x, df1, df2)
  polygon(c(f_crit, rejection_x, max(rejection_x)), 
          c(0, rejection_y, 0), 
          col = rgb(1, 0, 0, 0.2))
  text(max(x) * 0.8, max(y) * 0.4, "Rejection\nregion", col = "darkred")
  
  dev.off()
}

# 7.4 Confidence Intervals Analysis
cat("\n7.4 Confidence Intervals Analysis\n")

# Calculate confidence intervals for linear model coefficients
ci <- confint(lm_model, level = 0.95)
ci_df <- as.data.frame(ci)
ci_df$Variable <- rownames(ci)
colnames(ci_df)[1:2] <- c("Lower_CI", "Upper_CI")

# Merge with coefficients
ci_results <- merge(lm_coefs, ci_df, by = "Variable")

# Sort by significance and effect size
ci_results <- ci_results[order(ci_results$p_value, -abs(ci_results$Estimate)),]

# Save confidence intervals
write.csv(ci_results, "figure/visualize/inference/confidence_intervals.csv", row.names = FALSE)

# Plot confidence intervals for top predictors
png("figure/visualize/inference/confidence_intervals.png", width = 1000, height = 800)
par(mar = c(5, 10, 4, 2))  # Adjust margins for variable names

# Select top 10 coefficients by significance
ci_top <- head(ci_results, 10)

# Create empty plot with appropriate dimensions
plot(NA, NA, 
     xlim = c(min(ci_top$Lower_CI) * 1.1, max(ci_top$Upper_CI) * 1.1),
     ylim = c(0.5, nrow(ci_top) + 0.5),
     xlab = "Standardized Coefficient with 95% Confidence Interval",
     ylab = "",
     yaxt = "n",
     main = "Top Predictors of Cancer Mortality with 95% Confidence Intervals")

# Add grid lines for reference
abline(v = 0, lty = 2, col = "darkgray")
grid(nx = NULL, ny = NA, col = "lightgray", lty = "dotted")

# Add points and lines for each coefficient
for (i in 1:nrow(ci_top)) {
  # Plot the estimate point
  points(ci_top$Estimate[i], nrow(ci_top) - i + 1, 
         pch = 19, 
         col = ifelse(ci_top$p_value[i] < 0.05, "blue", "red"),
         cex = 1.2)
  
  # Plot the confidence interval line
  lines(c(ci_top$Lower_CI[i], ci_top$Upper_CI[i]), 
        c(nrow(ci_top) - i + 1, nrow(ci_top) - i + 1),
        col = ifelse(ci_top$p_value[i] < 0.05, "blue", "red"),
        lwd = 2)
}

# Add variable names as y-axis labels
axis(2, at = nrow(ci_top):1, labels = ci_top$Variable, las = 2, cex.axis = 0.8)

# Add legend
legend("bottomright", 
       legend = c("Significant (p < 0.05)", "Not significant"),
       col = c("blue", "red"),
       lwd = 2, 
       pch = 19,
       cex = 0.8)

dev.off()

# 7.5 Model Validation Analysis - Residual Diagnostics
cat("\n7.5 Model Validation Analysis - Residual Diagnostics\n")

# Comprehensive residual analysis for linear model
png("figure/visualize/inference/lm_residual_diagnostics.png", width = 1200, height = 900)
par(mfrow = c(2, 2))

# 1. Residuals vs Fitted
plot(lm_model, which = 1, main = "Residuals vs Fitted Values")

# 2. Normal Q-Q plot
plot(lm_model, which = 2, main = "Normal Q-Q Plot of Residuals")

# 3. Scale-Location (spread-location)
plot(lm_model, which = 3, main = "Scale-Location Plot")

# 4. Residuals vs Leverage
plot(lm_model, which = 5, main = "Residuals vs Leverage")

dev.off()

# Shapiro-Wilk test for normality of residuals
shapiro_test <- shapiro.test(residuals(lm_model))
cat("\nShapiro-Wilk normality test for linear model residuals:\n")
print(shapiro_test)

# Breusch-Pagan test for heteroscedasticity
if(require("lmtest")) {
  bp_test <- lmtest::bptest(lm_model)
  cat("\nBreusch-Pagan test for heteroscedasticity:\n")
  print(bp_test)
} else {
  cat("\nPackage 'lmtest' not available. Skipping Breusch-Pagan test.\n")
}

# 7.6 Prediction Interval Analysis for Linear Model
cat("\n7.6 Prediction Interval Analysis\n")

# Create prediction intervals for a sample of test cases
set.seed(456)
sample_idx_pi <- sample(1:nrow(X_test_scaled), min(10, nrow(X_test_scaled)))

# Generate predictions with intervals
pred_interval <- predict(lm_model, X_test_scaled[sample_idx_pi,], interval = "prediction", level = 0.95)

# Create data frame with actual values, predictions, and intervals
pi_df <- data.frame(
  Actual = y_test[sample_idx_pi],
  Predicted = pred_interval[, "fit"],
  Lower_PI = pred_interval[, "lwr"],
  Upper_PI = pred_interval[, "upr"]
)

# Calculate interval width and whether actual falls within interval
pi_df$PI_Width <- pi_df$Upper_PI - pi_df$Lower_PI
pi_df$Within_PI <- pi_df$Actual >= pi_df$Lower_PI & pi_df$Actual <= pi_df$Upper_PI
pi_coverage <- mean(pi_df$Within_PI) * 100

# Save results
write.csv(pi_df, "figure/visualize/inference/prediction_intervals.csv", row.names = FALSE)

# Visualize prediction intervals
png("figure/visualize/inference/prediction_intervals.png", width = 1000, height = 600)

# Sort by predicted value for cleaner plot
pi_df <- pi_df[order(pi_df$Predicted),]

# Plot
plot(1:nrow(pi_df), pi_df$Predicted, 
     type = "o", pch = 19, col = "blue",
     ylim = c(min(pi_df$Lower_PI) * 0.9, max(pi_df$Upper_PI) * 1.05),
     xlab = "Sample Index", ylab = "Cancer Death Rate",
     main = "95% Prediction Intervals for Cancer Mortality Rate",
     xaxt = "n")

# Add x-axis labels
axis(1, at = 1:nrow(pi_df), labels = 1:nrow(pi_df))

# Add prediction intervals
for(i in 1:nrow(pi_df)) {
  lines(c(i, i), c(pi_df$Lower_PI[i], pi_df$Upper_PI[i]), col = "lightblue", lwd = 10, lend = 1)
}

# Add actual values
points(1:nrow(pi_df), pi_df$Actual, pch = 17, col = "red", cex = 1.2)

# Add legend
legend("topleft", 
       legend = c("Predicted Value", "Actual Value", "95% Prediction Interval"),
       col = c("blue", "red", "lightblue"),
       pch = c(19, 17, NA),
       lwd = c(1, 1, 10),
       cex = 0.8)

# Add coverage information
mtext(sprintf("Coverage: %.1f%% of actual values fall within 95%% prediction intervals", pi_coverage),
      side = 3, line = -2, cex = 0.8)

dev.off()
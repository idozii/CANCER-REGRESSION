library(tidyverse)
library(randomForest)
library(caret)
library(ggplot2)
library(reshape2)
library(corrplot)

avghouseholdsize_data <- read.csv('data/avg-household-size.csv')
cancereg_data <- read.csv('data/cancer_reg.csv')

na_thresh <- 0.5
cancereg_data <- cancereg_data %>%
  select(where(~mean(is.na(.)) <= na_thresh))

for (col in names(cancereg_data)) {
  if (is.numeric(cancereg_data[[col]])) {
    cancereg_data[[col]][is.na(cancereg_data[[col]])] <- median(cancereg_data[[col]], na.rm = TRUE)
  } else {
    mode_val <- names(sort(table(cancereg_data[[col]]), decreasing = TRUE))[1]
    cancereg_data[[col]][is.na(cancereg_data[[col]])] <- mode_val
  }
}

merged_data <- merge(avghouseholdsize_data, cancereg_data, by = "geography")

remove_outliers <- function(df, cols, threshold = 2.5) {
  for (col in cols) {
    if (is.numeric(df[[col]])) {
      z <- scale(df[[col]])
      df <- df[abs(z) <= threshold | is.na(z), ]
    }
  }
  return(df)
}

num_cols <- names(merged_data)[sapply(merged_data, is.numeric)]
num_cols <- setdiff(num_cols, "target_deathrate")
merged_data_clean <- remove_outliers(merged_data, c(num_cols, "target_deathrate"), threshold = 2.5)

all_features <- setdiff(names(merged_data_clean), c("target_deathrate", "geography"))
X <- merged_data_clean[, all_features]
y <- merged_data_clean$target_deathrate

X <- as.data.frame(model.matrix(~ . - 1, data = X))

# 7. Feature importance using random forest
set.seed(42)
rf <- randomForest(x = X, y = y, ntree = 200, importance = TRUE)
importance_df <- data.frame(Feature = rownames(importance(rf)), Importance = importance(rf)[, 1])
importance_df <- importance_df[order(-importance_df$Importance), ]

# 8. Top 10 features
top_features <- head(importance_df$Feature, 10)

# 9. Correlation matrix (top 10 + target)
corr_data <- merged_data_clean[, c(top_features, "target_deathrate")]
corr_matrix <- cor(corr_data, use = "pairwise.complete.obs")
png("figure/visualize/correlation_matrix_top10.png", width = 900, height = 700)
corrplot(corr_matrix, method = "color", addCoef.col = "black", tl.cex = 1)
dev.off()

# 10. Boxplot & Histogram for top 10 features
dir.create("figure", showWarnings = FALSE)
for (col in top_features) {
  png(paste0("figure/visualize/box_hist_", col, ".png"), width = 900, height = 300)
  par(mfrow = c(1, 2))
  boxplot(merged_data_clean[[col]], main = paste("Boxplot of", col))
  hist(merged_data_clean[[col]], breaks = 30, main = paste("Histogram of", col), xlab = col)
  dev.off()
}

# 11. Scatter plots: feature vs target_deathrate
for (col in top_features) {
  png(paste0("figure/visualize/scatter_", col, "_vs_target_deathrate.png"), width = 600, height = 400)
  plot(merged_data_clean[[col]], merged_data_clean$target_deathrate,
       xlab = col, ylab = "target_deathrate",
       main = paste("Scatter:", col, "vs target_deathrate"), pch = 19, col = rgb(0,0,1,0.5))
  dev.off()
}

# 12. Train/test split (70/30)
set.seed(42)
train_idx <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# 13. Linear Regression
lm_model <- lm(y_train ~ ., data = as.data.frame(X_train))
lm_preds <- predict(lm_model, as.data.frame(X_test))
lm_mae <- mean(abs(y_test - lm_preds))
lm_mse <- mean((y_test - lm_preds)^2)
lm_r2 <- cor(y_test, lm_preds)^2

# 14. Random Forest
rf_model <- randomForest(x = X_train, y = y_train, ntree = 200)
rf_preds <- predict(rf_model, X_test)
rf_mae <- mean(abs(y_test - rf_preds))
rf_mse <- mean((y_test - rf_preds)^2)
rf_r2 <- cor(y_test, rf_preds)^2

# 15. Print model results
results <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  MAE = c(lm_mae, rf_mae),
  MSE = c(lm_mse, rf_mse),
  R2 = c(lm_r2, rf_r2)
)
print(results)
write.csv(results, "figure/visualize/model_comparison_results.csv", row.names = FALSE)
lib_path <- file.path(Sys.getenv("HOME"), "R", "library")
dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_path, .libPaths()))

avghouseholdsize_data <- read.csv('data/avg-household-size.csv', stringsAsFactors = FALSE)
cancereg_data <- read.csv('data/cancer_reg.csv', stringsAsFactors = FALSE)

numeric_cols <- sapply(cancereg_data, is.numeric)
for (col in names(cancereg_data)[numeric_cols]) {
  if (any(is.na(cancereg_data[[col]]))) {
    cancereg_data[[col]][is.na(cancereg_data[[col]])] <- median(cancereg_data[[col]], na.rm = TRUE)
  }
}

categorical_cols <- sapply(cancereg_data, is.character) | sapply(cancereg_data, is.factor)
for (col in names(cancereg_data)[categorical_cols]) {
  if (any(is.na(cancereg_data[[col]]))) {
    mode_val <- names(sort(table(cancereg_data[[col]]), decreasing = TRUE))[1]
    cancereg_data[[col]][is.na(cancereg_data[[col]])] <- mode_val
  }
}

merged_data <- merge(avghouseholdsize_data, cancereg_data, by = "geography")

all_features <- names(merged_data)[names(merged_data) != "target_deathrate"]
X <- merged_data[, all_features]
y <- merged_data$target_deathrate

if (any(categorical_cols)) {
  X <- model.matrix(~ ., data = X)[, -1]
}

set.seed(42)
n <- length(y)
train_indices <- sample(1:n, size = floor(0.8 * n))
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

correlations <- numeric(ncol(X))
for (i in 1:ncol(X)) {
  correlations[i] <- abs(cor(X[,i], y, use = "complete.obs"))
}
top_feature_indices <- order(correlations, decreasing = TRUE)[1:min(20, ncol(X))]
top_features <- colnames(X)[top_feature_indices]

X_train_top <- X_train[, top_features]
X_test_top <- X_test[, top_features]

X_train_means <- colMeans(X_train_top)
X_train_sds <- apply(X_train_top, 2, sd)
X_train_top_scaled <- scale(X_train_top, center = X_train_means, scale = X_train_sds)
X_test_top_scaled <- scale(X_test_top, center = X_train_means, scale = X_train_sds)

cat("\nTop features selected by correlation:\n")
for (i in 1:length(top_features)) {
  cat(paste(i, ":", top_features[i], "\n"))
}

cat("\nStarting Random Forest training...\n")
start_time <- Sys.time()

rf_model <- randomForest(
  x = data.frame(X_train_top_scaled),
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train_top_scaled))),
  nodesize = 5,
  importance = TRUE,
  sampsize = min(5000, length(y_train)),
  replace = TRUE,
  keep.forest = TRUE
)

end_time <- Sys.time()
training_time <- difftime(end_time, start_time, units = "mins")
cat(paste("Total training time:", round(as.numeric(training_time), 2), "minutes\n"))

cat("\nRandom Forest Model Summary:\n")
print(rf_model)

cat("\nVariable Importance (Top 10):\n")
var_imp <- importance(rf_model)
var_imp_df <- data.frame(
  Feature = rownames(var_imp),
  IncMSE = var_imp[, "%IncMSE"],
  IncNodePurity = var_imp[, "IncNodePurity"]
)
var_imp_sorted <- var_imp_df[order(var_imp_df$IncMSE, decreasing = TRUE), ]
print(head(var_imp_sorted, 10))

y_pred <- predict(rf_model, newdata = data.frame(X_test_top_scaled))

mae <- mean(abs(y_pred - y_test))
mse <- mean((y_pred - y_test)^2)
rmse <- sqrt(mse)
r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

cat("\nRandom Forest Regression Results:\n")
cat(paste('Mean Absolute Error:', round(mae, 4), '\n'))
cat(paste('Mean Squared Error:', round(mse, 4), '\n'))
cat(paste('Root Mean Squared Error:', round(rmse, 4), '\n'))
cat(paste('R2 Score:', round(r2, 4), '\n'))

plot_data <- data.frame(Predicted = y_pred, Actual = y_test)

png("rf_prediction_comparison.png", width = 800, height = 600)
plot(plot_data$Actual, plot_data$Predicted, 
     main = "Random Forest: Predicted vs Actual Values",
     xlab = "Actual Cancer Death Rate", 
     ylab = "Predicted Cancer Death Rate",
     pch = 16,
     col = "darkblue",
     cex = 0.7)
abline(0, 1, col = "red", lty = 2, lwd = 2)
legend("topleft", legend = c(
  paste("MAE:", round(mae, 2)),
  paste("R²:", round(r2, 2))
), bty = "n")
dev.off()

png("rf_feature_importance.png", width = 900, height = 600)
par(mar = c(5, 10, 4, 2))
barplot(rev(head(var_imp_sorted$IncMSE, 10)), 
        names.arg = rev(head(var_imp_sorted$Feature, 10)), 
        horiz = TRUE, 
        col = "steelblue",
        xlab = "% Increase in MSE when feature is permuted",
        main = "Random Forest Variable Importance",
        las = 1,
        cex.names = 0.8)
dev.off()
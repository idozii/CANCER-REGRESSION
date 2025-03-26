lib_path <- file.path(Sys.getenv("HOME"), "R", "library")
dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_path, .libPaths()))

install_and_load <- function(package) {
  if (!require(package, character.only = TRUE, quietly = TRUE)) {
    cat(paste("Installing package:", package, "\n"))
    install.packages(package, lib = lib_path, 
                    repos = c("https://cloud.r-project.org", "https://cran.rstudio.com"),
                    dependencies = TRUE)
    if (!require(package, character.only = TRUE, quietly = TRUE)) {
      cat(paste("Warning: Failed to load", package, "- using basic R functionality instead\n"))
      return(FALSE)
    }
  }
  cat(paste("Package loaded:", package, "\n"))
  return(TRUE)
}

required_packages <- c("randomForest", "caret", "ggplot2", "dplyr", "gridExtra", "glmnet")
packages_loaded <- sapply(required_packages, install_and_load)

avghouseholdsize_data <- read.csv('data/avg-household-size.csv', stringsAsFactors = FALSE)
cancereg_data <- read.csv('data/cancer_reg.csv', stringsAsFactors = FALSE)

numeric_cols <- sapply(cancereg_data, is.numeric)
for (col in names(cancereg_data)[numeric_cols]) {
    if (any(is.na(cancereg_data[[col]]))) {
        cat(paste("Imputing missing values in", col, "\n"))
        cancereg_data[[col]][is.na(cancereg_data[[col]])] <- median(cancereg_data[[col]], na.rm = TRUE)
    }
}

categorical_cols <- sapply(cancereg_data, is.character) | sapply(cancereg_data, is.factor)
for (col in names(cancereg_data)[categorical_cols]) {
    if (any(is.na(cancereg_data[[col]]))) {
        mode_val <- names(sort(table(cancereg_data[[col]]), decreasing = TRUE))[1]
        cat(paste("Imputing missing values in", col, "with mode:", mode_val, "\n"))
        cancereg_data[[col]][is.na(cancereg_data[[col]])] <- mode_val
    }
}

merged_data <- merge(avghouseholdsize_data, cancereg_data, by = "geography")
cat(paste("Merged dataset dimensions:", nrow(merged_data), "rows,", ncol(merged_data), "columns\n"))

remove_outliers <- function(data, columns, threshold = 2.5) {
    for (col in columns) {
        if (is.numeric(data[[col]])) {
        z_scores <- scale(data[[col]])
        data <- data[abs(z_scores) <= threshold, ]
        }
    }
    return(data)
}

numeric_features <- names(merged_data)[sapply(merged_data, is.numeric)]
merged_data_clean <- remove_outliers(merged_data, numeric_features, threshold = 2.5)
cat(paste("After outlier removal:", nrow(merged_data_clean), "rows remaining\n"))

all_features <- names(merged_data_clean)[names(merged_data_clean) != "target_deathrate"]
X <- merged_data_clean[, all_features]
y <- merged_data_clean$target_deathrate

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

y_pred_rf <- predict(rf_model, newdata = data.frame(X_test_top_scaled))

mae_rf <- mean(abs(y_pred_rf - y_test))
mse_rf <- mean((y_pred_rf - y_test)^2)
rmse_rf <- sqrt(mse_rf)
r2_rf <- 1 - sum((y_test - y_pred_rf)^2) / sum((y_test - mean(y_test))^2)

cat("\nRandom Forest Regression Results:\n")
cat(paste('Mean Absolute Error:', round(mae_rf, 4), '\n'))
cat(paste('Mean Squared Error:', round(mse_rf, 4), '\n'))
cat(paste('Root Mean Squared Error:', round(rmse_rf, 4), '\n'))
cat(paste('R2 Score:', round(r2_rf, 4), '\n'))

lr_model <- lm(y_train ~ ., data = data.frame(X_train_top_scaled))
y_pred_lr <- predict(lr_model, newdata = data.frame(X_test_top_scaled))

mae_lr <- mean(abs(y_pred_lr - y_test))
mse_lr <- mean((y_pred_lr - y_test)^2)
rmse_lr <- sqrt(mse_lr)
r2_lr <- 1 - sum((y_test - y_pred_lr)^2) / sum((y_test - mean(y_test))^2)

cat("\nLinear Regression Results:\n")
cat(paste('Mean Absolute Error:', round(mae_lr, 4), '\n'))
cat(paste('Mean Squared Error:', round(mse_lr, 4), '\n'))
cat(paste('Root Mean Squared Error:', round(rmse_lr, 4), '\n'))
cat(paste('R2 Score:', round(r2_lr, 4), '\n'))

models_data <- data.frame(
    Model = c("Random Forest Regressor", "Linear Regression"),
    MAE = c(mae_rf, mae_lr),
    MSE = c(mse_rf, mse_lr),
    RMSE = c(rmse_rf, rmse_lr),
    R2 = c(r2_rf, r2_lr)
)

cat("\nModel Comparison:\n")
print(models_data)

dir.create("figure/visualize", recursive = TRUE, showWarnings = FALSE)

plot_data <- data.frame(Predicted = y_pred_rf, Actual = y_test)
plot_data$Residuals <- plot_data$Actual - plot_data$Predicted

if(packages_loaded["ggplot2"]) {
    library(ggplot2)
  
    p1 <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
        geom_point(alpha = 0.6, color = "darkblue") +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) +
        labs(
        title = "Random Forest: Predicted vs Actual Values",
        x = "Actual Cancer Death Rate",
        y = "Predicted Cancer Death Rate"
        ) +
        annotate("text", x = min(plot_data$Actual) + 10, y = max(plot_data$Predicted) - 10,
                label = paste("MAE:", round(mae_rf, 2), "\nR²:", round(r2_rf, 2)),
                hjust = 0) +
        theme_minimal()
  
    imp_data <- head(var_imp_sorted, 10)
    p2 <- ggplot(imp_data, aes(x = reorder(Feature, IncMSE), y = IncMSE)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(
        title = "Random Forest Variable Importance",
        x = "",
        y = "% Increase in MSE when feature is permuted"
        ) +
        theme_minimal()
  
    p3 <- ggplot(plot_data, aes(x = Predicted, y = Residuals)) +
        geom_point(alpha = 0.6, color = "darkblue") +
        geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
        labs(
        title = "Residual Analysis",
        x = "Predicted Values",
        y = "Residuals"
        ) +
        theme_minimal()
  
    p4 <- ggplot(plot_data, aes(x = Residuals)) +
        geom_histogram(bins = 30, fill = "steelblue", color = "black", alpha = 0.7) +
        labs(
        title = "Distribution of Residuals",
        x = "Residual Value",
        y = "Frequency"
        ) +
        theme_minimal()
  
    p5 <- ggplot(models_data, aes(x = Model, y = MSE)) +
        geom_bar(stat = "identity", fill = c("steelblue", "darkgreen")) +
        geom_text(aes(label = round(MSE, 2)), vjust = -0.5, size = 4) +
        labs(
        title = "Model Comparison: Mean Squared Error",
        x = "",
        y = "MSE (Lower is Better)"
        ) +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
    ggsave("figure/visualize/rf_prediction_comparison.png", plot = p1, width = 8, height = 6, dpi = 300)
    ggsave("figure/visualize/rf_feature_importance.png", plot = p2, width = 9, height = 6, dpi = 300)
    ggsave("figure/visualize/residual_plot.png", plot = p3, width = 8, height = 6, dpi = 300)
    ggsave("figure/visualize/residual_distribution.png", plot = p4, width = 8, height = 6, dpi = 300)
    ggsave("figure/visualize/model_comparison.png", plot = p5, width = 8, height = 6, dpi = 300)
    
    if(packages_loaded["gridExtra"]) {
        library(gridExtra)
        grid_plot <- grid.arrange(p1, p3, p4, p2, ncol = 2)
        ggsave("figure/visualize/combined_analysis.png", plot = grid_plot, width = 12, height = 10, dpi = 300)
    }
    } else {
    cat("Using base R graphics for visualization...\n")
    
    png("figure/visualize/prediction_vs_actual.png", width = 800, height = 600)
    plot(plot_data$Actual, plot_data$Predicted, 
        main = "Random Forest: Predicted vs Actual", 
        xlab = "Actual Cancer Death Rate", 
        ylab = "Predicted Cancer Death Rate",
        pch = 19, col = "darkblue")
    abline(0, 1, col = "red", lty = 2)
    legend("topleft", 
            legend = c(paste("MAE:", round(mae_rf, 2)), paste("R²:", round(r2_rf, 2))),
            bty = "n")
    dev.off()
    
    png("figure/visualize/feature_importance.png", width = 900, height = 600)
    par(mar = c(5, 12, 4, 2)) 
    barplot(rev(head(var_imp_sorted$IncMSE, 10)), 
            names.arg = rev(head(var_imp_sorted$Feature, 10)),
            main = "Feature Importance", 
            horiz = TRUE, 
            las = 1,
            col = "steelblue",
            xlab = "% Increase in MSE")
    dev.off()
    
    png("figure/visualize/residual_plot.png", width = 800, height = 600)
    plot(plot_data$Predicted, plot_data$Residuals,
        main = "Residual Analysis",
        xlab = "Predicted Values",
        ylab = "Residuals",
        pch = 19, col = "darkblue")
    abline(h = 0, col = "red", lty = 2)
    dev.off()
    
    png("figure/visualize/residual_distribution.png", width = 800, height = 600)
    hist(plot_data$Residuals, 
        main = "Distribution of Residuals",
        xlab = "Residual Value",
        ylab = "Frequency",
        col = "steelblue",
        breaks = 30)
    dev.off()
    
    png("figure/visualize/model_comparison.png", width = 800, height = 600)
    barplot(models_data$MSE, 
            names.arg = models_data$Model,
            main = "Model Comparison: Mean Squared Error",
            ylab = "MSE (Lower is Better)",
            col = c("steelblue", "darkgreen"),
            las = 2)
    text(x = seq(0.7, by = 1.2, length.out = nrow(models_data)),
        y = models_data$MSE + max(models_data$MSE) * 0.05,
        labels = round(models_data$MSE, 2))
    dev.off()
}
write.csv(models_data, "figure/visualize/r_model_comparison_results.csv", row.names = FALSE)
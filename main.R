library(randomForest)
library(caret)
library(ggplot2)
library(corrplot)
library(stringr)
library(BSDA)

cancer_regData <- read.csv('data/cancer_reg.csv')

missing_counts <- colSums(is.na(cancer_regData))
total_rows <- nrow(cancer_regData)
missing_percent <- (missing_counts / total_rows) * 100
missing_vars <- missing_percent[missing_percent > 0]

png("figure/visualize/eda/missing_values_percent.png", width = 1000, height = 600)

if(length(missing_vars) > 0) {
  missing_data <- as.data.frame(missing_vars)
  missing_data$Variable <- rownames(missing_data)
  names(missing_data)[1] <- "Missing_Percent"
  missing_data <- missing_data[order(-missing_data$Missing_Percent),]
  
  par(mar=c(10, 4, 4, 2) + 0.1)
  
  barplot(missing_data$Missing_Percent,
          names.arg = missing_data$Variable,
          main = "Missing Values by Variable (Percentage)",
          ylab = "Percent Missing (%)",
          col = "skyblue3",
          las = 2,  
          cex.names = 0.9,
          ylim = c(0, max(missing_data$Missing_Percent) * 1.1))
  
  abline(h = 10, col = "red", lty = 2, lwd = 2)
  text(x = 0.5, y = 25.5, labels = "10% threshold", col = "red", pos = 4)
  
  text(x = seq_along(missing_data$Missing_Percent) * 1.2 - 0.5, 
       y = missing_data$Missing_Percent + max(missing_data$Missing_Percent) * 0.02,
       labels = sprintf("%.1f%%", missing_data$Missing_Percent),
       cex = 0.8)
  
  mtext("Variables above red line (>10%) will be dropped; below will be imputed", 
        side = 3, line = 0, cex = 0.8)
} else {
  plot(1, type="n", axes=F, xlab="", ylab="", main="No Missing Values Found in Dataset")
}

dev.off()

cancer_regData <- cancer_regData[, !names(cancer_regData) %in% c("pctsomecol18_24", "pctprivatecoveragealone")]

numeric_df <- cancer_regData[sapply(cancer_regData, is.numeric)]

cols_to_fill <- c('pctemployed16_over')
for (col in cols_to_fill) {
  if (col %in% names(numeric_df) && sum(is.na(numeric_df[[col]])) > 0) {
    numeric_df[[col]][is.na(numeric_df[[col]])] <- median(numeric_df[[col]], na.rm = TRUE)
  }
}

# Convert binnedinc like "(61494.5, 125635]" to its midpoint
if ('binnedinc' %in% names(cancer_regData)) {
  numeric_df$income_category <- sapply(
    as.character(cancer_regData$binnedinc),
    function(x) {
      nums <- as.numeric(unlist(regmatches(x, gregexpr("[0-9.]+", x))))
      if (length(nums) == 2) {
        return(mean(nums))
      } else {
        return(NA)
      }
    }
  )
}

cor_matrix <- cor(numeric_df, use = "pairwise.complete.obs")
target_correlations <- cor_matrix[, "target_deathrate"]
top_features <- sort(abs(target_correlations), decreasing = TRUE)[2:7]
print(top_features)

selected_features <- c(
  "medincome",
  "povertypercent",
  "pctpubliccoveragealone", 
  "pcths25_over", 
  "pctbachdeg25_over", 
  "incidencerate"
)

numeric_df_clean <- numeric_df[, c(selected_features, "target_deathrate")]
key_vars <- c(selected_features, "target_deathrate")

numeric_df_clean <- numeric_df[, key_vars]

summary(numeric_df_clean)
png("figure/visualize/eda/correlation_matrix_clean.png", width = 1000, height = 1000)
key_numeric <- numeric_df_clean[, key_vars]
corrplot(cor(key_numeric, use = "pairwise.complete.obs"), method = "color", type = "upper", tl.cex = 0.8, tl.col = "black", tl.srt = 45, addCoef.col = "black", order = "hclust")
dev.off()

for (var in key_vars) {
  if (is.numeric(numeric_df_clean[[var]])) {
    png(sprintf("figure/visualize/eda/%s_hist_box_clean.png", var), width = 900, height = 400)
    par(mfrow = c(1, 2))    
    hist(numeric_df_clean[[var]],
         main = paste("Histogram of", var),
         xlab = var,
         col = "lightblue",
         breaks = 20,
         border = "black",
         cex.main = 0.9)
    boxplot(numeric_df_clean[[var]],
            main = paste("Boxplot of", var),
            ylab = var,
            col = "lightgreen",
            cex.main = 0.9)
    
    dev.off()
  }
}

# Null hypothesis (H0): The average cancer death rate is not less than 180.
# Alternative hypothesis (H1): The average cancer death rate is less than 180.

# Calculate sample mean and standard deviation
sample_mean <- mean(numeric_df_clean$target_deathrate, na.rm = TRUE)
sample_sd <- sd(numeric_df_clean$target_deathrate, na.rm = TRUE)
n <- length(numeric_df_clean$target_deathrate)

# Calculate z-statistic
z_statistic <- (sample_mean - 180) / (sample_sd / sqrt(n))
# Determine critical z-value (z_alpha) for alpha = 0.05 (one-tailed test)
z_alpha <- qnorm(0.05, lower.tail = TRUE)

cat(sprintf("Z-statistic: %.4f\n", z_statistic))
cat(sprintf("Critical Z-value (Z-alpha): %.4f\n", z_alpha))

if (z_statistic < z_alpha) {
  cat("Conclusion: Reject the null hypothesis.\n")
  cat("There is sufficient evidence to say that the average cancer death rate across U.S. counties is less than 180 per 100,000 people.\n")
} else {
  cat("Conclusion: Fail to reject the null hypothesis.\n")
  cat("There is not sufficient evidence to say that the average cancer death rate across U.S. counties is less than 180 per 100,000 people.\n")
}

# Two-Sample T-Test for target_deathrate based on povertypercent


set.seed(2025)
train_index <- createDataPartition(numeric_df_clean$target_deathrate, p = 0.8, list = FALSE)
train_data <- numeric_df_clean[train_index, ]
test_data <- numeric_df_clean[-train_index, ]

X_train <- train_data[, selected_features]
y_train <- train_data$target_deathrate
X_test <- test_data[, selected_features]
y_test <- test_data$target_deathrate

lm_model <- lm(target_deathrate ~ ., data = train_data)
lm_preds <- predict(lm_model, X_test)
lm_mse <- mean((y_test - lm_preds)^2)
lm_r2 <- cor(y_test, lm_preds)^2
lm_mae <- mean(abs(y_test - lm_preds))
print(summary(lm_model))

rf_model <- randomForest(x = X_train, y = y_train, ntree = 200, importance = TRUE)
rf_preds <- predict(rf_model, X_test)
rf_mse <- mean((y_test - rf_preds)^2)
rf_r2 <- cor(y_test, rf_preds)^2
rf_mae <- mean(abs(y_test - rf_preds))

cat(sprintf("Multiple Linear Regression - MSE: %.2f, R²: %.4f, MAE: %.2f\n", lm_mse, lm_r2, lm_mae))
cat(sprintf("Random Forest          - MSE: %.2f, R²: %.4f, MAE: %.2f\n", rf_mse, rf_r2, rf_mae))

# Residual analysis: Only Q-Q plot
residuals <- y_test - lm_preds

png("figure/visualize/model/optimized_diagnostics.png", width = 800, height = 600)
qqnorm(residuals, main = "Normal Q-Q Plot")
qqline(residuals, col = "red")
dev.off()

set.seed(2025) 
sample_size <- min(20, length(y_test))
sample_indices <- sample(1:length(y_test), sample_size)

test_comparison <- data.frame(
  "Real_Values" = round(y_test[sample_indices], 1),
  "Multiple_Linear_Regression" = round(lm_preds[sample_indices], 1),
  "Random_Forest_Regression" = round(rf_preds[sample_indices], 1)
)

write.csv(test_comparison, "figure/visualize/model/model_testing_comparison.csv", row.names = FALSE)

png("figure/visualize/model/model_testing_scatter_full.png", width = 1200, height = 900)
par(mfrow = c(2, 2))

plot(y_test, lm_preds,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "(a) Multiple Linear Regression",
     pch = 19, col = rgb(0, 0, 1, 0.4),
     xlim = range(y_test),
     ylim = range(c(lm_preds, rf_preds)))
abline(0, 1, col = "red", lty = 2)

plot(y_test, rf_preds,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "(b) Random Forest Regression",
     pch = 19, col = rgb(0, 0, 1, 0.4),
     xlim = range(y_test),
     ylim = range(c(lm_preds, rf_preds)))
abline(0, 1, col = "red", lty = 2)

plot(y_test, lm_preds,
     xlab = "Actual Values", ylab = "Predicted Values",
     main = "(c) Combination of 2 models",
     pch = 19, col = rgb(1, 0.3, 0.3, 0.5),
     xlim = range(y_test),
     ylim = range(c(lm_preds, rf_preds)))
points(y_test, rf_preds, 
       pch = 17, col = rgb(0, 0.7, 0.7, 0.5))
abline(0, 1, col = "blue", lty = 2)

legend("topleft", 
       legend = c("Linear Regression", "Random Forest"),
       col = c(rgb(1, 0.3, 0.3, 0.7), rgb(0, 0.7, 0.7, 0.7)),
       pch = c(19, 17),
       cex = 0.8)

residuals_lm <- y_test - lm_preds
residuals_rf <- y_test - rf_preds
hist_range <- range(c(residuals_lm, residuals_rf))

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

abline(v = mean(residuals_lm), col = "red", lwd = 2, lty = 2)
abline(v = mean(residuals_rf), col = "cyan", lwd = 2, lty = 2)

legend("topright", 
       legend = c("Linear Regression", "Random Forest"),
       col = c("red", "cyan"),
       lwd = 2,
       cex = 0.8)

dev.off()
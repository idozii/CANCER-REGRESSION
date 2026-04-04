suppressPackageStartupMessages({
  library(caret)
  library(corrplot)
  library(randomForest)
})

DATA_PATH <- "data/cancer_reg.csv"
EDA_DIR <- "figure/visualize/eda"
MODEL_DIR <- "figure/visualize/model"

SEED <- 2025
MISSING_DROP_COLS <- c("pctsomecol18_24", "pctprivatecoveragealone")
IMPUTE_MEDIAN_COLS <- c("pctemployed16_over")

SELECTED_FEATURES <- c(
  "medincome",
  "povertypercent",
  "pctpubliccoveragealone",
  "pcths25_over",
  "pctbachdeg25_over",
  "incidencerate"
)

required_columns <- c(
  "target_deathrate",
  "povertypercent",
  "geography",
  "binnedinc",
  SELECTED_FEATURES
)

ensure_dir <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
}

save_base_plot <- function(file_path, width, height, plot_fn) {
  png(file_path, width = width, height = height)
  on.exit(dev.off(), add = TRUE)
  plot_fn()
}

parse_income_midpoint <- function(x) {
  nums <- as.numeric(unlist(regmatches(x, gregexpr("[0-9.]+", x))))
  if (length(nums) == 2) {
    mean(nums)
  } else {
    NA_real_
  }
}

compute_metrics <- function(actual, predicted) {
  list(
    mse = mean((actual - predicted)^2),
    r2 = cor(actual, predicted)^2,
    mae = mean(abs(actual - predicted))
  )
}

if (!file.exists(DATA_PATH)) {
  stop(sprintf("Data file not found: %s", DATA_PATH))
}

ensure_dir(EDA_DIR)
ensure_dir(MODEL_DIR)

cancer_reg_data <- read.csv(DATA_PATH)

missing_required <- setdiff(required_columns, names(cancer_reg_data))
if (length(missing_required) > 0) {
  stop(sprintf("Missing required columns: %s", paste(missing_required, collapse = ", ")))
}

missing_counts <- colSums(is.na(cancer_reg_data))
total_rows <- nrow(cancer_reg_data)
missing_percent <- (missing_counts / total_rows) * 100
missing_vars <- missing_percent[missing_percent > 0]

save_base_plot(
  file.path(EDA_DIR, "missing_values_percent.png"),
  width = 1000,
  height = 600,
  plot_fn = function() {
    if (length(missing_vars) > 0) {
      missing_data <- data.frame(
        Missing_Percent = as.numeric(missing_vars),
        Variable = names(missing_vars),
        stringsAsFactors = FALSE
      )
      missing_data <- missing_data[order(-missing_data$Missing_Percent), ]

      par(mar = c(10, 4, 4, 2) + 0.1)
      bar_positions <- barplot(
        missing_data$Missing_Percent,
        names.arg = missing_data$Variable,
        main = "Missing Values by Variable (Percentage)",
        ylab = "Percent Missing (%)",
        col = "skyblue3",
        las = 2,
        cex.names = 0.9,
        ylim = c(0, max(missing_data$Missing_Percent) * 1.1)
      )

      abline(h = 10, col = "red", lty = 2, lwd = 2)
      text(
        x = min(bar_positions),
        y = 10,
        labels = "10% threshold",
        col = "red",
        pos = 3
      )
      text(
        x = bar_positions,
        y = missing_data$Missing_Percent + max(missing_data$Missing_Percent) * 0.02,
        labels = sprintf("%.1f%%", missing_data$Missing_Percent),
        cex = 0.8
      )

      mtext(
        "Variables above red line (>10%) will be dropped; below will be imputed",
        side = 3,
        line = 0,
        cex = 0.8
      )
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", main = "No Missing Values Found in Dataset")
    }
  }
)

cancer_reg_data <- cancer_reg_data[, !names(cancer_reg_data) %in% MISSING_DROP_COLS]
numeric_df <- cancer_reg_data[sapply(cancer_reg_data, is.numeric)]

for (col in IMPUTE_MEDIAN_COLS) {
  if (col %in% names(numeric_df) && anyNA(numeric_df[[col]])) {
    numeric_df[[col]][is.na(numeric_df[[col]])] <- median(numeric_df[[col]], na.rm = TRUE)
  }
}

if ("binnedinc" %in% names(cancer_reg_data)) {
  numeric_df$income_category <- vapply(
    as.character(cancer_reg_data$binnedinc),
    parse_income_midpoint,
    numeric(1)
  )
}

if ("geography" %in% names(cancer_reg_data)) {
  numeric_df$state <- as.numeric(factor(cancer_reg_data$geography, levels = unique(cancer_reg_data$geography)))
}

if (!"target_deathrate" %in% names(numeric_df)) {
  stop("Column 'target_deathrate' must be numeric and present in dataset.")
}

cor_matrix <- cor(numeric_df, use = "pairwise.complete.obs")
target_correlations <- cor_matrix[, "target_deathrate"]
top_features <- sort(abs(target_correlations), decreasing = TRUE)[2:7]
cat("Top correlated features with target_deathrate (absolute):\n")
print(top_features)

missing_features <- setdiff(SELECTED_FEATURES, names(numeric_df))
if (length(missing_features) > 0) {
  stop(sprintf("Selected feature(s) missing from numeric data: %s", paste(missing_features, collapse = ", ")))
}

key_vars <- c(SELECTED_FEATURES, "target_deathrate")
numeric_df_clean <- numeric_df[, key_vars]

cat("\nSummary of cleaned modeling dataset:\n")
print(summary(numeric_df_clean))

save_base_plot(
  file.path(EDA_DIR, "correlation_matrix_clean.png"),
  width = 1000,
  height = 1000,
  plot_fn = function() {
    corrplot(
      cor(numeric_df_clean[, key_vars], use = "pairwise.complete.obs"),
      method = "color",
      type = "upper",
      tl.cex = 0.8,
      tl.col = "black",
      tl.srt = 45,
      addCoef.col = "black",
      order = "hclust"
    )
  }
)

for (var in key_vars) {
  if (is.numeric(numeric_df_clean[[var]])) {
    save_base_plot(
      file.path(EDA_DIR, sprintf("%s_hist_box_clean.png", var)),
      width = 900,
      height = 400,
      plot_fn = function() {
        par(mfrow = c(1, 2))
        hist(
          numeric_df_clean[[var]],
          main = paste("Histogram of", var),
          xlab = var,
          col = "lightblue",
          breaks = 20,
          border = "black",
          cex.main = 0.9
        )
        boxplot(
          numeric_df_clean[[var]],
          main = paste("Boxplot of", var),
          ylab = var,
          col = "lightgreen",
          cex.main = 0.9
        )
      }
    )
  }
}

# Hypothesis test 1: mean target_deathrate < 180
sample_mean <- mean(numeric_df_clean$target_deathrate, na.rm = TRUE)
sample_sd <- sd(numeric_df_clean$target_deathrate, na.rm = TRUE)
n <- sum(!is.na(numeric_df_clean$target_deathrate))
z_statistic <- (sample_mean - 180) / (sample_sd / sqrt(n))
z_alpha <- qnorm(0.05, lower.tail = TRUE)

cat(sprintf("\nHypothesis Test 1 - Z-statistic: %.4f\n", z_statistic))
cat(sprintf("Hypothesis Test 1 - Critical Z-value: %.4f\n", z_alpha))
if (z_statistic < z_alpha) {
  cat("Conclusion: Reject H0. Average cancer death rate is less than 180.\n")
} else {
  cat("Conclusion: Fail to reject H0. Insufficient evidence that average is less than 180.\n")
}

# Hypothesis test 2: lower-poverty counties have lower mean target_deathrate
median_poverty <- median(cancer_reg_data$povertypercent, na.rm = TRUE)
df_below_median <- cancer_reg_data[cancer_reg_data$povertypercent < median_poverty, ]
df_at_or_above_median <- cancer_reg_data[cancer_reg_data$povertypercent >= median_poverty, ]

mean_below_median <- mean(df_below_median$target_deathrate, na.rm = TRUE)
mean_at_or_above_median <- mean(df_at_or_above_median$target_deathrate, na.rm = TRUE)
sd_below_median <- sd(df_below_median$target_deathrate, na.rm = TRUE)
sd_at_or_above_median <- sd(df_at_or_above_median$target_deathrate, na.rm = TRUE)
n_below_median <- nrow(df_below_median)
n_at_or_above_median <- nrow(df_at_or_above_median)

z_statistic <- (mean_below_median - mean_at_or_above_median) /
  sqrt((sd_below_median^2 / n_below_median) + (sd_at_or_above_median^2 / n_at_or_above_median))

z_alpha <- qnorm(0.05, lower.tail = TRUE)
cat(sprintf("\nHypothesis Test 2 - Z-statistic: %.4f\n", z_statistic))
cat(sprintf("Hypothesis Test 2 - Critical Z-value: %.4f\n", z_alpha))
if (z_statistic < z_alpha) {
  cat("Conclusion: Reject H0. Lower-poverty counties have lower average mortality.\n")
} else {
  cat("Conclusion: Fail to reject H0. No significant evidence for lower average mortality.\n")
}

# ANOVA: state-level differences in target_deathrate
if (!"state" %in% names(numeric_df)) {
  stop("Column 'state' is missing; ANOVA cannot be computed.")
}

anova_df <- data.frame(
  state = numeric_df$state,
  mortality_rate = numeric_df$target_deathrate
)

anova_model <- aov(mortality_rate ~ state, data = anova_df)
anova_summary <- summary(anova_model)

ss_treatment <- anova_summary[[1]]$`Sum Sq`[1]
ss_error <- anova_summary[[1]]$`Sum Sq`[2]
df_treatment <- anova_summary[[1]]$Df[1]
df_error <- anova_summary[[1]]$Df[2]
ms_treatment <- ss_treatment / df_treatment
ms_error <- ss_error / df_error
f_statistic <- anova_summary[[1]]$`F value`[1]
p_value <- anova_summary[[1]]$`Pr(>F)`[1]

anova_table <- data.frame(
  Source_of_Variation = c("Treatment (state)", "Residual"),
  Sum_of_Squares = c(ss_treatment, ss_error),
  Degrees_of_Freedom = c(df_treatment, df_error),
  Mean_Square = c(ms_treatment, ms_error),
  F_value = c(f_statistic, NA),
  P_value = c(p_value, NA)
)

cat(sprintf("\nANOVA F-statistic: %.4f\n", f_statistic))
cat(sprintf("ANOVA P-value: %.4f\n", p_value))
cat(sprintf("ANOVA df between groups: %d\n", df_treatment))
cat(sprintf("ANOVA df within groups: %d\n", df_error))
cat("ANOVA table:\n")
print(anova_table)

# Train/test split and model training
set.seed(SEED)
train_index <- createDataPartition(numeric_df_clean$target_deathrate, p = 0.8, list = FALSE)
train_data <- numeric_df_clean[train_index, ]
test_data <- numeric_df_clean[-train_index, ]

x_train <- train_data[, SELECTED_FEATURES]
y_train <- train_data$target_deathrate
x_test <- test_data[, SELECTED_FEATURES]
y_test <- test_data$target_deathrate

lm_model <- lm(target_deathrate ~ ., data = train_data)
lm_preds <- predict(lm_model, x_test)
lm_metrics <- compute_metrics(y_test, lm_preds)

cat("\nLinear model summary:\n")
print(summary(lm_model))

rf_model <- randomForest(x = x_train, y = y_train, ntree = 200, importance = TRUE)
rf_preds <- predict(rf_model, x_test)
rf_metrics <- compute_metrics(y_test, rf_preds)

cat(sprintf("\nMultiple Linear Regression - MSE: %.2f, R2: %.4f, MAE: %.2f\n", lm_metrics$mse, lm_metrics$r2, lm_metrics$mae))
cat(sprintf("Random Forest Regression  - MSE: %.2f, R2: %.4f, MAE: %.2f\n", rf_metrics$mse, rf_metrics$r2, rf_metrics$mae))

residuals <- y_test - lm_preds
save_base_plot(
  file.path(MODEL_DIR, "optimized_diagnostics.png"),
  width = 800,
  height = 600,
  plot_fn = function() {
    qqnorm(residuals, main = "Normal Q-Q Plot")
    qqline(residuals, col = "red")
  }
)

set.seed(SEED)
sample_size <- min(20, length(y_test))
sample_indices <- sample(seq_along(y_test), sample_size)

test_comparison <- data.frame(
  Real_Values = round(y_test[sample_indices], 1),
  Multiple_Linear_Regression = round(lm_preds[sample_indices], 1),
  Random_Forest_Regression = round(rf_preds[sample_indices], 1)
)

write.csv(
  test_comparison,
  file.path(MODEL_DIR, "model_testing_comparison.csv"),
  row.names = FALSE
)

save_base_plot(
  file.path(MODEL_DIR, "model_testing_scatter_full.png"),
  width = 1200,
  height = 900,
  plot_fn = function() {
    par(mfrow = c(2, 2))

    plot(
      y_test,
      lm_preds,
      xlab = "Actual Values",
      ylab = "Predicted Values",
      main = "(a) Multiple Linear Regression",
      pch = 19,
      col = rgb(0, 0, 1, 0.4),
      xlim = range(y_test),
      ylim = range(c(lm_preds, rf_preds))
    )
    abline(0, 1, col = "red", lty = 2)

    plot(
      y_test,
      rf_preds,
      xlab = "Actual Values",
      ylab = "Predicted Values",
      main = "(b) Random Forest Regression",
      pch = 19,
      col = rgb(0, 0, 1, 0.4),
      xlim = range(y_test),
      ylim = range(c(lm_preds, rf_preds))
    )
    abline(0, 1, col = "red", lty = 2)

    plot(
      y_test,
      lm_preds,
      xlab = "Actual Values",
      ylab = "Predicted Values",
      main = "(c) Combination of 2 models",
      pch = 19,
      col = rgb(1, 0.3, 0.3, 0.5),
      xlim = range(y_test),
      ylim = range(c(lm_preds, rf_preds))
    )
    points(
      y_test,
      rf_preds,
      pch = 17,
      col = rgb(0, 0.7, 0.7, 0.5)
    )
    abline(0, 1, col = "blue", lty = 2)
    legend(
      "topleft",
      legend = c("Linear Regression", "Random Forest"),
      col = c(rgb(1, 0.3, 0.3, 0.7), rgb(0, 0.7, 0.7, 0.7)),
      pch = c(19, 17),
      cex = 0.8
    )

    residuals_lm <- y_test - lm_preds
    residuals_rf <- y_test - rf_preds
    hist_range <- range(c(residuals_lm, residuals_rf))

    hist(
      residuals_lm,
      xlim = hist_range,
      col = rgb(1, 0.3, 0.3, 0.5),
      border = "red",
      breaks = 20,
      main = "(d) Error Distribution",
      xlab = "Residuals (Actual - Predicted)"
    )
    hist(
      residuals_rf,
      add = TRUE,
      col = rgb(0, 0.7, 0.7, 0.5),
      border = "cyan",
      breaks = 20
    )

    abline(v = mean(residuals_lm), col = "red", lwd = 2, lty = 2)
    abline(v = mean(residuals_rf), col = "cyan", lwd = 2, lty = 2)
    legend(
      "topright",
      legend = c("Linear Regression", "Random Forest"),
      col = c("red", "cyan"),
      lwd = 2,
      cex = 0.8
    )
  }
)
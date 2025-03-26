# Install required packages if not already installed
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("caret")) install.packages("caret")
if (!require("randomForest")) install.packages("randomForest")
if (!require("e1071")) install.packages("e1071")
if (!require("neuralnet")) install.packages("neuralnet")

# Load libraries
# Set up a personal library location
lib_path <- file.path(Sys.getenv("HOME"), "R", "library")
dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)

# Install packages to your personal library
install.packages(c("tidyverse", "caret", "randomForest", "e1071", "neuralnet"), 
                 lib = lib_path, 
                 repos = "https://cran.rstudio.com/")

# Update .libPaths() to include your personal library
.libPaths(c(lib_path, .libPaths()))

# Set theme for plots
theme_set(theme_bw())

# Read data
avghouseholdsize_data <- read.csv('data/avg-household-size.csv')
cancereg_data <- read.csv('data/cancer_reg.csv')

# Data preprocessing - impute missing values
numeric_cols <- sapply(cancereg_data, is.numeric)
if (any(numeric_cols)) {
  for (col in names(cancereg_data)[numeric_cols]) {
    if (any(is.na(cancereg_data[[col]]))) {
      cancereg_data[[col]][is.na(cancereg_data[[col]])] <- median(cancereg_data[[col]], na.rm = TRUE)
    }
  }
}

categorical_cols <- sapply(cancereg_data, is.character) | sapply(cancereg_data, is.factor)
if (any(categorical_cols)) {
  for (col in names(cancereg_data)[categorical_cols]) {
    if (any(is.na(cancereg_data[[col]]))) {
      mode_val <- names(sort(table(cancereg_data[[col]]), decreasing = TRUE))[1]
      cancereg_data[[col]][is.na(cancereg_data[[col]])] <- mode_val
    }
  }
}

# Merge data
merged_data <- merge(avghouseholdsize_data, cancereg_data, by = "geography", how = "inner")

# Prepare features
all_features <- names(merged_data)[names(merged_data) != "target_deathrate"]
X <- merged_data[, all_features]
y <- merged_data$target_deathrate

# Create dummy variables for categorical columns
if (any(categorical_cols)) {
  X <- model.matrix(~ ., data = X)[, -1] # Remove intercept
}

# Split data
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Feature selection with RandomForest
rf_model <- randomForest(x = X, y = y, importance = TRUE)
feature_importance <- importance(rf_model)
feature_importance_df <- data.frame(
  Feature = rownames(feature_importance),
  Importance = feature_importance[, "%IncMSE"]
)
feature_importance_df <- feature_importance_df[order(feature_importance_df$Importance, decreasing = TRUE), ]
top_features <- feature_importance_df$Feature[1:20]

# Select top features
X_train_top <- X_train[, top_features]
X_test_top <- X_test[, top_features]

# Scale features
preprocess_params <- preProcess(X_train_top, method = c("center", "scale"))
X_train_top_scaled <- predict(preprocess_params, X_train_top)
X_test_top_scaled <- predict(preprocess_params, X_test_top)

# Prepare data for neural network
nn_data_train <- cbind(X_train_top_scaled, target_deathrate = y_train)
nn_formula <- as.formula(paste("target_deathrate ~", paste(colnames(X_train_top_scaled), collapse = " + ")))

# Train neural network
start_time <- Sys.time()
nn_model <- neuralnet(
  formula = nn_formula,
  data = nn_data_train,
  hidden = c(128, 64, 32),
  linear.output = TRUE,
  threshold = 0.01,
  stepmax = 1e+06,
  rep = 1,
  act.fct = "relu"
)
end_time <- Sys.time()
training_time <- end_time - start_time
print(paste("Total training time:", as.numeric(training_time), "seconds"))

# Make predictions
nn_predictions <- compute(nn_model, X_test_top_scaled)
y_pred <- nn_predictions$net.result

# Evaluate model
mae <- mean(abs(y_pred - y_test))
mse <- mean((y_pred - y_test)^2)
r2 <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

# Print results
print("R Neural Network Regressor")
print(paste('Mean Absolute Error:', mae))
print(paste('Mean Squared Error:', mse))
print(paste('R2 Score:', r2))

# Plot training progress
# Note: In R neuralnet, we don't have direct access to loss history like in PyTorch
# You would need to modify the code to store loss values during training
# This is a placeholder for where you would plot training/validation loss
plot_data <- data.frame(
  x = 1:length(y_pred),
  Predicted = y_pred,
  Actual = y_test
)

ggplot() +
  geom_point(data = plot_data, aes(x = Actual, y = Predicted)) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(
    title = "Predicted vs Actual Values",
    x = "Actual Values",
    y = "Predicted Values"
  ) +
  theme_minimal()

ggsave("prediction_comparison.png", width = 10, height = 6)
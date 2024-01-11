library(dplyr)
library(packageName)
library(tidyverse)  # includes dplyr, ggplot2, tidyr
library(randomForest)
library(caret)
library(rpart)
library(glmnet)
library(class)
library(ggplot2)
library(caret)
# Specify the file path
file_path <- '/content/New Spreadsheet 1.xlsx'

# Read the Excel file into a data frame
df <- read_excel(file_path)

# Display the data frame
print(df)

# Display information about the data frame
str(df)

# Display summary statistics of the data frame
summary(df)

# Check for missing values in the data frame
colSums(is.na(df))
library(ggplot2)

# Countplot of the 'y' column
ggplot(df, aes(x = y)) +
  geom_bar() +
  ggtitle("Subscription Outcome Count")

# Histogram of age distribution
ggplot(df, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
  ggtitle("Age Distribution") +
  xlab("Age") +
  ylab("Count")

# Boxplot of age by subscription outcome
ggplot(df, aes(x = y, y = age)) +
  geom_boxplot() +
  ggtitle("Age Distribution by Subscription Outcome")

# Visualization of campaign outcomes by day of the week
ggplot(df, aes(x = day_of_week, fill = y, order = c('mon', 'tue', 'wed', 'thu', 'fri'))) +
  geom_bar(position = "dodge") +
  ggtitle("Campaign Outcomes by Day of the Week")

# Visualization of previous marketing campaign outcomes
ggplot(df, aes(x = as.factor(previous), fill = y)) +
  geom_bar(position = "dodge") +
  ggtitle("Previous Marketing Campaign Outcomes")
# Identify categorical columns
categorical_columns <- names(df)[sapply(df, is.character)]

# Create dummy variables
df_dummies <- dummyVars(" ~ .", data = df[, categorical_columns], fullRank = TRUE)
df_dummies <- predict(df_dummies, newdata = df[, categorical_columns])

# Combine the dummy variables with the original data frame
df <- cbind(df, df_dummies)

# Remove the original categorical columns
df <- df[, -which(names(df) %in% categorical_columns)]

# Display the modified data frame
print(df)
library(caret)

# Set a seed for reproducibility
set.seed(123)

# Separate features and target variable
X <- df[, !names(df) %in% c("y_yes")]
y <- df$y_yes

# Create a train/test split
index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[index, ]
X_test <- X[-index, ]
y_train <- y[index]
y_test <- y[-index]

# Set up K-fold cross-validation
set.seed(125)
cv <- createFolds(y_train, k = 3, list = TRUE, returnTrain = FALSE)

# Display the structure of the data partitions
str(X_train)
str(X_test)
str(y_train)
str(y_test)

install.packages(c("randomForest", "Metrics", "pROC"))
library(randomForest)
library(Metrics)
library(pROC)

# Define the parameter grid
params_rf_alternative <- list(n_estimators = c(50, 100, 200),
                              min_samples_split = c(2, 5, 10),
                              min_samples_leaf = c(1, 2, 4))

# Grid search with alternative hyperparameters for Random Forest
grid_search_rf_alternative <- train(x = X_train, y = y_train,
                                    method = "rf",
                                    tuneGrid = params_rf_alternative,
                                    trControl = trainControl(method = "cv", number = 3),
                                    verbose = TRUE)

# Display the best parameters and cross-validated best score for Random Forest
print("The best parameters for Random Forest are:", grid_search_rf_alternative$bestTune)
print("Cross-validated best score for Random Forest: {}%".format(round(grid_search_rf_alternative$results$Accuracy * 100, 3)))

# Construct a Random Forest classifier object with the best hyperparameters
clf_rf_alternative <- randomForest(y_train ~ ., data = cbind(y_train, X_train), ntree = grid_search_rf_alternative$bestTune$n_estimators,
                                   mtry = sqrt(ncol(X_train)), nodesize = grid_search_rf_alternative$bestTune$min_samples_leaf)

# Predictions on the test set
y_pred_rf_alternative <- predict(clf_rf_alternative, newdata = X_test)

# Accuracy of the Random Forest Model on test data
print("Accuracy of the Random Forest Model on test data:", confusionMatrix(y_pred_rf_alternative, y_test)$overall['Accuracy'])

# Confusion Matrix
print('Confusion Matrix:', confusionMatrix(y_pred_rf_alternative, y_test)$table)
print(classification_report(y_test, y_pred_rf_alternative))
# Kappa Statistic for Random Forest
kappa_rf <- kappa2(y_pred_rf_alternative, y_test)$value
print("Cohen's Kappa Statistic for Random Forest:", kappa_rf)
# ROC Curve for Random Forest
y_pred_proba_rf <- predict(clf_rf_alternative, newdata = X_test, type = "prob")[, 2]
roc_rf <- roc(y_test, y_pred_proba_rf)

# Plot ROC curve for Random Forest
plot(roc_rf, col = "darkorange", main = "Receiver Operating Characteristic (ROC) Curve for Random Forest")
abline(a = 0, b = 1, lty = 2, col = "navy")


install.packages(c("caret", "rpart", "rpart.plot", "Metrics", "pROC"))
library(caret)
library(rpart)
library(rpart.plot)
library(Metrics)
library(pROC)

# Decision Tree
params_dt <- list(
  criterion = c("gini", "entropy"),
  splitter = c("best", "random"),
  max_depth = c(NA, 10, 20, 30)
)

grid_search_dt <- train(x = X_train, y = y_train,
                        method = "rpart",
                        tuneGrid = params_dt,
                        trControl = trainControl(method = "cv", number = 5),
                        verbose = TRUE)

# Display the best parameters and cross-validated best score for Decision Tree
print("The best parameters for Decision Tree are:", grid_search_dt$bestTune)
print("Cross-validated best score for Decision Tree: {}%".format(round(grid_search_dt$results$Accuracy * 100, 3)))

# Construct a Decision Tree classifier object with the best hyperparameters
clf_dt <- rpart(y_train ~ ., data = cbind(y_train, X_train),
                method = "class",
                control = rpart.control(cp = 0.01))  # Use an appropriate value for cp

# Plot the Decision Tree
prp(clf_dt, extra = 1, type = 2, branch = 0.5, varlen = 0, yesno = 2, main = "Decision Tree")

# Predictions on the test set for Decision Tree
y_pred_dt <- predict(clf_dt, newdata = X_test, type = "class")

# Accuracy of the Decision Tree Model on test data
print("Accuracy of the Decision Tree Model on test data:", confusionMatrix(y_pred_dt, y_test)$overall['Accuracy'])

# Confusion Matrix for Decision Tree
print('Confusion Matrix for Decision Tree:', confusionMatrix(y_pred_dt, y_test)$table)
print(classification_report(y_test, y_pred_dt))
# Kappa Statistic for Decision Tree
kappa_dt <- kappa2(y_pred_dt, y_test)$value
print("Cohen's Kappa Statistic for Decision Tree:", kappa_dt)
# ROC Curve for Decision Tree
y_pred_proba_dt <- predict(clf_dt, newdata = X_test, type = "prob")[, 2]
roc_dt <- roc(y_test, y_pred_proba_dt)

# Plot ROC curve for Decision Tree
plot(roc_dt, col = "darkorange", main = "Receiver Operating Characteristic (ROC) Curve for Decision Tree")
abline(a = 0, b = 1, lty = 2, col = "navy")


# Install and load necessary libraries (run this only once)
install.packages(c("caret", "Metrics", "pROC"))
library(caret)
library(Metrics)
library(pROC)

# Logistic Regression
params_lr <- list(
  penalty = c("l1", "l2"),
  C = c(0.001, 0.01, 0.1, 1, 10, 100, 1000)
)

grid_search_lr <- train(x = X_train, y = y_train,
                        method = "glm",
                        trControl = trainControl(method = "cv", number = 5),
                        verbose = TRUE)

# Display the best parameters and cross-validated best score for Logistic Regression
print("The best parameters for Logistic Regression are:", grid_search_lr$bestTune)
print("Cross-validated best score for Logistic Regression: {}%".format(round(grid_search_lr$results$Accuracy * 100, 3)))

# Construct a Logistic Regression classifier object with the best hyperparameters
clf_lr <- glm(y_train ~ ., data = cbind(y_train, X_train), family = binomial)

# Predictions on the test set for Logistic Regression
y_pred_lr <- predict(clf_lr, newdata = X_test, type = "response")
y_pred_lr <- ifelse(y_pred_lr > 0.5, 1, 0)

# Accuracy of the Logistic Regression Model on test data
print("Accuracy of the Logistic Regression Model on test data:", confusionMatrix(y_pred_lr, y_test)$overall['Accuracy'])

# Confusion Matrix for Logistic Regression
print('Confusion Matrix for Logistic Regression:', confusionMatrix(y_pred_lr, y_test)$table)
print(classification_report(y_test, y_pred_lr))

# Kappa Statistic for Logistic Regression
kappa_lr <- kappa2(y_pred_lr, y_test)$value
print("Cohen's Kappa Statistic for Logistic Regression:", kappa_lr)
# ROC Curve for Logistic Regression
roc_lr <- roc(y_test, y_pred_lr)

# Plot ROC curve for Logistic Regression
plot(roc_lr, col = "darkorange", main = "Receiver Operating Characteristic (ROC) Curve for Logistic Regression")
abline(a = 0, b = 1, lty = 2, col = "navy")
# Install and load necessary libraries (run this only once)
install.packages(c("caret", "Metrics", "pROC"))
library(caret)
library(Metrics)
library(pROC)

# Create a KNN classifier
knn_classifier <- train(x = X_train, y = y_train,
                        method = "knn",
                        tuneGrid = data.frame(k = 5),
                        trControl = trainControl(method = "cv", number = 5),
                        verbose = TRUE)

# Make predictions on the test set
y_pred_knn <- predict(knn_classifier, newdata = X_test)

# Evaluate the performance of the KNN model
accuracy <- confusionMatrix(y_pred_knn, y_test)$overall['Accuracy']
conf_matrix <- confusionMatrix(y_pred_knn, y_test)$table
classification_rep <- confusionMatrix(y_test, y_pred_knn)$byClass

print("Accuracy of the KNN Model on test data:", accuracy)
print("Confusion Matrix for KNN:", conf_matrix)
print("Classification Report for KNN:\n", classification_rep)
# Kappa Statistic
kappa_statistic <- kappa2(y_pred_knn, y_test)$value
print("Kappa Statistic for KNN:", kappa_statistic)
# ROC Curve for KNN
y_pred_proba_knn <- as.numeric(predict(knn_classifier, newdata = X_test, type = "prob")[, 2])
roc_knn <- roc(y_test, y_pred_proba_knn)

# Plot ROC curve for KNN
plot(roc_knn, col = "darkorange", main = "Receiver Operating Characteristic (ROC) Curve for KNN")
abline(a = 0, b = 1, lty = 2, col = "navy")
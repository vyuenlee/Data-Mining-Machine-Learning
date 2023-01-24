#-----------------------------------------------------------------------------
#
# Data Mining & Machine Learning
# Title: Dimension Reduction & Performance Evaluation on Housing Data Models
# By:  Vivian Yuen-Lee
#
#-----------------------------------------------------------------------------

# The data file contains information collected by the US Bureau of Census
# concerning housing in the area of Boston. The goal is to predict the median
# house price in new tracts based on information such as crime rate, pollution, 
# and number of rooms, etc.

# Load the required libraries
library(corrplot)         # library for correlations plot
library(forecast)         # library for accuracy measures

# Load the csv file
bhouse_df_all <- read.csv("BostonHousing.csv")
View(bhouse_df_all)           # take a look at the data
t(t(names(bhouse_df_all)))    # list the names of the columns
# Exclude the last column CAT..MEDV
bhouse_df <- bhouse_df_all[ ,-14]
t(t(names(bhouse_df)))    # list the names of the columns

# Create a multiple linear regression model as a function of CRIM, CHAS & RM
bhouse_lm <- lm(MEDV ~ CRIM+CHAS+RM, data = bhouse_df)
summary(bhouse_lm)

# Generate a prediction for a home that does not bound the Charles River, 
# has a crime rate of 0.1, and has an average of 6 rooms per house
bhouse_lmpred <- predict(bhouse_lm, data.frame("CHAS"=0, "CRIM"= 0.1, "RM"=6),
                         se.fit = TRUE)
bhouse_lmpred$fit         # predicted MEDV value
bhouse_lmpred$se.fit      # prediction error

# -------------------------- DIMENSION REDUCTION ------------------------------

# Display the correlations between predictors in a plot
corrplot(cor(bhouse_df), method = "number", type = "upper", diag = FALSE)
# According to this correlations plot, the target variable MEDV is negatively
# correlated to LSTAT and positively correlated to RM.
corrplot(cor(bhouse_df[ ,c("INDUS","NOX","TAX")]), method = "number")
# The three variables INDUS, NOX and TAX are all displaying strong positive 
# correlations.

# Display the correlations in a table format
cor(bhouse_df[ , -4])         # correlation matrix without the dummy variable
# Alternatively, the correlation plot above can also be used.
# The highly correlated paris include:
# •	INDUS and NOX (0.76)
# •	INDUS and DIS (-0.71)
# •	INDUS and TAX (0.72)
# •	AGE and NOX (0.73)
# •	AGE and DIS (-0.75)
# •	NOX and DIS (-0.77)
# •	TAX and RAD (0.91)

# Use stepwise regression (backward, forward, both) to reduce the remaining 
# predictors 
library(gains)          # load package for lift charts
# Partitioning the data
set.seed(123)   # set seed for reproducing the partition
n <- nrow(bhouse_df)
train_index <- sample(c(1:n), round(0.6*n))   # use 60% for training
valid_index <- setdiff(c(1:n), train_index)   # use the rest for validation
# Apply the randomly selected indices to create the training set
bhtrain <- bhouse_df[train_index, ]   
# Apply the remaining observations to the validation set
bhvalid <- bhouse_df[valid_index, ]

# Create a multiple linear regression model that includes all predictors
bhouse_lm_all <- lm(MEDV ~ ., data = bhtrain)

# Perform the 3 types of step regression: 1) Backward selection, 2) Forward 
# selection, and 3) Both
# 1) BACKWARD SELECTION
bhouse_back <- step(bhouse_lm_all, direction = "backward")
summary(bhouse_back)   # view model summary
# The output shows that the best model has AIC = 928.64 
# Backward selection yields a 10-predictor model: 
# MEDV ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + LSTAT
# Fit the linear model using the chosen list of predictors
bhouse_lmback <- lm(MEDV ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + 
                      PTRATIO + LSTAT,
                    data = bhtrain)
# Make predictions
bhouse_predback <- predict(bhouse_lmback, bhvalid)
# Compute the model accuracy
back_acc <- accuracy(bhouse_predback, bhvalid$MEDV)
back_acc
# 2) FORWARD SELECTION
# First, create an initial model with no predictors
bhouse_init <- lm(MEDV ~ 1, data = bhtrain)
# Use step() function to run forward selection
bhouse_for <- step(bhouse_init, 
                   scope = list(lower=bhouse_init, upper=bhouse_lm_all), 
                   direction = "forward")
summary(bhouse_for)   # view model summary
# The output shows that the best model has AIC = 928.64
# Forward selection yields a 10-predictor model:
# MEDV ~ LSTAT + RM + PTRATIO + CHAS + DIS + NOX + ZN + CRIM + RAD + TAX
# Fit the linear model using the chosen list of predictors
bhouse_lmfor <- lm(MEDV ~ LSTAT + RM + PTRATIO + CHAS + DIS + NOX + ZN + CRIM + 
                     RAD + TAX,
                   data = bhtrain)
# Make predictions
bhouse_predfor <- predict(bhouse_lmfor, bhvalid)
# Compute the model accuracy
for_acc <- accuracy(bhouse_predfor, bhvalid$MEDV)
for_acc

# 3) STEPWISE REGRESSION IN BOTH DIRECTIONS
bhouse_both <- step(bhouse_lm_all, direction = "both")
summary(bhouse_both)   # view model summary
# The output shows that the best model has AIC = 928.64
# Forward selection yields a 10-predictor model:
# MEDV ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + LSTAT
# Fit the linear model using the chosen list of predictors
bhouse_lmboth <- lm(MEDV ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + 
                      PTRATIO + LSTAT,
                   data = bhtrain)
# Make predictions
bhouse_predboth <- predict(bhouse_lmboth, bhvalid)
# Compute the model accuracy
both_acc <- accuracy(bhouse_predboth, bhvalid$MEDV)
both_acc

# ----------------------- PERFORMANCE COMPARISON ------------------------------
# Combine the results of all models into a single table
all_acc <- as.data.frame(rbind(back_acc, for_acc, both_acc))
rownames(all_acc) <- c("Backward Selection", "Forward Selection", 
                       "Both Directions")
all_acc

# Comparing the lift charts for all models
actual_MEDV <- bhvalid$MEDV
# Lift chart for backward selection
lift_back <- gains(actual_MEDV, bhouse_predback, group = 10)
plot(c(0, lift_back$cume.pct.of.total*sum(actual_MEDV))~c(0, lift_back$cume.obs), 
     xlab = "# of Cases",
     ylab = "Cumulative MEDV",
     main = "Lift Chart for Backward Selection",
     type = "l",
     lwd = 2)
# Add baseline to the lift chart
lines(c(0, sum(actual_MEDV)) ~ c(0, nrow(bhvalid)),
      col = "grey",
      lty = 2,
      lwd = 2)
# Lift chart for forward selection
lift_for <- gains(actual_MEDV, bhouse_predfor, group = 10)
plot(c(0, lift_for$cume.pct.of.total*sum(actual_MEDV))~c(0, lift_for$cume.obs), 
     xlab = "# of Cases",
     ylab = "Cumulative MEDV",
     main = "Lift Chart for Forward Selection",
     type = "l",
     lwd = 2)
# Add baseline to the lift chart
lines(c(0, sum(actual_MEDV)) ~ c(0, nrow(bhvalid)),
      col = "grey",
      lty = 2,
      lwd = 2)
# Lift chart for stepwise in both directions
lift_both <- gains(actual_MEDV, bhouse_predboth, group = 10)
plot(c(0, lift_both$cume.pct.of.total*sum(actual_MEDV))~c(0, lift_both$cume.obs), 
     xlab = "# of Cases",
     ylab = "Cumulative MEDV",
     main = "Lift Chart for Stepwise in Both Directions",
     type = "l",
     lwd = 2)
# Add baseline to the lift chart
lines(c(0, sum(actual_MEDV)) ~ c(0, nrow(bhvalid)),
      col = "grey",
      lty = 2,
      lwd = 2)

#-------------------------- K-NEAREST NEIGHBORS -------------------------------
# Perform a k-NN prediction with all 12 predictors. Try values of k from 1 to 5

# Load the required libraries
library(caret)
library(class)

# Partition the data into training (60%) and validation (40%) sets.
set.seed(321)   # set seed for reproducing the partition
n <- nrow(bhouse_df_all)
train_index2 <- sample(c(1:n), round(0.6*n))   # use 60% for training
valid_index2 <- setdiff(c(1:n), train_index)   # use the rest for validation
# Apply the randomly selected indices to create the training set
bhtrain2 <- bhouse_df_all[train_index2, -14]   # ignore the CAT..MEDV column
# Apply the remaining observations to the validation set
bhvalid2 <- bhouse_df_all[valid_index2, -14]   # ignore the CAT..MEDV column

# Setup and initialize the normalized data sets
bhtrain_norm <- bhtrain2                # create a copy of the data sets
bhvalid_norm <- bhvalid2
bhouse_all_norm <- bhouse_df_all
# Normalize using the preProcess() function in the caret package
norm_values <- preProcess(bhtrain2, method = c("center","scale"))
bhtrain_norm <- predict(norm_values, bhtrain2)
bhvalid_norm <- predict(norm_values, bhvalid2)
bhouse_all_norm <- predict(norm_values, bhouse_df_all)

# Initialize a new data frame for storing accuracy measure inside the for loop
accuracy_df <- data.frame(k = seq(1,5,1), RMSE = rep(0,5))

# Compute accuracy for different k on validation using a for loop
for(i in 1:5){
  knn_pred <- class::knn(train = bhtrain_norm[,-13],  
                         test = bhvalid_norm[,-13],   
                         cl = bhtrain2$MEDV, k = i)   # train target variables
  # Compare predictions with targets in the validation set & compute accuracy
  accuracy_df[i,2] <- RMSE(as.numeric(as.character(knn_pred)), bhvalid2$MEDV)
}
accuracy_df

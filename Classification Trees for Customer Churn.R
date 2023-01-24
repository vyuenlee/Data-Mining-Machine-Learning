#-----------------------------------------------------------------------------
#
# Data Mining & Machine Learning
# Title: Classification Trees and Rule-Based Classification Models for 
#        Customer Churn Analysis
# By: Vivian Yuen-Lee
#
#-----------------------------------------------------------------------------

# The purpose is to demonstrate the application of various classification 
# trees and rule-based models to the mlc_churn dataset from "modeldata".
# These rule-based algorithms include:
# - CART
# - Bagged trees
# - Random forest
# - C5.0

# Load the appropriate libraries
library(modeldata)      # library that contains the dataset
library(caret)          # for general model fitting
library(e1071)
library(pROC)           # for calculating ROC

# Load and store the mic_churn dataset
# This dataset came from the MLC++ machine learning software for modeling 
# customer churn.
data(mlc_churn)   # loading dataset
ccdata_all <- data.frame(mlc_churn)   # assigning it to object "ccdata"

# Examine the dataset
View(ccdata_all)   # open with the Data Viewer 
str(ccdata_all)   # examine its structure
# This dataset has 20 variables and 5000 observations.
# The target variable = churn (factor with 2 levels, yes/no) 
# There are 19 predictors, 2 of which are factors with 2 levels (no/yes), 1 of  
# which is a factor with 3 levels (area code), 1 of which is a factor with 51  
# levels (states abbrev.).  The remaining 15 predictors are numeric/integer.
# Next, generate a frequency table for the target variable, "churn"
table(ccdata_all$churn)
# As shown by this table, the dataset has a class imbalance: "no" is the 
# majority class and "yes" is the minority class.

# According to the variable importance functions for each model, the following
# predictors will be removed for their low importance: "state", "area_code",
# "account_length", total_night_calls" & "total_eve_calls" in order to simplify
# model fitting and the predicting processes.
ccdata <- ccdata_all[, !names(mlc_churn) %in% c("state",
                                                "area_code",
                                                "account_length",
                                                "total_night_calls",
                                                "total_eve_calls")]
# Check the summarized details for the dataset
summary(ccdata)

# Split the dataset into training and testing subset using a stratified random
# split and a 70%-30% ratio.
set.seed(120)
# First, create index to split using the 70-30 ratio
index <- createDataPartition(ccdata$churn, 
                             p = 0.7, 
                             list = FALSE)
# Apply this index to generate the training subset
cctrain <- ccdata[index,]
# Apply the remaining observations to generate the test subset
cctest <- ccdata[-index,]

# --------------------------------- CART ---------------------------------
# "rpart" is the primary package in R for creating single classification trees.
# The rpart() function can be used to fit a CART model.
# By default, rpart uses gini impurity to select splits to perform 
# classification

library(rpart)  

# Fit the CART model to the predictors
# This automaticlly grows and prunes the tree using internal cross-validation
cartModel <- rpart(churn ~., 
                   data = cctrain)   
# Prune back the tree by selecting a tree size that minimizes the cross-
# valication error, "xerror".
printcp(cartModel)   # take a look at the CP Table
cartModelp <- prune(cartModel, 
                    cp = cartModel$cptable[which.min(cartModel$cptable
                                                     [ ,"xerror"]),"CP"])
# Examine the model output
cartModelp
# This shows the split variable/value and how many samples were partitioned 
# into the branches as well as the class probabilities.

# Check which variable is the most influential in the model
cartVar <- data.frame(cartModelp$variable.importance)
cartVarPlot <- cartVar[order(cartVar[ , 1]), , drop = FALSE]
barplot(cartVarPlot[ ,1],
        names.arg = rownames(cartVarPlot),
        cex.names = 0.5,
        horiz = TRUE,
        las = 1,
        col = "steelblue",
        main = "Variable Importance for CART Model",
        xlab = "Variable Importance")
# Here we can see that 'total_day_charge' and "total_day_minutes" are the most 
# important predictors.

# Visualize the pruned tree using the plot function in the "rpart.plot" package
library(rpart.plot)
rpart.plot(cartModelp, type = 4, extra = 1)

# Generate predictions on the test subset
# First, predict the "churn" class. Output will be a factor vector of the 
# predicted class levels
cartPredClass <- predict(cartModelp, 
                         newdata = cctest,
                         type = "class")
# Next, predict the probabilities for each class. Output will be a matrix of 
# probabilities for the "yes" and "no" classes.
cartPredProb <- predict(cartModelp,
                        newdata = cctest,
                        type = "prob")
# Combine the two outputs and see the first 6 predictions
# Convert "cartChurnClass" to a data frame to get the class names
cartPred <- cbind(as.data.frame(cartPredClass) , cartPredProb)
head(cartPred)

# To evaluate model performance, a common method fo describing the performance
# of a classification model is the confusion matrix.  
cartCM <- confusionMatrix(data = cartPred$cartPredClass,
                          reference = cctest$churn)
# The confusionMatrix() command gives a confusion matrix and overall statistics
# including Accuracy, Kappa statistics, Sensitivity, and Specificity, etc.
cartCM   # take a look at the results

# Next, evaluate class probabilities using the ROC curve
cartROC <- roc(response = cctest$churn,
               predictor = cartPred$no)  
plot(cartROC,   # plotting the curve
     legacy.axes = TRUE, 
     ylim = c(0,1),
     main = "ROC Curve for CART")
cartAUC <- auc(cartROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
cartMetrics <- data.frame(Accuracy = cartCM$overall["Accuracy"], 
                          Kappa = cartCM$overall["Kappa"],
                          Sensitivity = cartCM$byClass["Sensitivity"], 
                          Specificity = cartCM$byClass["Specificity"],
                          ROC_AUC = cartAUC, 
                          row.names = "CART")
cartMetrics   # Take a look at all metrics


# ---------------------------- BAGGED TREE ----------------------------
# The ipred package has the function bagging() for generating bagged trees and
# the function can be applied to classification.

library(ipred)   

# Fit the bagged tree model to the predictors
# These trees are grown deep and are not pruned. 
set.seed(321)   # set seed to make bootstrapping reproducible
btreeModel <- bagging(churn ~., 
                      data = cctrain,
                      coob = TRUE,
                      nbagg = 100)   
# Number of bootstrap replications is set to 100
btreeModel   # take a look at the model details
summary(btreeModel)   # take a look at more details
# Generally, bagged trees are left unpruned. Each individual tree has high 
# variance, but low bias. Averaging the trees will reduce the variance.

# Check which variable is the most influential in the model
btreeVar <- data.frame(varImp(btreeModel))
btreeVarPlot <- btreeVar[order(btreeVar$Overall), , drop = FALSE]
barplot(btreeVarPlot$Overall,
        names.arg = rownames(btreeVarPlot),
        cex.names = 0.5,
        horiz = TRUE,
        las = 1,
        col = "steelblue",
        main = "Variable Importance for Bagged Tree",
        xlab = "Variable Importance")
# We see that 'total_day_minutes' and "total_day_charge" are the most 
# important predictors.

# Generate predictions on the test subset
# First, predict the "churn" class. Output will be a factor vector of the 
# predicted class levels
btreePredClass <- predict(btreeModel, 
                          newdata = cctest,
                          type = "class")
# Next, predict the probabilities for each class. Output will be a matrix of 
# probabilities for the "yes" and "no" classes.
btreePredProb <- predict(btreeModel,
                         newdata = cctest,
                         type = "prob")
# Combine the two outputs and see the first 6 predictions
# Convert "cartChurnClass" to a data frame to get the class names
btreePred <- cbind(as.data.frame(btreePredClass) , btreePredProb)
head(btreePred)

# To evaluate model performance, first examine the confusion matrix
btreeCM <- confusionMatrix(data = btreePred$btreePredClass,
                           reference = cctest$churn)
btreeCM   # take a look at the results

# Next, evaluate class probabilities using the ROC curve
btreeROC <- roc(response = cctest$churn,
                predictor = btreePred$no)  
plot(btreeROC,   # plotting the curve
     legacy.axes = TRUE, 
     ylim = c(0,1),
     main = "ROC Curve for Bagged Tree")
btreeAUC <- auc(btreeROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
btreeMetrics <- data.frame(Accuracy = btreeCM$overall["Accuracy"], 
                           Kappa = btreeCM$overall["Kappa"],
                           Sensitivity = btreeCM$byClass["Sensitivity"], 
                           Specificity = btreeCM$byClass["Specificity"],
                           ROC_AUC = btreeAUC, 
                          row.names = "Bagged Tree")
btreeMetrics   # Take a look at all metrics

# ---------------------------- RANDOM FOREST ----------------------------
# Random forest functions are provided by the randomForest package. 

library(randomForest)   

# The tuning parameter "mtry" is the number of variables that are selected at
# each split of each tree. We will use the train() function in the "caret"
# package to find the optimal "mtry".
rfModel <- train(churn ~ ., 
                 data = cctrain,
                 method = "rf",
                 trControl = trainControl("cv", number=10))
rfModel$bestTune   # the optimized mtry = 14
rfModel   # take a look at other model details

# Check which variable is the most influential in the model
rfVar <- data.frame(varImp(rfModel)$importance)
rfVarPlot <- rfVar[order(rfVar$Overall), , drop = FALSE]
barplot(rfVarPlot$Overall,
        names.arg = rownames(rfVarPlot),
        cex.names = 0.5,
        horiz = TRUE,
        las = 1,
        col = "steelblue",
        main = "Variable Importance for Random Forest",
        xlab = "Variable Importance")
# We see that 'total_day_charge' and "total_day_minutes" are the most 
# important predictors, followed by "number_customer_service_calls".

# Generate predictions on the test subset
# First, predict the "churn" class. Output will be a factor vector of the 
# predicted class levels
rfPredClass <- predict(rfModel, 
                       newdata = cctest,
                       type = "raw")
# Next, predict the probabilities for each class. Output will be a matrix of 
# probabilities for the "yes" and "no" classes.
rfPredProb <- predict(rfModel,
                      newdata = cctest,
                      type = "prob")
# Combine the two outputs and see the first 6 predictions
# Convert "cartChurnClass" to a data frame to get the class names
rfPred <- cbind(as.data.frame(rfPredClass) , rfPredProb)
head(rfPred)

# To evaluate model performance, a confusion matrix is first generated
rfCM <- confusionMatrix(data = rfPred$rfPredClass,
                        reference = cctest$churn)
rfCM   # take a look at the results

# Next, evaluate class probabilities using the ROC curve
rfROC <- roc(response = cctest$churn,
             predictor = rfPred$no)  
plot(rfROC,   # plotting the curve
     legacy.axes = TRUE, 
     ylim = c(0,1),
     main = "ROC Curve for Random Forest")
rfAUC <- auc(rfROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
rfMetrics <- data.frame(Accuracy = rfCM$overall["Accuracy"], 
                        Kappa = rfCM$overall["Kappa"],
                        Sensitivity = rfCM$byClass["Sensitivity"], 
                        Specificity = rfCM$byClass["Specificity"],
                        ROC_AUC = rfAUC, 
                        row.names = "Random Forest")
rfMetrics   # Take a look at all metrics

# --------------------------------- C5.0 ---------------------------------
# A single C5.0 tree can be created using the C5.0() function within the C50 
# package
# The C5.0 algorithm uses the post-pruning method to control the size of the 
# tree. It first grows an overfitting large tree that contains all the 
# possibilities of partitioning. Then, it cuts out nodes and branches with 
# little effect on classification errors.

library(C50)  

# Fit the C5.0 model to the predictors
c5Model <- C5.0(churn ~., 
                data = cctrain)
# Examine the model output
c5Model
summary(c5Model)

# Visualizing the tree
plot(c5Model)
# This is a large tree and unfortunately the visualization is not very 
# customizable.

# Check which variable is the most influential in the model
c5Var <- data.frame(C5imp(c5Model))
c5VarPlot <- c5Var[order(c5Var$Overall), , drop = FALSE]
barplot(c5VarPlot$Overall,
        names.arg = rownames(c5VarPlot),
        cex.names = 0.5,
        horiz = TRUE,
        las = 1,
        col = "steelblue",
        main = "Variable Importance for C5.0 Model",
        xlab = "Variable Importance")
# Here we can see that "total_day_minutes", "number_customer_service_calls", 
# "total_eve_charge" and "international_plan" are by far the most important
# predictors.

# Generate predictions on the test subset
# First, predict the "churn" class. Output will be a factor vector of the 
# predicted class names
c5PredClass <- predict(c5Model, 
                       newdata = cctest,
                       type = "class")
# Next, predict the probabilities for each class. Output will be a matrix of 
# probabilities for the "yes" and "no" classes.
c5PredProb <- predict(c5Model,
                      newdata = cctest,
                      type = "prob")
# Combine the two outputs and see the first 6 predictions
# Convert "cartChurnClass" to a data frame to get the class names
c5Pred <- cbind(as.data.frame(c5PredClass) , c5PredProb)
head(c5Pred)

# To evaluate model performance, generate a confusion matrix  
c5CM <- confusionMatrix(data = c5Pred$c5PredClass,
                        reference = cctest$churn)
c5CM   # take a look at the results
# Specificity is high but sensitivity is low because many 'yes's' were 
# predicted as 'no's' according to the confusion matrix.

# Next, evaluate class probabilities using the ROC curve
c5ROC <- roc(response = cctest$churn,
             predictor = c5Pred$no)  
plot(c5ROC,   # plotting the curve
     legacy.axes = TRUE, 
     ylim = c(0,1),
     main = "ROC Curve for C5.0 Tree")
c5AUC <- auc(c5ROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
c5Metrics <- data.frame(Accuracy = c5CM$overall["Accuracy"], 
                        Kappa = c5CM$overall["Kappa"],
                        Sensitivity = c5CM$byClass["Sensitivity"], 
                        Specificity = c5CM$byClass["Specificity"],
                        ROC_AUC = c5AUC, 
                        row.names ="C5.0")
c5Metrics   # Take a look at all metrics

#-------------------------------- RESULTS -------------------------------- 
# Combining performance metrics for all classification trees:
rbind(cartMetrics,
      btreeMetrics,
      rfMetrics,
      c5Metrics)
# Next, combine all ROC curves on the same plot
plot(cartROC, col = "red", main = "Comparison of ROC Curves")   # base plot
plot(btreeROC, add = TRUE, col = "blue", lty = 2)
plot(rfROC, add = TRUE, col = "green")
plot(c5ROC, add = TRUE, col = "orange", lty = 2)
legend("bottomright",
       legend = c("CART","Bagged Trees", "Random Forest", "C5.0"),
       col = c("red", "blue", "green", "orange"),
       lty = 1:2,
       cex = 0.8,
       text.font = 0.5)
# Based on this summary, the bagged tree and random forest have comparable 
# performance. The basic pruned CART performed slightly better than the C5.0
# model.  
# Sensitivity values are lower than Specificity for all models as a result of
# class imbalance in the target variable.  However, When compared to the models 
# evaluated last week (i.e. logistic regression, linear discriminant analysis,
# and K-nearest neighbors), these tree-based algorithms appear to handle class
# imbalance much more effectively.  The gap between Sensitivity and Specificity
# is significantly smaller than last week's model even without oversampling.



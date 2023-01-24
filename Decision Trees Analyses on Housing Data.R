#------------------------------------------------------------------------------
#
# Data Mining & Machine Learning
# Title: Various Decision Trees for Housing Data Analysis
# By: Vivian Yuen-Lee
#
#------------------------------------------------------------------------------

# Ths purpose is to implement different decision tree algorithms using the 
# BostonHousing dataset from the "mlbench" package.
# The tree algorithms include:
# - Conditional inference tree
# - Bagged tree
# - Random forests
# - Conditional inference forests
# - Boosted trees
# - Cubist

# Load the appropriate libraries
library(caret)
library(e1071)
library(mlbench)
library(party)          # library for conditional inference tree
library(ipred)          # library for bagging
library(randomForest)   # library for random forests
library(gbm)            # library for gradient boosted trees
library(Cubist)         # library for cubist regression

# Load and store the BostonHousing dataset
data(BostonHousing)   # loading dataset
bhdata <- BostonHousing   # assigning it to object "bhdata"

# Examine the dataset
View(bhdata)   # open with the Data Viewer 
str(bhdata)   # examine its structure
# Note that tree-based models require either numeric or factor data types, and
# this dataset does not require any conversions.
# The target variable = medv (median value of owner-occupied homes in $1000's) 

# Split the dataset into training and testing subset using a stratified random
# split and the 80%-20% ratio.
set.seed(100)
# First, create index to split using the 80-20 ratio
index <- createDataPartition(bhdata$medv, p=0.8, list=FALSE)
# Apply this index to generate the training subset
bhtrain <- bhdata[index,]
# Apply the remaining observations to generate the test subset
bhtest <- bhdata[-index,]

# -------------------- CONDITIONAL INFERENCE TREE (CIT) --------------------
# The CIT is an extension of the basic decision tree type "classification and
# regression tree (CART).  The CIT selects and splits variables using 
# significance tests.
# In R, the "party" package makes splits based on the conditional inference
# framework using the ctree() function

# First, generate the conditional inference tree
citModel <- ctree(medv ~., data=bhtrain)
# Examine the model details
citModel
# This conditional inference tree has 16 terminal nodes
# Make a tree plot
plot(citModel, main="Conditional Inference Tree for Boston's Median Home Value")
# Make predictions on the test subset
citPred <- predict(citModel, bhtest)
head(citPred)   # see the first 6 predictions
# Collect the observed and predicted values into a data frame
# Note that citPred is a matrix/array
citValues <- data.frame(obs=bhtest$medv, pred=citPred[,])
# Use the caret function defaultSummary() to estimate test performance
citPerformance <- defaultSummary(citValues)
citPerformance

# ---------------------------- BAGGED TREE ----------------------------
# Bagged trees (bagging = boostrap aggregation) use bootstrapping along with 
# decision tree models to combine the predictions from multiple models.
# The ipred package has the function bagging() for generating bagged trees.

# Create the bagged tree model using training data
bagModel<- bagging(medv ~., data=bhtrain, 
                   coob=TRUE,   # "Compute out-of-bag error" is set to true
                   nbagg=100)
# Number of bootstrap replications is chosen to be 100 since this number gives
# the most optimal prediction accuracy & goodness-of-fit.
bagModel   # check out the model details

# Make predictions on the test subset
bagPred <- predict(bagModel, bhtest)
head(bagPred)   # see the first 6 predictions
# Collect the observed and predicted values into a data frame
bagValues <- data.frame(obs=bhtest$medv, pred=bagPred)
# Use the caret function defaultSummary() to estimate test performance
bagPerformance <- defaultSummary(bagValues)
bagPerformance

# ---------------------------- RANDOM FOREST ----------------------------
# Random forests reduces correlation among predictors by adding randomness to 
# the tree construction process.  For each split selection, trees are built
# using a random subset of the original predictors.
# In R, random forest functions are provided by the package randomForest.

# Create the random forest model using training data
rfModel <- randomForest(medv ~., data=bhtrain, 
                        importance=TRUE,   # enable variable importance scoring
                        mtry=6)
# The tuning parameter "mtry" is the number of variables that are selected at 
# each split of each tree.  It is manually set to 6 for the best residual error 
# and % variance explained.
rfModel   # take a look at the model details

# Alternatively, the "mtry" parameter can be tuned using resampling methods
rfModel_t <- train(medv~., data=bhtrain,
                  method="rf",
                  trControl=trainControl("cv", number=10))
rfModel_t$bestTune   # the optimized mtry = 7
rfModel_t   # take a look at the model details

# Use the tuned model to make predictions on the test subset
rfPred_t <- predict(rfModel_t, bhtest)
head(rfPred_t)   # see the first 6 predictions
# Collect the observed and predicted values into a data frame
rfValues_t <- data.frame(obs=bhtest$medv, pred=rfPred_t)
# Use the caret function defaultSummary() to estimate test performance
rfPerformance <- defaultSummary(rfValues_t)
rfPerformance

# -------------------- CONDITIONAL INFERENCE FOREST --------------------
# Conditional inference forest is an implementation of the random forest and
# bagging ensemble algorithms with conditional inference trees as base learners
# The cforest() function in the party package can be used to build forests 
# with conditional inference trees.

# First, generate the conditional inference forest
cifModel <- cforest(medv ~., data=bhtrain)
cifModel

# Similar to the random forest, the tuning "mtry" parameter can be optimized 
# using resampling methods
cifModel_t <- train(medv~., data=bhtrain,
                   method="cforest",
                   trControl=trainControl("cv", number=10))
cifModel_t$bestTune   # the optimized mtry = 7
cifModel_t   # take a look at the model details

# Next, use the tuned model to make predictions on the test subset
cifPred_t <- predict(cifModel_t, bhtest)
head(cifPred_t)   # see the first 6 predictions
# Collect the observed and predicted values into a data frame
cifValues_t <- data.frame(obs=bhtest$medv, pred=cifPred_t)
# Use the caret function defaultSummary() to estimate test performance
cifPerformance <- defaultSummary(cifValues_t)
cifPerformance

# ---------------------------- BOOSTED TREE ----------------------------
# The boosted trees technique combines many regression tree models, weighted
# according to their prediction accuracy, into an aggregate model.
# The gbm package (which stands for gradient boosting machine) provides the  
# most widely used functions for boosting regression trees.

# Create a regression model with the gbm() function and training data
gbtModel <- gbm(medv ~., data=bhtrain,
                distribution="gaussian")
# Distribution is set to Gaussian for a continuous response
gbtModel
# 100 iterations were performed.  Model indicated that 10 out of 13 predictors
# had non-zero influence (ie. some degree of significance).

# Gradient boosted trees have 2 main tuning parameters: tree depth & number of 
# iterations. We will attempt to optimize these parameters with a search grid.
# First, define a tuning grid
gbtGrid <- expand.grid(.interaction.depth = seq(1, 10, by = 2),
                       .n.trees = seq(100, 1000, by=50),
                       .shrinkage = c(0.01, 0.1),
                       .n.minobsinnode = 10)
# Next, we will use resampling to tune these parameters
gbtModel_t <- train(medv~., data=bhtrain,
                    method="gbm",
                    tuneGrid=gbtGrid,
                    verbose=FALSE) 
gbtModel_t$bestTune   # check out the tuned parameters
gbtModel_t   # take a look at the model details

# Next, use the tuned model to make predictions on the test subset
gbtPred_t <- predict(gbtModel_t, bhtest)
head(gbtPred_t)   # see the first 6 predictions
# Collect the observed and predicted values into a data frame
gbtValues_t <- data.frame(obs=bhtest$medv, pred=gbtPred_t)
# Use the caret function defaultSummary() to estimate test performance
gbtPerformance <- defaultSummary(gbtValues_t)
gbtPerformance

# -------------------------------- CUBIST --------------------------------
# In a Cubist model, a tree is built with linear regression models at the 
# terminal nodes as well as each intermediate step of the tree. Prediction is 
# made using the linear regression model at the terminal node, but smoothed 
# over other linear models in the previous node of the tree.
# An R package "Cubist" was created to perform cubist regression.  Note that
# the function cubist() does not have a formula method; it follows the 
# cubist(x,y) syntax.

# First, create a simple rule-based model with a single committee
cubModel <- cubist(bhtrain[,1:13], bhtrain$medv)
cubModel   # take a look at the model details

# Alternative, the train function in the caret package can tune the model over 
# values of committees and neighbors through resampling
cubModel_t <- train(medv~., data=bhtrain,
                    method="cubist",
                    trControl=trainControl("cv", number=10))
cubModel_t$bestTune   # the optimized committees=20, neighbors=5
cubModel_t   # take a look at the model details

# Next, use the tuned model to make predictions on the test subset
# Use the "neighbors" argument to set instance-based adjustment.
cubPred_t <- predict(cubModel_t, bhtest, neighbors=1)
head(cubPred_t)   # see the first 6 predictions
# Collect the observed and predicted values into a data frame
cubValues_t <- data.frame(obs=bhtest$medv, pred=cubPred_t)
# Use the caret function defaultSummary() to estimate test performance
cubPerformance <- defaultSummary(cubValues_t)
cubPerformance

# ----------------------------- COMPARISON -----------------------------
# Finally, combine the performance metrics for all tree models in a single
# table for better comparison:
rbind("Conditional Inference Tree" = citPerformance,
      "Bagged Tree" = bagPerformance,
      "Random Forest" = rfPerformance,
      "Conditional Inference Forest" = cifPerformance,
      "Gradient Boosted Tree" = gbtPerformance, 
      "Cubist" = cubPerformance)
# Based on this table, the Cubist regression model provided the highest 
# predictive accuracy and the best goodness-of-fit, followed closely by the 
# Gradient Boosted Tree model.



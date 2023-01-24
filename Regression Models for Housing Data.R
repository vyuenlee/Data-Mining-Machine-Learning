#------------------------------------------------------------------------------
#
# Data Mining & Machine Learning
# Title: Various Regression Models for Housing Data Analysis
# By: Vivian Yuen-Lee
#
#------------------------------------------------------------------------------

# Ths purpose is to demonstrate the application of different regression  
# algorithms to the BostonHousingdataset from the "mlbench" package.
# These algorithms include:
# - Linear regression
# - Partial least squares
# - Ridge regression (penalized regression)
# - Elasticnet regression (penalized regression)
# - Multivariate adaptive regression splines
# - Support vector machines (radial & polynomial)
# - k-nearest neighbors

# Load the appropriate libraries
library(caret)
library(e1071)
library(mlbench)
library(corrplot)

# Load and store the BostonHousing dataset
data(BostonHousing)   # loading dataset
bhdata <- BostonHousing   # assigning it to object "bhdata"

# Examine the dataset
View(bhdata)   # open with the Data Viewer 
str(bhdata)   # examine its structure
# The dataset contains 506 observations of 14 variables.
# The target variable = medv (median value of owner-occupied homes in $1000's) 

# Examine the relationship between variables
# cor() can only be applied to numeric columns, hence omit column 4 "chas"
corrplot(cor(bhdata[,-4]))

# Split the dataset into training and testing subset using a stratified random
# split and the 80%-20% ratio.
set.seed(100)
# First, create index to split using the 80-20 ratio
index <- createDataPartition(bhdata$medv, p=0.8, list=FALSE)
# Apply this index to generate the training subset
bhtrain <- bhdata[index,]
# Apply the remaining observations to generate the test subset
bhtest <- bhdata[-index,]

# ------------------------- LINEAR REGRESSION -------------------------
# Perform Linear Regression:
# First, generate a linear model with the training data 
lmModel <- lm(medv ~ ., data=bhtrain)
summary(lmModel)   # show the model stats
# Next, make predictions on the test dataset
lmPred <- predict(lmModel, bhtest)
head(lmPred)   # check out the first 6 predictions
# Collect the observed and predicted values into a data frame
lmValues <- data.frame(obs=bhtest$medv, pred=lmPred)
# Use the caret function defaultSummary() to estimate the test performance
LM_Performance <- defaultSummary(lmValues)
LM_Performance

# ----------------------- PARTIAL LEAST SQUARES -----------------------
# Generate Two Variations of Partial Least Squares: PLS & SIMPLS
# Partial Least Squares (PLS):
# The pls package has functions for PLS and PCR
library(pls)  # load the pls package
# Generate the model using training data with the plsr() function, adding a 
# standard 10-fold cross-validation
plsModel <- plsr(medv ~ ., data=bhtrain, validation="CV")
summary(plsModel)   # show the model stats
# Model shows that 3 components explained close to 99% of the variance
# 10 components explained all 100% of the variance
# Next, make predictions on the test data
plsPred <- predict(plsModel, bhtest,ncomp=10)   # consider 10 components
# Collect the observed and predicted values into a data frame
# Note that plsPred is an array
plsValues <- data.frame(obs=bhtest$medv, pred=plsPred[,,])
# Use the function defaultSummary() to estimate the test performance
PLS_Performance <- defaultSummary(plsValues)
PLS_Performance

# SIMPLS:
# Generate the model using training data with the plsr() function, adding a 
# standard 10-fold cross-validation  
simModel <- plsr(medv ~ ., data=bhtrain, validation="CV", method="simpls")
summary(simModel)   # show the model stats
# Model shows that 3 components explained close to 99% of the variance
# 10 components explained all 100% of the variance
# Make predictions on the test data
simPred <- predict(simModel, bhtest, ncomp=10)   # consider 10 components
# Collect the observed and predicted values into a data frame
# Note that simPred is an array
simValues <- data.frame(obs=bhtest$medv, pred=simPred[,,])
# Use the function defaultSummary() to estimate the test performance
SIMPLS_Performance <- defaultSummary(simValues)
SIMPLS_Performance

# ------------------------ PENALIZED REGRESSION ------------------------
# Create Two Penalized Regression Models: Ridge & Elastic Net Regressions
# Ridge regression can be created using glmnet() function in the glmnet package
library(glmnet)   # load the glmnet package
# Find the best lambda using cross-validation
x <- model.matrix(medv~., data=bhtrain)[,-1]   # define the predictor variable
y <- bhtrain$medv   # define the output variable
set.seed(100)
cv <- cv.glmnet(x, y)   # use cv.glmnet to optimize lamdba
# Store the best lambda value
ridgeLambda <- cv$lambda.min
ridgeLambda  # take a look at the selected value
# Generate the ridge regression model using training data and the above lambda
ridgeModel <- glmnet(x, y, alpha=0, lambda=ridgeLambda)
# Make predictions on the test data
xtest <- model.matrix(medv~., bhtest)[,-1]   # define predictors in test set
ridgePred <- predict(ridgeModel, xtest)
# Collect the observed and predicted values into a data frame
ridgeValues <- data.frame(obs=bhtest$medv, pred=ridgePred[,])
# Use the function defaultSummary() to estimate the test performance
Ridge_Performance <- defaultSummary(ridgeValues)
Ridge_Performance

# Generate the elastic net model with training data
# Use caret workflow to invoke the glmnet package and set trControl argument to
# automatically select the optimal tuning parameters "alpha" & "lambda".
enModel <- train(medv~., data=bhtrain, 
                 method="glmnet", 
                 trControl=trainControl("cv", number=10))
enModel$bestTune   # check out the final tuning parameters
enModel   # check out the model details
# Make predictions on the test data
enPred <- predict(enModel, bhtest)
# Collect the observed and predicted values into a data frame
enValues <- data.frame(obs=bhtest$medv, pred=enPred)
# Use the function defaultSummary() to estimate the test performance
EN_Performance <- defaultSummary(enValues)
EN_Performance

# --------------- MULTIVARIATE ADAPTIVE REGRESSION SPLINE ---------------
# Generate a Multivariate Adaptive Regression Spline Model (MARS):
library(earth)   #loading the required library
# First, generate the MARS model with 10-fold cross-validation
marsModel <- earth(medv~., data=bhtrain, nfold=10)
summary(marsModel)   # take a look at the model details
# This model uses the internal GCV technique for model selection.
# Next, make predictions on the test data
marsPred <- predict(marsModel, bhtest) 
# Collect the observed and predicted values into a data frame
# Note that marsPred is a matrix/array
marsValues <- data.frame(obs=bhtest$medv, pred=marsPred[,])
# Use the function defaultSummary() to estimate the test performance
MARS_Performance <- defaultSummary(marsValues)
MARS_Performance

# ----------------------- SUPPORT VECTOR MACHINE -----------------------
# Generate Two Variations of Support Vector Machines: Radial & Polynomial
# Support Vector Machine (SVM) - Radial
# Since values of the cost and kernal parameters are unknown, they can be 
# estimated through resampling (using the train function). 
library(kernlab)   # loading the required library
# Generate the model using training data and 10-fold cross-validation
svmrModel <- train(medv~., data=bhtrain, 
                 method="svmRadial", 
                 trControl=trainControl("cv", number=10))
svmrModel$bestTune   # examine the final tuning parameters
svmrModel   # look at the model details
# Make predictions on the test data
svmrPred <- predict(svmrModel, bhtest)
# Collect the observed and predicted values into a data frame
svmrValues <- data.frame(obs=bhtest$medv, pred=svmrPred)
# Use the function defaultSummary() to estimate the test performance
SVMradial_Performance <- defaultSummary(svmrValues)
SVMradial_Performance

# Support Vector Machine (SVM) - Polynomial
# Generate the model using the training dataset and 10-fold cross-validation
svmpModel <- train(medv~., data=bhtrain, 
                   method="svmPoly", 
                   trControl=trainControl("cv", number=10))
svmpModel$bestTune   # check out the final tuning parameters
svmpModel   # check out the model details
# Make predictions on the test data
svmpPred <- predict(svmpModel, bhtest)
# Collect the observed and predicted values into a data frame
svmpValues <- data.frame(obs=bhtest$medv, pred=svmpPred)
# Use the function defaultSummary() to estimate the test performance
SVMpoly_Performance <- defaultSummary(svmpValues)
SVMpoly_Performance

# ------------------------ K-NEAREST NEIGHBORS ------------------------
# Create a K-Nearest Neighbors (KNN) Model:
# The parameter K will be estimated through resampling
# First, generate the model using the training data
# Parameter tuning is performed using standard 10-fold cross-validation
knnModel <- train(medv~., data=bhtrain, 
                   method="knn", 
                   trControl=trainControl("cv", number=10))
knnModel$bestTune   # check out the best K value
knnModel   # examine the model details
# Note that the model selected 5 neighbors based on the smallest value of RMSE
# Make predictions on the test data
knnPred <- predict(knnModel, bhtest)
# Collect the observed and predicted values into a data frame
knnValues <- data.frame(obs=bhtest$medv, pred=knnPred)
# Use the function defaultSummary() to estimate the test performance
KNN_Performance <- defaultSummary(knnValues)
KNN_Performance

# The most important hyperparameter for KNN is the number of neighbors, K
# An alternative tuning strategy is to create a GRID SEARCH:
# Create a new resampling method
gridcv <- trainControl(method = "repeatedcv",   # 5-fold with 3 repeats
                      number = 5,
                      repeats = 3,
                      verboseIter = FALSE,
                      returnData = FALSE)
# Create the hyperparameter grid search
hypergrid <- expand.grid(k = floor(seq(1, nrow(bhtrain)/3, length.out = 20)))
# Apply to the model
knngModel <- train(medv~., data=bhtrain, 
                  method="knn", 
                  trControl=gridcv,
                  tuneGrid=hypergrid)
knngModel$bestTune   # check out the final value for K neighbors
knngModel   # check out the model details
# Make predictions on the test data
knngPred <- predict(knngModel, bhtest)
# Collect the observed and predicted values into a data frame
knngValues <- data.frame(obs=bhtest$medv, pred=knngPred)
# Use the function defaultSummary() to estimate the test performance
KNNgrid_Performance <- defaultSummary(knngValues)
KNNgrid_Performance

# A third tuning strategy  is to create a RANDOM SEARCH:
set.seed(123)
# Define a new resampling method
randomcv <- trainControl(method = "repeatedcv",   # 5-fold with 3 repeats
                         number = 5,
                         repeats = 3,
                         search = "random",    # perform random search
                         verboseIter = FALSE,
                         returnData = FALSE)
# Apply to the model
knnrModel <- train(medv~., data=bhtrain, 
                   method="knn", 
                   trControl=randomcv)
knnrModel$bestTune   # check out the final value for K neighbors
knnrModel   # check out the model details
# Make predictions on the test data
knnrPred <- predict(knnrModel, bhtest)
# Collect the observed and predicted values into a data frame
knnrValues <- data.frame(obs=bhtest$medv, pred=knnrPred)
# Use the function defaultSummary() to estimate the test performance
KNNrandom_Performance <- defaultSummary(knnrValues)
KNNrandom_Performance

# Now, combine the performance metrics for all regression models for a better
# comparison:
rbind(LM_Performance, 
      PLS_Performance, 
      SIMPLS_Performance, 
      Ridge_Performance, 
      EN_Performance,
      MARS_Performance,
      SVMradial_Performance,
      SVMpoly_Performance,
      KNN_Performance,
      KNNgrid_Performance,
      KNNrandom_Performance)
# Based on this table, the Support Vector Machine (polynomial) model provided
# the highest prediction accuracy and the best goodness-of-fit.

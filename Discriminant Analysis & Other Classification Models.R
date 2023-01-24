#-----------------------------------------------------------------------------
#
# Data Mining & Machine Learning
# Title: Discriminant Analysis & Linear / Non-linear Classification Models
#        For Customer Churn Analysis
# By: Vivian Yuen-Lee
#
#-----------------------------------------------------------------------------

# The purpose is to demonstrate the application of different classification 
# models to the mlc_churn dataset from the "modeldata" package.
# These classification algorithms include:
# - Logistic regression
# - Linear discriminant analysis
# - Partial least squares discriminant analysis
# - Penalized logistic regression
# - Penalized LDA
# - Nearest shrunken centroids
# - Non-linear discriminant analysis
# - Support vector machines
# - K-nearest neighbors
# - Naive Bayes

# Load the appropriate libraries
library(modeldata)      # library that contains the dataset
library(caret)
library(e1071)
library(pROC)

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
# Next, generate a frequency table for the target variable, 'churn'
table(ccdata_all$churn)
# As shown by this table, the dataset has a class imbalance: 'no' is the 
# majority class and 'yes' is the minority class.

# Next, check for missing data
sum(is.na(ccdata_all))
# There is no missing data, so imputation not required.
# Then, remove predictors that are not useful ('state' & 'area_code')
ccdata <- ccdata_all[, !names(mlc_churn) %in% c("state","area_code")]
# Check the summarized details for the dataset
summary(ccdata)

# Split the dataset into training and testing subset using a stratified random
# split and a 70%-30% ratio.
set.seed(123)
# First, create index to split using the 70-30 ratio
index <- createDataPartition(ccdata$churn, p=0.7, list=FALSE)
# Apply this index to generate the training subset
cctrain <- ccdata[index,]
# Apply the remaining observations to generate the test subset
cctest <- ccdata[-index,]

# ------------------------ LOGISTIC REGRESSION ------------------------
# The glm function is commonly used to fit logistic regression models.
# Recall that 'churn' is a factor of 2 levels, Level 1 = 'yes' & 2 = 'no'
# The function treats the 2nd factor level as the event of interest.
# For other predictor variables that are factors, the glm function will convert
# each factor level into a dummy binary variable of 1's and 0's.

# Generate the logistic regression model
logModel <- glm(churn ~., 
                data=cctrain,
                family=binomial)   # 'binomial' is used for logistic regression
# Examine the model details and the coefficients
summary(logModel)
# The summary() function returns the estimate, standard errors, z-score, and  
# p-values on each of the coefficients.
# Check which variable is the most influential in the model
varImp(logModel)
# Here we can see that 'international_plan' is the most important predictor,
# followed by 'number_customer_service_calls'.

# Generate predictions on the test subset
logProb <- predict(logModel, cctest,
                   type="response")   # function will return probabilities
head(logProb)   # take a look at the first 6 probabilities
# Next, set a probability threshold of 0.5 to make predictions. Since 'no' is  
# the event of interest, if the probability of 'churn = no' <0.5, the 
# prediction should be set to 'yes', else, set to 'no'.
logPred <- ifelse(logProb<0.5, "yes","no")
# Convert it to a factor, with Level 1 = 'yes' & 2 = 'no'
logPredf <- factor(logPred, levels=c("yes","no"))
head(logPredf)   # check the first 6 entries

# To evaluate model performance, a common method fo describing the performance
# of a classification model is the confusion matrix.  
logcm <- confusionMatrix(data=logPredf,
                reference=cctest$churn)
# The confusionMatrix() command gives a confusion matrix and overall statistics
# including Accuracy, Kappa statistics, Sensitivity, and Specificity, etc.
logcm   # take a look at the results
# Specificity is high but sensitivity is low because many 'yes's' were 
# predicted as 'no's' according to the confusion matrix.

# Next, evaluate class probabilities using the ROC curve
logROC <- roc(response=cctest$churn,
              predictor=logProb)  
plot(logROC,   # plotting the curve
     legacy.axes=TRUE, 
     main="ROC Curve for Logstic Regression")
logAUC <- auc(logROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
logMetrics <- data.frame(Accuracy=logcm$overall["Accuracy"], 
                        Kappa=logcm$overall["Kappa"],
                        Sensitivity=logcm$byClass["Sensitivity"], 
                        Specificity= logcm$byClass["Specificity"],
                        ROC_AUC = logAUC, 
                        row.names="Logistic Regression")
logMetrics   # Take a look at all metrics

# ------------------ LINEAR DISCRIMINANT ANALYSIS (LDA) ------------------
# The MASS package is commonly used to create LDA models using the lda() 
# function.

library(MASS) 

# Generate the LDA model
ldaModel <- lda(churn ~ ., 
                data=cctrain)
# Examine the model details and the coefficients
ldaModel

# Make predictions on the test subset
ldaProb <- predict(ldaModel, cctest)
str(ldaProb)   # check out the structure of the predictions
# There are 3 elements in ldaProb: class, posterior & x
# The posterior probability has two parts: 'yes' and 'no'
ldaProb$posterior[1:6,]   # see the first 6 rows of posterior probabilities
# Now, set a probability threshold of 0.5 to make predictions. if posterior 
# probability for 'no' <0.5, the prediction would be set to 'yes', else set
# to "no".
ldaPred <- ifelse(ldaProb$posterior[,2]<0.5, "yes","no")
# Next, convert it to a factor, with Level 1 = 'yes' & 2 = 'no'
ldaPredf <- factor(ldaPred, levels=c("yes","no"))    
head(ldaPredf)   # check out the first 6 entries

# Evaluate the performance of this classification model with a confusion matrix 
ldacm <- confusionMatrix(data=ldaPredf,
                         reference=cctest$churn)
ldacm   # see the results
# Same as before, the specificity is high but sensitivity is low because many 
# 'yes's' were predicted as 'no's'.

# Next, evaluate class probabilities using the ROC curve
ldaROC <- roc(response=cctest$churn,
              predictor=ldaProb$posterior[,2])  
plot(ldaROC,   # plotting the curve
     legacy.axes=TRUE, 
     main="ROC Curve for Linear Discriminant Analysis")
ldaAUC <- auc(ldaROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
ldaMetrics <- data.frame(Accuracy=ldacm$overall["Accuracy"], 
                         Kappa=ldacm$overall["Kappa"],
                         Sensitivity=ldacm$byClass["Sensitivity"], 
                         Specificity= ldacm$byClass["Specificity"],
                         ROC_AUC = ldaAUC, 
                         row.names="Linear Discriminant Analysis")
ldaMetrics   # Take a look at all metrics

# -------------------- PENALIZED LOGISTIC REGRESSION --------------------
# The glmnet() function in the glmnet package can be used for computing
# penalized logistic regression.

library(glmnet) 
# First define the x and y inputs
x <- model.matrix(churn ~ ., data=cctrain)[,-18]
y <- cctrain$churn
# Find the best lambda value using cross-validation
plogcv <- cv.glmnet(x,y, family="binomial")
plogLambda <- plogcv$lambda.min   # recording the best Lambda
# Build the model with training dataset
plogModel <- glmnet(x, y, 
                    family="binomial",
                    alpha=1,     # alpha=1 for LASSO regression
                    lambda=plogLambda)
plogModel   # take a look at the model details

# Make predictions on the test data
xtest <- model.matrix(churn ~ ., data=cctest)[,-18]   # define the x input
plogProb <- predict(plogModel, xtest,
                    type="response")   # function will return probabilities
head(plogProb)   # check out the first 6 entries
# Use a probability threshold of 0.5 to make predictions. If the probability of 
# "churn = no" <0.5, the prediction would be set to 'yes', else set to 'no'.
plogPred <- ifelse(plogProb<0.5, "yes","no")
# Next, convert it to a factor, with Level 1 = 'yes' & 2 = 'no'
plogPredf <- factor(plogPred, levels=c("yes","no")) 
head(plogPredf)   # check out the first 6 predictions after conversion

# To evaluate model performance, generate and examine the confusion matrix.  
plogcm <- confusionMatrix(data=plogPredf,
                         reference=cctest$churn)
plogcm   # take a look at the matrix
# Same as before, the specificity is high but sensitivity is low because many 
# 'yes's' were predicted as 'no's'.

# Next, evaluate class probabilities using the ROC curve
plogROC <- roc(response=cctest$churn,
              predictor=as.numeric(plogProb))  
plot(plogROC,   # plotting the curve
     legacy.axes=TRUE, 
     main="ROC Curve for LASSO Logistic Regression")
plogAUC <- auc(plogROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
plogMetrics <- data.frame(Accuracy=plogcm$overall["Accuracy"], 
                          Kappa=plogcm$overall["Kappa"],
                          Sensitivity=plogcm$byClass["Sensitivity"], 
                          Specificity= plogcm$byClass["Specificity"],
                          ROC_AUC = plogAUC, 
                          row.names="LASSO Logistic Regression")
plogMetrics   # Take a look at all metrics

# ------------------------- K-NEAREST NEIGHBORS -------------------------
# In order to obtain an optimized value for K, the KNN classification model 
# will be generated using the train() function in the caret package, with
# the 'method' parameter set to 'knn'.
# This time, ROC will be used to choose the best model.
# To allow each predictor to contribute equally to the distance calculation, 
# centering and scaling will be applied to all predictors.  
# Parameter tuning will be performed using a 10-fold cross-validation
# Class probabilities are needed to score models using the AUC.
set.seed(10)
# Generate the KNN model with training data
knnModel <- train(churn~., data=cctrain,
                  method="knn",
                  metric="ROC",   # optimized using ROC
                  preProc=c("center", "scale"),   # data pre-processing
                  trControl=trainControl(method="cv",  # 10-fold cv
                                         number=10, 
                                         classProbs=TRUE,
                                         summaryFunction=twoClassSummary))
knnModel   # check out the model details
# The optimal K for this model is 9.
plot(knnModel, 
     main="Cross Validation for K Optimization")   # confirm with a plot
# Check which variable is the most influential in the model
varImp(knnModel)
# This shows that the most important predictors are 'total_day_charge' and
# 'total_day_minutes'.

# Make predictions on the test subset
knnProb <- predict(knnModel, cctest, 
                   type="prob")   # function will return probabilities
head(knnProb)   # see the first 6 rows
# The result is a data frame with 2 columns 'yes' and 'no'
# Next, set a probability threshold to 0.5 and make predictions. If the 
# probability for 'no' <0.5, the prediction would be set to 'yes', else set
# to 'no'.
knnPred <- ifelse(knnProb$no<0.5, "yes","no")
# Convert to a factor with Level 1 = 'yes' & 2 = 'no'
knnPredf <- factor(knnPred, levels=c("yes","no"))  
head(knnPredf)

# To evaluate model performance, first, create a confusion matrix.
knncm <- confusionMatrix(data=knnPredf,
                          reference=cctest$churn)
knncm   # take a look at the results
# KNN is no different, the specificity is high but sensitivity is low because  
# many 'yes's' were predicted as 'no's'.

# Next, evaluate class probabilities using the ROC curve
knnROC <- roc(response=cctest$churn,
               predictor=knnProb$no)  
plot(knnROC,   # plotting the curve
     legacy.axes=TRUE, 
     main="ROC Curve for K-Nearest Neighbors")
knnAUC <- auc(knnROC)   # compute area under curve

# Finally, collect all performance metrics into a single table
knnMetrics <- data.frame(Accuracy=knncm$overall["Accuracy"], 
                         Kappa=knncm$overall["Kappa"],
                         Sensitivity=knncm$byClass["Sensitivity"], 
                         Specificity= knncm$byClass["Specificity"],
                         ROC_AUC = knnAUC, 
                         row.names="K-Nearest Neighbors")
knnMetrics   # Take a look at all metrics


# ------------------------------ RESULTS ------------------------------
# Consolidating the performance metrics for all classification models, we get:
rbind(logMetrics,
      ldaMetrics,
      plogMetrics,
      knnMetrics)
# Next, combine all ROC curves on the same plot
plot(logROC, col = "red", main = "Comparison of ROC Curves")   # base plot
plot(ldaROC, add=TRUE, col = "blue", lty = 2)   # add lines
plot(plogROC, add=TRUE, col = "green")
plot(knnROC, add=TRUE, col = "orange", lty = 2)
legend("bottomright",    # add a legend to the bottom-right corner
       legend=c("Logistic Regression", "LDA", "LASSO Logistic","KNN"), 
       col=c("red","blue","green","orange"),
       lty=1:2,
       cex=0.8,
       text.font=0.5)
# From this summary, we can see that the KNN model has the best performance.  
# Sensitivity (i.e. model correctly predicting 'yes' when it is 'yes') is low
# for all models, likely as a result of imbalanced classes in the dataset.
# Recall that 'no' is the majority class while 'yes' is the minority class.
# Classification models tend to have lower performance in predicting the 
# minority class when the training dataset is imbalanced.



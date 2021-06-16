knitr::opts_chunk$set(echo = TRUE)
library(tidyverse)
library(caret)
# 1. Import the data
setwd("~/Desktop/R")
data <- read.csv("insurance_claims.csv")
data <- data[-40] # The last column is not a variable
dataY <- factor(data$fraud_reported, levels = c("Y","N"))
dataX <- subset(data, select = -fraud_reported)
data
# Auto Information (3)
subset(dataX, select = c(auto_make,auto_model,auto_year))
# Customer Information (8)
subset(dataX, select = c(age,insured_education_level,insured_hobbies,
                         insured_occupation,insured_relationship,insured_sex,
                         insured_zip,months_as_customer))
# Incident Information (16)
subset(dataX, select = c(authorities_contacted,	bodily_injuries,	capital.gains,
                         capital.loss,	collision_type,	incident_city,	incident_date,
                         incident_hour_of_the_day,	incident_location,
                         incident_severity,incident_state,	incident_type,
                         number_of_vehicles_involved,police_report_available,
                         property_damage,	witnesses))
# Policy Information (10)
subset(dataX, select = c(policy_annual_premium,policy_bind_date,policy_csl,
                         policy_deductable,policy_state,property_claim,
                         total_claim_amount,umbrella_limit,vehicle_claim,injury_claim))
# 2. Selection of Predictors

### 2.1. Generate new variables
str(dataX$incident_date)
str(dataX$policy_bind_date)
# display predictors `policy_bind_date` and `incident_date`
dataX$days_after_bind <- as.Date(dataX$incident_date, format="%Y-%m-%d")-
  as.Date(dataX$policy_bind_date, format="%Y-%m-%d")
dataX$days_after_bind <- as.integer(dataX$days_after_bind)
# Display the result
subset(dataX, select = c(incident_date, policy_bind_date, days_after_bind))

dataX$incident_year = substr(dataX$incident_date, start = 1, stop = 4) 
dataX$years_after_auto_year = as.integer(dataX$incident_year) - dataX$auto_year
subset(dataX, select = c(auto_year, incident_date, incident_year, years_after_auto_year))

dataX$policy_bind_year <- as.integer(substr(dataX$policy_bind_date, start = 1, stop = 4))
subset(dataX, select = c(policy_bind_date, policy_bind_year))

### 2.2. Identify useless variables
# Delete predictors `policy_bind_date` and `incident_date`
dataX <- subset(dataX, select = -c(policy_bind_date, incident_date, incident_year))
str(dataX$insured_zip)
dataX$insured_zip <- as.factor(dataX$insured_zip)
str(dataX$insured_zip)
dataX <- subset(dataX,select = -c(insured_zip))

head(dataX$incident_location)
str(dataX$incident_location) # 1000 levels

# Remove the first 6 digits of the incident_location and save to data$incident_location2
dataX$incident_location2 = substr(dataX$incident_location, start = 6, stop = 100)
str(dataX$incident_location2) # still 1000 levels

# remove both incident_location and incident_location2
dataX <- subset(dataX, select = -c(incident_location,incident_location2))

str(dataX$policy_number)
dataX$policy_number <- as.factor(dataX$policy_number)
str(dataX$policy_number)
dataX <- subset(dataX, select = -c(policy_number))

# 3. Pre-process
### 3.1. Missing values
sum(is.na(dataX))
# missing values are stored as "?"
library(tidyverse)
na <- dataX %>% 
  summarise_all(funs(sum(. == "?"))) %>%
  gather(key = Attributes, value = Count.Missing.Values) %>%
  arrange(desc(Count.Missing.Values))
na

# Display the information of variables that have missing values
str(subset(dataX, select = c(property_damage, police_report_available,collision_type )))
### 3.2. Predictors with near 0 variance
nearZeroVar(dataX)
### 3.3. Split the data
set.seed(111)
training = createDataPartition(dataY, p = .7)[[1]]
trainX<-dataX[training,]
testX<-dataX[-training,]
trainY <- dataY[training]
testY <- dataY[-training]
dim(trainX)
### 3.4. Numerical data feature engineering
#  Pre-process the data. Apply Boxcox, center, and scale transformations
N_PP <- preProcess(trainX, c("BoxCox", "center", "scale"))
trainX <- predict(N_PP, trainX)
testX <- predict(N_PP, testX)
### 3.5. Categorical data feature engineering
dmy <- dummyVars(" ~ .", data = trainX, fullRank = T)
trainX_dmy <- data.frame(predict(dmy, newdata = trainX))
testX_dmy <- data.frame(predict(dmy, newdata = testX))
dim(trainX_dmy)
table(trainY)

set.seed(100)
trainXY_dmy <- cbind(trainY, trainX_dmy)
upSampledTrainXY_dmy <- upSample(x = trainXY_dmy[,-1],
                                 y = trainXY_dmy$trainY,
                                 ## keep the class variable name the same:
                                 yname = "trainY")

# Result:
upSampledTrainX_dmy <-subset(upSampledTrainXY_dmy, select = -c(trainY))
upSampledTrainY <- upSampledTrainXY_dmy$trainY
table(upSampledTrainY)

dim(dataX)
dim(trainX)
dim(trainX_dmy)

ctrl <- trainControl(method = "cv", 
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     savePredictions = TRUE,
                     verboseIter = TRUE)

### 4.1.1. Baseline Model
# Fit mod
library(sparseLDA)
library(kernlab)
set.seed(201)
simgaRangeFull <- sigest(as.matrix(trainX_dmy))
svmRGridFull <- expand.grid(sigma = as.vector(sigmaRangeFull)[1],
                            C = 2^(-1:4))
svmRGridFull
set.seed(100)
SVMmodel <- train(x = trainX_dmy, 
                  y = trainY,
                  method = "svmRadial",
                  preProc=c("center", "scale"),
                  metric = "ROC",
                  tuneGrid = svmRGridFull,
                  trControl = ctrl)
SVMmodel
# Variable importance
SVMImp <- varImp(SVMmodel, scale = FALSE)
plot(SVMImp, top = 10, scales = list(y = list(cex = .95)), main = "Top 10 Important Variables for svmRadial Model (Original Dataset)")

# Plot of Model Tuning
plot(SVMmodel, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")

# Predict testX_dmy
SVM_evalResults <- predict(SVMmodel,newdata = testX_dmy,type = "prob") # probability for class Y and N
SVM_evalResults$Prediction <- predict(SVMmodel,newdata = testX_dmy) # Prediction result
SVM_evalResults$True <- testY # True class
SVM_evalResults

# Validation on test set
confusionMatrix(SVM_evalResults$Prediction, SVM_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
SVMRoc_test <- roc(response = SVM_evalResults$True,
                     predictor = SVM_evalResults$Y,
                     levels =c("N","Y"),
                     direction = "<")
plot(SVMRoc_test, type = "s", legacy.axes = TRUE, main = "ROC of SVMmodel on test set")

# Get AUC value of test set
SVMRoc_test$auc

### 4.1.2. Model with Alternate Cutoff Method
# To get new cut off, we should create the ROC curve on training set first
# Create the list for Plotting ROC of the training set
library(pROC)
SVMRoc_training <- roc(response = SVMmodel$pred$obs,
                         predictor = SVMmodel$pred$Y,
                         levels =c("N","Y"),
                         direction = "<")

# Get the new cutoff
SVMThreshN <- coords(SVMRoc_training, x = "best", best.method = "closest.topleft") 
SVMThreshN

# Predict testX_dmy with new cutoff
SVM_evalResults$Prediction_NewCut <- factor(ifelse(SVM_evalResults$Y > SVMThreshN$threshold, 
                                                     "Y", "N"),
                                              levels = c("Y","N")) # Using new cutoff to predict test set
SVM_evalResults

# Validation on test set
confusionMatrix(SVM_evalResults$Prediction_NewCut, SVM_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
SVMRoc_test$auc
# Get AUC value of training set
print("training set")
SVMRoc_training$auc

# Plot ROC for training set
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
SVMThreshO <- coords(SVMRoc_training, x = 0.5, input = "threshold", transpose = FALSE) 
SVMThreshO
#### Get The specificity and sensitivity for the alternate threshold of 0.027 on training set
SVMThreshN
#### Visualize the original threshold and new threshold on training set
plot(SVMRoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of SVM model on training set")
points(SVMThreshN$specificity,
       SVMThreshN$sensitivity,pch=19,col="red")
points(SVMThreshO$specificity,
       SVMThreshO$sensitivity,pch=19,col="blue")

# Plot ROC for test set
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
SVMThreshO_test <- coords(SVMRoc_test, x = 0.5, input = "threshold", transpose = FALSE) 
SVMThreshO_test
#### Get The specificity and sensitivity for the alternate threshold of 0.027 on test set
SVMThreshN_test <- coords(SVMRoc_test, x = SVMThreshN$threshold, input = "threshold", transpose = FALSE) 
SVMThreshN_test
#### Visualize the original threshold and new threshold on test set
plot(SVMRoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of SVM model on test set")
points(SVMThreshN_test$specificity,
       SVMThreshN_test$sensitivity,pch=19,col="red")
points(SVMThreshO_test$specificity,
       SVMThreshO_test$sensitivity,pch=19,col="blue")

### 4.2.1. Baseline Model

# Fit model
library(sparseLDA)

set.seed(201)
simgaRangeFullup <- sigest(as.matrix(upSampledTrainX_dmy))
svmRGridFullup <- expand.grid(sigma = as.vector(simgaRangeFullup)[1],
                            C = 2^(-1:4))
svmRGridFullup

set.seed(100)
SVMmodel_up <- train(x = upSampledTrainX_dmy, 
                     y = upSampledTrainY,
                     method = "svmRadial",
                     preProc=c("center", "scale"),
                     metric = "ROC",
                     tuneGrid = svmRGridFullup,
                     trControl = ctrl)
SVMmodel_up

# Variable importance
SVMImp_up <- varImp(SVMmodel_up, scale = FALSE)
plot(SVMmodel_up, top = 25, scales = list(y = list(cex = .95)), main = "sparseLDA (UpSampled Dataset)")

# Plot of Model Tuning
plot(SVMmodel_up, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")

# Predict testX_dmy
SVM_up_evalResults <- predict(SVMmodel_up,newdata = testX_dmy,type = "prob") # probability for class Y and N
SVM_up_evalResults$Prediction <- predict(SVMmodel_up,newdata = testX_dmy) # Prediction result
SVM_up_evalResults$True <- testY # True class
SVM_up_evalResults

# Validation on test set
confusionMatrix(SVM_evalResults$Prediction, SVM_up_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
SVMRoc_up_test <- roc(response = SVM_up_evalResults$True,
                        predictor = SVM_up_evalResults$Y,
                        levels =c("N","Y"),
                        direction = "<")
plot(SVMRoc_up_test, type = "s", legacy.axes = TRUE, main = "ROC of SVM on test set")

# Get AUC value of test set
SVMRoc_up_test$auc

### 4.2.2. Model with Alternate Cutoff Method
# To get new cut off, we should create the ROC curve on training set first
# Create the list for Plotting ROC of the training set
library(pROC)
SVMRoc_up_training <- roc(response = SVMmodel_up$pred$obs,
                            predictor = SVMmodel_up$pred$Y,
                            levels =c("N","Y"),
                            direction = "<")

# Get the new cutoff
SVMThreshN_up <- coords(SVMRoc_up_training, x = "best", best.method = "closest.topleft") 
SVMThreshN_up

# Predict testX_dmy with new cutoff
SVM_up_evalResults$Prediction_NewCut <- factor(ifelse(SVM_up_evalResults$Y > SVMThreshN_up$threshold, "Y", "N"),
                                                 levels = c("Y","N")) # Using new cutoff to predict test set
SVM_up_evalResults

# Validation on test set
confusionMatrix(SVM_up_evalResults$Prediction_NewCut, SVM_up_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
SVMRoc_up_test$auc
# Get AUC value of training set
print("training set")
SVMRoc_up_training$auc

# Plot ROC for training set
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
SVMThreshO_up <- coords(SVMRoc_up_training, x = 0.5, input = "threshold", transpose = FALSE) 
SVMThreshO_up
#### Get The specificity and sensitivity for the alternate threshold of 0.2398 on training set
SVMThreshN_up
#### Visualize the original threshold and new threshold on training set
plot(SVMRoc_up_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of SVM model on training set")
points(SVMThreshN_up$specificity,
       SVMThreshN_up$sensitivity,pch=19,col="red")
points(SVMThreshO_up$specificity,
       SVMThreshO_up$sensitivity,pch=19,col="blue")

# Plot ROC for test set
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
SVMThreshO_up_test <- coords(SVMRoc_up_test, x = 0.5, input = "threshold", transpose = FALSE) 
SVMThreshO_up_test
#### Get The specificity and sensitivity for the alternate threshold of 0.2398 on test set
SVMThreshN_up_test <- coords(SVMRoc_up_test, x = SVMThreshN_up$threshold, input = "threshold", transpose = FALSE) 
SVMThreshN_up_test
#### Visualize the original threshold and new threshold on test set
plot(SVMRoc_up_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on test set")
points(SVMThreshN_up_test$specificity,
       SVMThreshN_up_test$sensitivity,pch=19,col="red")
points(SVMThreshO_up_test$specificity,
       SVMThreshO_up_test$sensitivity,pch=19,col="blue")

## 4.3. Summary
plot(SVMmodel, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")
plot(SVMmodel_up, scales = list(x = list(log = 10)), add = TRUE, ylab = "ROC AUC (Hold-Out Data)")

# Original data
plot(SVMRoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of SVM model on training set")
points(SVMThreshN$specificity,
       SVMThreshN$sensitivity,pch=17,col="red")
points(SVMThreshO$specificity,
       SVMThreshO$sensitivity,pch=17,col="blue")
# Upsampled data
plot(SVMRoc_up_training, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of SVM model on training set")
points(SVMThreshN_up$specificity,
       SVMThreshN_up$sensitivity,pch=19,col="red")
points(SVMThreshO_up$specificity,
       SVMThreshO_up$sensitivity,pch=19,col="blue")
text(0.05,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
text(0.05,0.13, "Method: closest.topleft", cex=0.8)

#### Visualize the original threshold and new threshold on test set
# Original Variables
plot(SVMRoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of SVM model on test set")
points(SVMThreshN_test$specificity,
       SVMThreshN_test$sensitivity,pch=17,col="red")
points(SVMThreshO_test$specificity,
       SVMThreshO_test$sensitivity,pch=17,col="blue")
# Up-sampling dataset
plot(SVMRoc_up_test, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of SVM model on test set")
points(SVMThreshN_up_test$specificity,
       SVMThreshN_up_test$sensitivity,pch=19,col="red")
points(SVMThreshO_up_test$specificity,
       SVMThreshO_up_test$sensitivity,pch=19,col="blue")
text(0.05,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
text(0.05,0.13, "Method: closest.topleft", cex=0.8)




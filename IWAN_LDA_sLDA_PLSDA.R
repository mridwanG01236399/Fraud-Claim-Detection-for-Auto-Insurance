library(tidyverse)
library(caret)

# 1. Import the data
################################################################################################
# Read the data
################################################################################################

data <- read.csv("D:/George Mason University/4. Fall 2020/Applied Predictive Analytics (OR 568)/Project Guideline/insurance_claims.csv")
data <- data[-40] # The last column is not a variable
dataY <- factor(data$fraud_reported, levels = c("Y","N"))
dataX <- subset(data, select = -fraud_reported)
data

################################################################################################
# Data Introduction
################################################################################################

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

################################################################################################

# 2. Selection of Predictors
################################################################################################
### 2.1. Generate new variables
################################################################################################

str(dataX$incident_date)
str(dataX$policy_bind_date)

# display predictors `policy_bind_date` and `incident_date`
dataX$days_after_bind <- as.Date(dataX$incident_date, format="%Y-%m-%d")-as.Date(dataX$policy_bind_date, format="%Y-%m-%d")

dataX$days_after_bind <- as.integer(dataX$days_after_bind)

# Display the result
subset(dataX, select = c(incident_date, policy_bind_date, days_after_bind))

################################################################################################

# generate a new predictor `years_after_auto_year` to **describe the length of time from the production year of an auto to an incident year.**
dataX$incident_year = substr(dataX$incident_date, start = 1, stop = 4) 
dataX$years_after_auto_year = as.integer(dataX$incident_year) - dataX$auto_year
subset(dataX, select = c(auto_year, incident_date, incident_year, years_after_auto_year))

################################################################################################

# keep only the year information which is save as `policy_bind_year` to have a overall description.
dataX$policy_bind_year <- as.integer(substr(dataX$policy_bind_date, start = 1, stop = 4))
subset(dataX, select = c(policy_bind_date, policy_bind_year))

################################################################################################
### 2.2. Identify useless variables
################################################################################################

# Delete predictors `policy_bind_date` and `incident_date`
dataX <- subset(dataX, select = -c(policy_bind_date, incident_date, incident_year))

################################################################################################

# Convert the data type of `insured_zip` from integer to Factor because zipcode is nominal data
str(dataX$insured_zip)
dataX$insured_zip <- as.factor(dataX$insured_zip)

# there are 995 different levels out of 1000 observations, which is too messy to make prediction. Therefore, we delete this predictor.
str(dataX$insured_zip)
dataX <- subset(dataX,select = -c(insured_zip))

################################################################################################

head(dataX$incident_location)
str(dataX$incident_location) # 1000 levels

# Remove the first 6 digits of the incident_location and save to data$incident_location2
dataX$incident_location2 = substr(dataX$incident_location, start = 6, stop = 100)
str(dataX$incident_location2) # still 1000 levels

# remove both incident_location and incident_location2
dataX <- subset(dataX, select = -c(incident_location,incident_location2))

################################################################################################

str(dataX$policy_number)
dataX$policy_number <- as.factor(dataX$policy_number)

str(dataX$policy_number)
dataX <- subset(dataX, select = -c(policy_number))

################################################################################################

str(dataX$policy_number)
dataX$policy_number <- as.factor(dataX$policy_number)

str(dataX$policy_number)
dataX <- subset(dataX, select = -c(policy_number))

################################################################################################

# 3. Pre-process
################################################################################################
### 3.1. Missing values
################################################################################################

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

################################################################################################
### 3.2. Predictors with near 0 variance
################################################################################################

nearZeroVar(dataX)

################################################################################################
### 3.3. Split the data
################################################################################################

set.seed(111)
training = createDataPartition(dataY, p = .7)[[1]]
trainX<-dataX[training,]
testX<-dataX[-training,]
trainY <- dataY[training]
testY <- dataY[-training]
dim(trainX)
dim(testX)

################################################################################################
### 3.4. Numerical data feature engineering
################################################################################################

#  Pre-process the data. Apply Boxcox, center, and scale transformations
N_PP <- preProcess(trainX, c("BoxCox", "center", "scale"))
trainX <- predict(N_PP, trainX)
testX <- predict(N_PP, testX)

################################################################################################
### 3.5. Categorical data feature engineering
################################################################################################

dmy <- dummyVars(" ~ .", data = trainX, fullRank = T)
trainX_dmy <- data.frame(predict(dmy, newdata = trainX))
testX_dmy <- data.frame(predict(dmy, newdata = testX))
dim(trainX_dmy)
dim(testX_dmy)

################################################################################################
### 3.6. Up-Sampling Method
################################################################################################

# imbalance class
table(trainY)

# Up sampling of TrainX_dmy
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

################################################################################################
### 3.7. Review of datasets
################################################################################################

# The whole data X without feature engineering
str(dataX)
dim(dataX)

# The whole data Y
str(dataY)
dim(as.data.frame(dataY))

# Training X without dummy variable
str(trainX)
dim(trainX)

# Training Y
str(trainY)
dim(as.data.frame(trainY))

# Training X with dummy variable
str(trainX_dmy)
dim(trainX_dmy)

# Training X and Y with dummy variable
dim(trainXY_dmy)

# Up-Sampled training X and Y with dummy variable
dim(upSampledTrainXY_dmy)

# Up-Sampled training X with dummy variable
dim(upSampledTrainX_dmy)

# Up-Sampled training Y
dim(as.data.frame(upSampledTrainY))

# Test X without dummy variable
dim(testX)

# Test X with dummy variable
dim(testX_dmy)

# Test Y
dim(as.data.frame(testY))

################################################################################################

ctrl <- trainControl(method = "cv", 
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary,
                     savePredictions = TRUE,
                     verboseIter = TRUE)

################################################################################################

# 4. Sparse LDA
################################################################################################
## 4.1. Original Dataset
################################################################################################

# Training X with dummy variable
trainX_dmy

# Training Y
trainY

# Test X with dummy variable
testX_dmy

# Test Y
testY

################################################################################################
### 4.1.1. Baseline Model
################################################################################################

# Fit model
library(sparseLDA)
set.seed(100)
spLDAFit <- train(x = trainX_dmy, 
                  y = trainY,
                  method = "sparseLDA",
                  tuneGrid = expand.grid(lambda = c(.1),
                                         NumVars = c(1, 5, 10, 15, 20, 50, 100, 250, 500, 1000)),
                  metric = "ROC",
                  trControl = ctrl)
spLDAFit

# Variable importance
spLDAImp <- varImp(spLDAFit, scale = FALSE)
plot(spLDAImp, top = 25, scales = list(y = list(cex = .95)), main = "sparseLDA")

# Plot of Model Tuning
plot(spLDAFit, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")

# Predict testX_dmy
spLDA_evalResults <- predict(spLDAFit,newdata = testX_dmy,type = "prob") # probability for class Y and N
spLDA_evalResults$Prediction <- predict(spLDAFit,newdata = testX_dmy) # Prediction result
spLDA_evalResults$True <- testY # True class
spLDA_evalResults

# Validation on test set
confusionMatrix(spLDA_evalResults$Prediction, spLDA_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
spLDARoc_test <- roc(response = spLDA_evalResults$True,
                     predictor = spLDA_evalResults$Y,
                     levels =c("N","Y"),
                     direction = "<")
plot(spLDARoc_test, type = "s", legacy.axes = TRUE, main = "ROC of sparseLDA on test set")

# Get AUC value of test set
spLDARoc_test$auc

################################################################################################
### 4.1.2. Model with Alternate Cutoff Method
################################################################################################

library(pROC)
spLDARoc_training <- roc(response = spLDAFit$pred$obs,
                         predictor = spLDAFit$pred$Y,
                         levels =c("N","Y"),
                         direction = "<")

# Get the new cutoff
spLDAThreshN <- coords(spLDARoc_training, x = "best", best.method = "closest.topleft") 
spLDAThreshN

# Predict testX_dmy with new cutoff
spLDA_evalResults$Prediction_NewCut <- factor(ifelse(spLDA_evalResults$Y > spLDAThreshN$threshold, 
                                                     "Y", "N"),
                                              levels = c("Y","N")) # Using new cutoff to predict test set
spLDA_evalResults

# Validation on test set
confusionMatrix(spLDA_evalResults$Prediction_NewCut, spLDA_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
spLDARoc_test$auc

# Get AUC value of training set
print("training set")
spLDARoc_training$auc

################################################################################################

# Plot ROC for training set
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
spLDAThreshO <- coords(spLDARoc_training, x = 0.5, input = "threshold", transpose = FALSE) 
spLDAThreshO

#### Get The specificity and sensitivity for the alternate threshold of 0.027 on training set
spLDAThreshN

#### Visualize the original threshold and new threshold on training set
plot(spLDARoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on training set")
points(spLDAThreshN$specificity,
       spLDAThreshN$sensitivity,pch=19,col="red")
points(spLDAThreshO$specificity,
       spLDAThreshO$sensitivity,pch=19,col="blue")

################################################################################################

# Plot ROC for test set
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
spLDAThreshO_test <- coords(spLDARoc_test, x = 0.5, input = "threshold", transpose = FALSE) 
spLDAThreshO_test

#### Get The specificity and sensitivity for the alternate threshold of 0.027 on test set
spLDAThreshN_test <- coords(spLDARoc_test, x = spLDAThreshN$threshold, input = "threshold", transpose = FALSE) 
spLDAThreshN_test

#### Visualize the original threshold and new threshold on test set
plot(spLDARoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on test set")
points(spLDAThreshN_test$specificity,
       spLDAThreshN_test$sensitivity,pch=19,col="red")
points(spLDAThreshO_test$specificity,
       spLDAThreshO_test$sensitivity,pch=19,col="blue")

################################################################################################
## 4.2. Upsampled Dataset
################################################################################################

# Up-Sampled training X with dummy variable
upSampledTrainX_dmy

# Up-Sampled training Y
upSampledTrainY

# Test X with dummy variable
testX_dmy

# Test Y
testY

################################################################################################
### 4.2.1. Baseline Model
################################################################################################

# Fit model
library(sparseLDA)
set.seed(100)
spLDAFit_up <- train(x = upSampledTrainX_dmy, 
                     y = upSampledTrainY,
                     method = "sparseLDA",
                     tuneGrid = expand.grid(lambda = c(.1),
                                            NumVars = c(1, 5, 10, 15, 20, 50, 100, 250, 500, 1000)),
                     metric = "ROC",
                     trControl = ctrl)
spLDAFit_up

# Variable importance
spLDAImp_up <- varImp(spLDAFit_up, scale = FALSE)
plot(spLDAImp_up, top = 25, scales = list(y = list(cex = .95)), main = "sparseLDA (UpSampled Dataset)")

# Plot of Model Tuning
plot(spLDAFit_up, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")

# Predict testX_dmy
spLDA_up_evalResults <- predict(spLDAFit_up,newdata = testX_dmy,type = "prob") # probability for class Y and N
spLDA_up_evalResults$Prediction <- predict(spLDAFit_up,newdata = testX_dmy) # Prediction result
spLDA_up_evalResults$True <- testY # True class
spLDA_up_evalResults

# Validation on test set
confusionMatrix(spLDA_up_evalResults$Prediction, spLDA_up_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
spLDARoc_up_test <- roc(response = spLDA_up_evalResults$True,
                        predictor = spLDA_up_evalResults$Y,
                        levels =c("N","Y"),
                        direction = "<")
plot(spLDARoc_up_test, type = "s", legacy.axes = TRUE, main = "ROC of sparseLDA on test set")

# Get AUC value of test set
spLDARoc_up_test$auc

################################################################################################
### 4.2.2. Model with Alternate Cutoff Method
################################################################################################
# To get new cut off, we should create the ROC curve on training set first
# Create the list for Plotting ROC of the training set
library(pROC)
spLDARoc_up_training <- roc(response = spLDAFit_up$pred$obs,
                            predictor = spLDAFit_up$pred$Y,
                            levels =c("N","Y"),
                            direction = "<")

# Get the new cutoff
spLDAThreshN_up <- coords(spLDARoc_up_training, x = "best", best.method = "closest.topleft") 
spLDAThreshN_up

# Predict testX_dmy with new cutoff
spLDA_up_evalResults$Prediction_NewCut <- factor(ifelse(spLDA_up_evalResults$Y > spLDAThreshN_up$threshold, "Y", "N"),
                                                 levels = c("Y","N")) # Using new cutoff to predict test set
spLDA_up_evalResults

# Validation on test set
confusionMatrix(spLDA_up_evalResults$Prediction_NewCut, spLDA_up_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
spLDARoc_up_test$auc

# Get AUC value of training set
print("training set")
spLDARoc_up_training$auc
################################################################################################
# Plot ROC for training set
################################################################################################
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
spLDAThreshO_up <- coords(spLDARoc_up_training, x = 0.5, input = "threshold", transpose = FALSE) 
spLDAThreshO_up

#### Get The specificity and sensitivity for the alternate threshold of 0.2398 on training set
spLDAThreshN_up

#### Visualize the original threshold and new threshold on training set
plot(spLDARoc_up_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on training set")
points(spLDAThreshN_up$specificity,
       spLDAThreshN_up$sensitivity,pch=19,col="red")
points(spLDAThreshO_up$specificity,
       spLDAThreshO_up$sensitivity,pch=19,col="blue")
################################################################################################
# Plot ROC for test set
################################################################################################
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
spLDAThreshO_up_test <- coords(spLDARoc_up_test, x = 0.5, input = "threshold", transpose = FALSE) 
spLDAThreshO_up_test

#### Get The specificity and sensitivity for the alternate threshold of 0.2398 on test set
spLDAThreshN_up_test <- coords(spLDARoc_up_test, x = spLDAThreshN_up$threshold, input = "threshold", transpose = FALSE) 
spLDAThreshN_up_test

#### Visualize the original threshold and new threshold on test set
plot(spLDARoc_up_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on test set")
points(spLDAThreshN_up_test$specificity,
       spLDAThreshN_up_test$sensitivity,pch=19,col="red")
points(spLDAThreshO_up_test$specificity,
       spLDAThreshO_up_test$sensitivity,pch=19,col="blue")
################################################################################################
################################################################################################
#### Visualize the original threshold and new threshold on training set
# Original data
plot(spLDARoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on training set")
points(spLDAThreshN$specificity,
       spLDAThreshN$sensitivity,pch=17,col="red")
points(spLDAThreshO$specificity,
       spLDAThreshO$sensitivity,pch=17,col="blue")
# Upsampled data
plot(spLDARoc_up_training, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of sparseLDA model on training set")
points(spLDAThreshN_up$specificity,
       spLDAThreshN_up$sensitivity,pch=19,col="red")
points(spLDAThreshO_up$specificity,
       spLDAThreshO_up$sensitivity,pch=19,col="blue")
text(-0.1,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
text(-0.1,0.13, "Method: closest.topleft", cex=0.8)
################################################################################################
################################################################################################
#### Visualize the original threshold and new threshold on test set
# Original Variables
plot(spLDARoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of sparseLDA model on test set")
points(spLDAThreshN_test$specificity,
       spLDAThreshN_test$sensitivity,pch=17,col="red")
points(spLDAThreshO_test$specificity,
       spLDAThreshO_test$sensitivity,pch=17,col="blue")
# Up-sampling dataset
plot(spLDARoc_up_test, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of sparseLDA model on test set")
points(spLDAThreshN_up_test$specificity,
       spLDAThreshN_up_test$sensitivity,pch=19,col="red")
points(spLDAThreshO_up_test$specificity,
       spLDAThreshO_up_test$sensitivity,pch=19,col="blue")
text(-0.1,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
################################################################################################
################################################################################################

# 5. LDA
################################################################################################
## 5.1. Original Dataset
################################################################################################

# Training X with dummy variable
trainX_dmy

# Training Y
trainY

# Test X with dummy variable
testX_dmy

# Test Y
testY

################################################################################################
### 5.1.1. Baseline Model
################################################################################################

# Fit the model
set.seed(100)
ldaFit <- train(x = trainX_dmy, 
                y = trainY,
                method = "lda",
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
ldaFit

# Variable importance
LDAImp <- varImp(ldaFit, scale = FALSE)
plot(LDAImp, top = 25, scales = list(y = list(cex = .95)), main = "LDA")

# Plot of Model Tuning
#plot(ldaFit, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")

# Predict testX_dmy
LDA_evalResults <- predict(ldaFit, newdata = testX_dmy, type = "prob") # probability for class Y and N
LDA_evalResults$Prediction <- predict(ldaFit, newdata = testX_dmy) # Prediction result
LDA_evalResults$True <- testY # True class
LDA_evalResults

# Validation on test set
confusionMatrix(LDA_evalResults$Prediction, LDA_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
LDARoc_test <- roc(response = LDA_evalResults$True,
                   predictor = LDA_evalResults$Y,
                   levels =c("N","Y"),
                   direction = "<")
plot(LDARoc_test, type = "s", legacy.axes = TRUE, main = "ROC of LDA on test set")

# Get AUC value of test set
LDARoc_test$auc

################################################################################################
### 5.1.2. Model with Alternate Cutoff Method
################################################################################################

library(pROC)
LDARoc_training <- roc(response = ldaFit$pred$obs,
                         predictor = ldaFit$pred$Y,
                         levels =c("N","Y"),
                         direction = "<")

# Get the new cutoff
LDAThreshN <- coords(LDARoc_training, x = "best", best.method = "closest.topleft") 
LDAThreshN

# Predict testX_dmy with new cutoff
LDA_evalResults$Prediction_NewCut <- factor(ifelse(LDA_evalResults$Y > LDAThreshN$threshold, 
                                                     "Y", "N"),
                                              levels = c("Y","N")) # Using new cutoff to predict test set
LDA_evalResults

# Validation on test set
confusionMatrix(LDA_evalResults$Prediction_NewCut, LDA_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
LDARoc_test$auc

# Get AUC value of training set
print("training set")
LDARoc_training$auc

################################################################################################

# Plot ROC for training set
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
LDAThreshO <- coords(LDARoc_training, x = 0.5, input = "threshold", transpose = FALSE) 
LDAThreshO

#### Get The specificity and sensitivity for the alternate threshold of 0.137 on training set
LDAThreshN

#### Visualize the original threshold and new threshold on training set
plot(LDARoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on training set")
points(LDAThreshN$specificity,
       LDAThreshN$sensitivity,pch=19,col="red")
points(LDAThreshO$specificity,
       LDAThreshO$sensitivity,pch=19,col="blue")

################################################################################################

# Plot ROC for test set
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
LDAThreshO_test <- coords(LDARoc_test, x = 0.5, input = "threshold", transpose = FALSE) 
LDAThreshO_test

#### Get The specificity and sensitivity for the alternate threshold of 0.137 on test set
LDAThreshN_test <- coords(LDARoc_test, x = LDAThreshN$threshold, input = "threshold", transpose = FALSE) 
LDAThreshN_test

#### Visualize the original threshold and new threshold on test set
plot(LDARoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on test set")
points(LDAThreshN_test$specificity,
       LDAThreshN_test$sensitivity,pch=19,col="red")
points(LDAThreshO_test$specificity,
       LDAThreshO_test$sensitivity,pch=19,col="blue")

################################################################################################
## 5.2. Upsampled Dataset
################################################################################################

# Up-Sampled training X with dummy variable
upSampledTrainX_dmy

# Up-Sampled training Y
upSampledTrainY

# Test X with dummy variable
testX_dmy

# Test Y
testY

################################################################################################
### 5.2.1. Baseline Model
################################################################################################
# Fit model
set.seed(100)
ldaFit_up <- train(x = upSampledTrainX_dmy,
                   y = upSampledTrainY,
                   method = "lda",
                   preProc = c("center","scale"),
                   metric = "ROC",
                   trControl = ctrl)
ldaFit_up

# Variable importance
LDAImp_up <- varImp(ldaFit_up, scale = FALSE)
plot(LDAImp_up, top = 25, scales = list(y = list(cex = .95)), main = "LDA (UpSampled Dataset)")

# Plot of Model Tuning
#plot(ldaFit_up, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)")

# Predict testX_dmy
LDA_up_evalResults <- predict(ldaFit_up, newdata = testX_dmy, type = "prob") # probability for class Y and N
LDA_up_evalResults$Prediction <- predict(ldaFit_up, newdata = testX_dmy) # Prediction result
LDA_up_evalResults$True <- testY # True class
LDA_up_evalResults

# Validation on test set
confusionMatrix(LDA_up_evalResults$Prediction, LDA_up_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
LDARoc_up_test <- roc(response = LDA_up_evalResults$True,
                        predictor = LDA_up_evalResults$Y,
                        levels =c("N","Y"),
                        direction = "<")
plot(LDARoc_up_test, type = "s", legacy.axes = TRUE, main = "ROC of sparseLDA on test set")

# Get AUC value of test set
LDARoc_up_test$auc

################################################################################################
### 5.2.2. Model with Alternate Cutoff Method
################################################################################################
# To get new cut off, we should create the ROC curve on training set first
# Create the list for Plotting ROC of the training set
library(pROC)
LDARoc_up_training <- roc(response = ldaFit_up$pred$obs,
                          predictor = ldaFit_up$pred$Y,
                          levels =c("N","Y"),
                          direction = "<")

# Get the new cutoff
LDAThreshN_up <- coords(LDARoc_up_training, x = "best", best.method = "closest.topleft") 
LDAThreshN_up

# Predict testX_dmy with new cutoff
LDA_up_evalResults$Prediction_NewCut <- factor(ifelse(LDA_up_evalResults$Y > LDAThreshN_up$threshold, "Y", "N"),
                                                 levels = c("Y","N")) # Using new cutoff to predict test set
LDA_up_evalResults

# Validation on test set
confusionMatrix(LDA_up_evalResults$Prediction_NewCut, LDA_up_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
LDARoc_up_test$auc

# Get AUC value of training set
print("training set")
LDARoc_up_training$auc
################################################################################################
# Plot ROC for training set
################################################################################################
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
LDAThreshO_up <- coords(LDARoc_up_training, x = 0.5, input = "threshold", transpose = FALSE) 
LDAThreshO_up

#### Get The specificity and sensitivity for the alternate threshold of 0.5132156 on training set
LDAThreshN_up

#### Visualize the original threshold and new threshold on training set
plot(LDARoc_up_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on training set")
points(LDAThreshN_up$specificity,
       LDAThreshN_up$sensitivity,pch=19,col="red")
points(LDAThreshO_up$specificity,
       LDAThreshO_up$sensitivity,pch=19,col="blue")
################################################################################################
# Plot ROC for test set
################################################################################################
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
LDAThreshO_up_test <- coords(LDARoc_up_test, x = 0.5, input = "threshold", transpose = FALSE) 
LDAThreshO_up_test

#### Get The specificity and sensitivity for the alternate threshold of 0.2398 on test set
LDAThreshN_up_test <- coords(LDARoc_up_test, x = LDAThreshN_up$threshold, input = "threshold", transpose = FALSE) 
LDAThreshN_up_test

#### Visualize the original threshold and new threshold on test set
plot(LDARoc_up_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on test set")
points(LDAThreshN_up_test$specificity,
       LDAThreshN_up_test$sensitivity,pch=19,col="red")
points(LDAThreshO_up_test$specificity,
       LDAThreshO_up_test$sensitivity,pch=19,col="blue")
################################################################################################
################################################################################################
#### Visualize the original threshold and new threshold on training set
# Original data
plot(LDARoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on training set")
points(LDAThreshN$specificity,
       LDAThreshN$sensitivity,pch=17,col="red")
points(LDAThreshO$specificity,
       LDAThreshO$sensitivity,pch=17,col="blue")
# Upsampled data
plot(LDARoc_up_training, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of LDA model on training set")
points(LDAThreshN_up$specificity,
       LDAThreshN_up$sensitivity,pch=19,col="red")
points(LDAThreshO_up$specificity,
       LDAThreshO_up$sensitivity,pch=19,col="blue")
text(-0.1,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
text(-0.1,0.13, "Method: closest.topleft", cex=0.8)
################################################################################################
################################################################################################
#### Visualize the original threshold and new threshold on test set
# Original Variables
plot(LDARoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on test set")
points(LDAThreshN_test$specificity,
       LDAThreshN_test$sensitivity,pch=17,col="red")
points(LDAThreshO_test$specificity,
       LDAThreshO_test$sensitivity,pch=17,col="blue")
# Up-sampling dataset
plot(LDARoc_up_test, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of LDA model on test set")
points(LDAThreshN_up_test$specificity,
       LDAThreshN_up_test$sensitivity,pch=19,col="red")
points(LDAThreshO_up_test$specificity,
       LDAThreshO_up_test$sensitivity,pch=19,col="blue")
text(-0.1,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
################################################################################################
################################################################################################

# 6. PLS DA
################################################################################################
## 6.1. Original Dataset
################################################################################################

# Training X with dummy variable
trainX_dmy

# Training Y
trainY

# Test X with dummy variable
testX_dmy

# Test Y
testY

################################################################################################
### 6.1.1. Baseline Model
################################################################################################

# Fit the model
library(pls)
set.seed(100)
plsFit <- train(x = trainX_dmy, 
                y = trainY,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:10),
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
plsFit

# Variable importance
plsImp <- varImp(plsFit, scale = FALSE)
plot(plsImp, top = 25, scales = list(y = list(cex = .95)), main = "PLS-DA")

# Plot of Model Tuning
plot(plsFit, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)", main = "PLS-DA Hyperparameter Tuning (Original Dataset)")

# Predict testX_dmy
pls_evalResults <- predict(plsFit, newdata = testX_dmy, type = "prob") # probability for class Y and N
pls_evalResults$Prediction <- predict(plsFit, newdata = testX_dmy) # Prediction result
pls_evalResults$True <- testY # True class
pls_evalResults

# Validation on test set
confusionMatrix(pls_evalResults$Prediction, pls_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
plsRoc_test <- roc(response = pls_evalResults$True,
                   predictor = pls_evalResults$Y,
                   levels =c("N","Y"),
                   direction = "<")
plot(plsRoc_test, type = "s", legacy.axes = TRUE, main = "ROC of PLS-DA on test set")

# Get AUC value of test set
plsRoc_test$auc

################################################################################################
### 6.1.2. Model with Alternate Cutoff Method
################################################################################################

library(pROC)
plsRoc_training <- roc(response = plsFit$pred$obs,
                       predictor = plsFit$pred$Y,
                       levels =c("N","Y"),
                       direction = "<")

# Get the new cutoff
plsThreshN <- coords(plsRoc_training, x = "best", best.method = "closest.topleft") 
plsThreshN

# Predict testX_dmy with new cutoff
pls_evalResults$Prediction_NewCut <- factor(ifelse(pls_evalResults$Y > plsThreshN$threshold, 
                                                   "Y", "N"),
                                            levels = c("Y","N")) # Using new cutoff to predict test set
pls_evalResults

# Validation on test set
confusionMatrix(pls_evalResults$Prediction_NewCut, pls_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
plsRoc_test$auc

# Get AUC value of training set
print("training set")
plsRoc_training$auc

################################################################################################

# Plot ROC for training set
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
plsThreshO <- coords(plsRoc_training, x = 0.5, input = "threshold", transpose = FALSE) 
plsThreshO

#### Get The specificity and sensitivity for the alternate threshold of 0.3989202 on training set
plsThreshN

#### Visualize the original threshold and new threshold on training set
plot(plsRoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of PLS-DA model on training set")
points(plsThreshN$specificity,
       plsThreshN$sensitivity,pch=19,col="red")
points(plsThreshO$specificity,
       plsThreshO$sensitivity,pch=19,col="blue")

################################################################################################

# Plot ROC for test set
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
plsThreshO_test <- coords(plsRoc_test, x = 0.5, input = "threshold", transpose = FALSE) 
plsThreshO_test

#### Get The specificity and sensitivity for the alternate threshold of 0.137 on test set
plsThreshN_test <- coords(plsRoc_test, x = plsThreshN$threshold, input = "threshold", transpose = FALSE) 
plsThreshN_test

#### Visualize the original threshold and new threshold on test set
plot(plsRoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of PLS-DA model on test set")
points(plsThreshN_test$specificity,
       plsThreshN_test$sensitivity,pch=19,col="red")
points(plsThreshO_test$specificity,
       plsThreshO_test$sensitivity,pch=19,col="blue")

################################################################################################
## 6.2. Upsampled Dataset
################################################################################################

# Up-Sampled training X with dummy variable
upSampledTrainX_dmy

# Up-Sampled training Y
upSampledTrainY

# Test X with dummy variable
testX_dmy

# Test Y
testY

################################################################################################
### 6.2.1. Baseline Model
################################################################################################
# Fit model
set.seed(100)
plsFit_up <- train(x = upSampledTrainX_dmy, 
                y = upSampledTrainY,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:10),
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
plsFit_up

# Variable importance
plsImp_up <- varImp(plsFit_up, scale = FALSE)
plot(plsImp_up, top = 25, scales = list(y = list(cex = .95)), main = "PLS-DA (UpSampled Dataset)")

# Plot of Model Tuning
plot(plsFit_up, scales = list(x = list(log = 10)), ylab = "ROC AUC (Hold-Out Data)", main = "PLS-DA Hyperparameter Tuning (Un-sampling Dataset)")

# Predict testX_dmy
pls_up_evalResults <- predict(plsFit_up, newdata = testX_dmy, type = "prob") # probability for class Y and N
pls_up_evalResults$Prediction <- predict(plsFit_up, newdata = testX_dmy) # Prediction result
pls_up_evalResults$True <- testY # True class
pls_up_evalResults

# Validation on test set
confusionMatrix(pls_up_evalResults$Prediction, pls_up_evalResults$True, positive = "Y")

# Plot ROC of test set
library(pROC)
plsRoc_up_test <- roc(response = pls_up_evalResults$True,
                      predictor = pls_up_evalResults$Y,
                      levels =c("N","Y"),
                      direction = "<")
plot(plsRoc_up_test, type = "s", legacy.axes = TRUE, main = "ROC of PLS-DA on test set")

# Get AUC value of test set
plsRoc_up_test$auc

################################################################################################
### 6.2.2. Model with Alternate Cutoff Method
################################################################################################
# To get new cut off, we should create the ROC curve on training set first
# Create the list for Plotting ROC of the training set
library(pROC)
plsRoc_up_training <- roc(response = plsFit_up$pred$obs,
                          predictor = plsFit_up$pred$Y,
                          levels =c("N","Y"),
                          direction = "<")

# Get the new cutoff
plsThreshN_up <- coords(plsRoc_up_training, x = "best", best.method = "closest.topleft") 
plsThreshN_up

# Predict testX_dmy with new cutoff
pls_up_evalResults$Prediction_NewCut <- factor(ifelse(pls_up_evalResults$Y > plsThreshN_up$threshold, "Y", "N"),
                                               levels = c("Y","N")) # Using new cutoff to predict test set
pls_up_evalResults

# Validation on test set
confusionMatrix(pls_up_evalResults$Prediction_NewCut, pls_up_evalResults$True, positive = "Y")

# Get AUC value of test set
print("test set")
plsRoc_up_test$auc

# Get AUC value of training set
print("training set")
plsRoc_up_training$auc
################################################################################################
# Plot ROC for training set
################################################################################################
#### Get the specificity and sensitivity for the original threshold of 0.5 on training set
plsThreshO_up <- coords(plsRoc_up_training, x = 0.5, input = "threshold", transpose = FALSE) 
plsThreshO_up

#### Get The specificity and sensitivity for the alternate threshold of 0.5036994 on training set
plsThreshN_up

#### Visualize the original threshold and new threshold on training set
plot(plsRoc_up_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of PLS-DA model on training set")
points(plsThreshN_up$specificity,
       plsThreshN_up$sensitivity,pch=19,col="red")
points(plsThreshO_up$specificity,
       plsThreshO_up$sensitivity,pch=19,col="blue")
################################################################################################
# Plot ROC for test set
################################################################################################
#### Get the specificity and sensitivity for the original threshold of 0.5 on test set
plsThreshO_up_test <- coords(plsRoc_up_test, x = 0.5, input = "threshold", transpose = FALSE) 
plsThreshO_up_test

#### Get The specificity and sensitivity for the alternate threshold of 0.2398 on test set
plsThreshN_up_test <- coords(plsRoc_up_test, x = plsThreshN_up$threshold, input = "threshold", transpose = FALSE) 
plsThreshN_up_test

#### Visualize the original threshold and new threshold on test set
plot(plsRoc_up_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of PLS-DA model on test set")
points(plsThreshN_up_test$specificity,
       plsThreshN_up_test$sensitivity,pch=19,col="red")
points(plsThreshO_up_test$specificity,
       plsThreshO_up_test$sensitivity,pch=19,col="blue")
################################################################################################
################################################################################################
#### Visualize the original threshold and new threshold on training set
# Original data
plot(LDARoc_training, type = "s", legacy.axes = TRUE, 
     main = "ROC of LDA model on training set")
points(plsThreshN$specificity,
       plsThreshN$sensitivity,pch=17,col="red")
points(plsThreshO$specificity,
       plsThreshO$sensitivity,pch=17,col="blue")
# Upsampled data
plot(plsRoc_up_training, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of PLS-DA model on training set")
points(plsThreshN_up$specificity,
       plsThreshN_up$sensitivity,pch=19,col="red")
points(plsThreshO_up$specificity,
       plsThreshO_up$sensitivity,pch=19,col="blue")
text(-0.1,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
text(-0.1,0.13, "Method: closest.topleft", cex=0.8)
################################################################################################
################################################################################################
#### Visualize the original threshold and new threshold on test set
# Original Variables
plot(plsRoc_test, type = "s", legacy.axes = TRUE, 
     main = "ROC of PLS-DA model on test set")
points(plsThreshN_test$specificity,
       plsThreshN_test$sensitivity,pch=17,col="red")
points(plsThreshO_test$specificity,
       plsThreshO_test$sensitivity,pch=17,col="blue")
# Up-sampling dataset
plot(plsRoc_up_test, type = "s", legacy.axes = TRUE, add = TRUE,col = rgb(.5, .5, .5, .5),
     main = "ROC of PLS-DA model on test set")
points(plsThreshN_up_test$specificity,
       plsThreshN_up_test$sensitivity,pch=19,col="red")
points(plsThreshO_up_test$specificity,
       plsThreshO_up_test$sensitivity,pch=19,col="blue")
text(-0.1,0.05, "Black Line: Original dataset\nGray Line: Up-sampling dataset", cex=0.8)
################################################################################################
################################################################################################

#setwd("~/Dropbox/STAT542/Project/3_loan/")
# setwd("~/Downloads/")

###################################################################################
###############################   Pre-Process   ###################################
###################################################################################

time_start = proc.time()

# Load data
# loan <- read.csv("loan.csv")
# Q1 <- read.csv("LoanStats_2016Q1.csv", skip = 1, header = TRUE)
# n_Q1 <- nrow(Q1)
# 
# # Split 75% data as train and 25% data as test
# n <- nrow(loan)
# set.seed(123)
# test.id <- sample(1:n, floor(n/4))
# train <- loan[-test.id,]
# test <- loan[test.id,]
# 
# # Combine Q1 with test data
# # Transform factor to numeric: revol_util, int_rate, id, member_id
# Q1$int_rate <- as.numeric(sub("%","",Q1$int_rate))
# Q1$revol_util <- as.numeric(sub("%","",Q1$revol_util))
# Q1$id <- as.numeric(Q1$id)
# Q1$member_id <- as.numeric(Q1$member_id)
# # Drop different variables
# drop_var = setdiff(colnames(Q1), colnames(test))
# Q1 = Q1[, !names(Q1) %in% drop_var, drop = F]
# test = rbind(test,Q1)
# #write.csv(train,'train.csv')
# #write.csv(train,'test.csv')
# 
# # Basic descriptive statistics
# #library(DescTools)
# #Desc(train$loan_status, plotit = T)

train <- read.csv("train.csv")
test <- read.csv("test.csv")
# train = train[,-1]
# test = test[,-1]
test$loan_status = rep("De", nrow(test))

# Pre-processing Part
drop_miss_train = function(data){
  n_data <- nrow(data)
  numNA <- colSums(apply(data, 2, is.na))
  miss_var <- numNA[which(numNA != 0)]
  # Drop variables whose missing values are more than 60%
  drop_var <- which(miss_var/n_data > 0.6) 
  drop_name <<- names(drop_var)
  data <- data[, ! names(data) %in% drop_name, drop = F]
  return(data)
}

drop_miss_test = function(data){
  data <- data[, ! names(data) %in% drop_name, drop = F]
  return(data)
}

pre_process = function(data){
  # Add Default/ Not Default label
  default_indicators <- c("Charged Off ",
                      "Default",
                      "Does not meet the credit policy. Status:Charged Off",
                      "Late (16-30 days)",
                      "Late (31-120 days)")
  # Assign label 0 to represent default 
  data$label <- ifelse(data$loan_status %in% default_indicators, 1, ifelse(data$loan_status=="", NA, 0))
  
  # Drop some varialbes that will not be used for prediction
  drop_list <- c('loan_status', 'emp_title', 'url', 'desc', 'title', 'zip_code', 'grade', 'pymnt_plan', 'verification_status_joint', 'policy_code')
  data <- data[, ! names(data) %in% drop_list, drop = F]
  
  # Date keeps year and transform to numeric: issue_d, last_pymnt_d, last_credit_pull_d, next_pymnt_d, earliest_cr_line
  date_var <- c('issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'next_pymnt_d', 'earliest_cr_line')
  for (i in 1:length(date_var)){
    data[,date_var[i]] <-substr(data[,date_var[i]],5,8)
    data[,date_var[i]] <- as.numeric(data[,date_var[i]])
  }
  
  # Missing values
  # Define data type of variables
  numNA_1 <- colSums(apply(data, 2, is.na))
  miss_var_1 <- numNA_1[which(numNA_1 != 0)]
  miss_type <- sapply(data[,names(which(numNA_1 != 0))], class)
  cbind(miss_var_1, miss_type)
  
  # Fill missing values for catogorical variables
  # No categorical variables have missing values
  factor_var <- names(miss_var_1)[which(miss_type == 'factor')]
  if (length(factor_var)>0){
    for (i in 1:length(factor_var)){
      miss_obs_1 <- is.na(data[,factor_var[i]])
      table <- table(data[, factor_var[i]])
      data[,factor_var[i]][miss_obs_1] <- names(table) [table == max(table)][1]
    }
  }
  # Fill missing values for numerical variables
  numeric_var <- names(miss_var_1)[which(miss_type != 'factor')]
  for (i in 1:length(numeric_var)){
    miss_obs <- is.na(data[,numeric_var[i]])
    data[,numeric_var[i]][miss_obs] <- median(data[, numeric_var[i]], na.rm = TRUE)[1]
  }
  
  # Some Factor transform to numeric: sub_grade, emp_length
  data$sub_grade <- as.numeric(data$sub_grade)
  emp_length <- data$emp_length
  library(memisc)
  emp_length <- cases(
    '0' = emp_length == 'n/a',
    '1' = emp_length == '< 1 year',
    '2' = emp_length == '1 year',
    '3' = emp_length == '2 years',
    '4' = emp_length == '3 years',
    '5' = emp_length == '4 years',
    '6' = emp_length == '5 years',
    '7' = emp_length == '6 years',
    '8' = emp_length == '7 years',
    '9' = emp_length == '8 years',
    '10' = emp_length == '9 years',
    '11' = emp_length == '10+ years')
  data$emp_length <- as.numeric(emp_length) -1
  
  # Deal with unusual values: pub_rec, initial_list_status
  # fw value for initial_list_status
  status <- as.character(data$initial_list_status)
  table = table(status)
  vote_term = names(table) [table == max(table)]
  level_term = c("f","w")
  status = sapply(status, function(value){
    ifelse(value %in% level_term, value, vote_term)})
  data$initial_list_status = as.factor(status)
  # Winsorization for pub_rec
  library(DescTools)
  data$pub_rec <- Winsorize(data$pub_rec, na.rm = TRUE)
  
  return(data)
}

# Catogorical variables with too many levels
combine_train <- function(data, n_level){
  data_type <- sapply(data, class)
  factor_list <<- which(data_type == "factor")
  
  library(plyr)
  for (i in factor_list){
    data[,i] <- as.character(data[,i])
  }
  
  refer_list <<- NULL
  refer_name <<- NULL
  origin_list <<- NULL
  origin_name <<- NULL
  list = as.matrix(factor_list)
  for (i in 1:nrow(list)){
    col = list[i,]
    sum <- summary(as.factor(data[,col]))
    if(length(sum)<=n_level){
      origin_list <<- c(origin_list, col)
      sum <- sort(sum, decreasing = TRUE)
      origin_name <<- rbind.fill(as.data.frame(origin_name), as.data.frame(t(names(sum))))
    }
  }
  for (i in 1:nrow(list)){
    col = list[i,]
    sum <- summary(as.factor(data[,col]))
    if (length(sum)>n_level){
      refer_list <<- c(refer_list, col)
      sum <- sort(sum, decreasing = TRUE)[1:n_level]
      refer_name <<- cbind(refer_name, names(sum))
      top_name <- names(sum)
      data[,col] = sapply(data[,col], function(value){
        aa = ifelse(value %in% top_name, value, 'other')})
    }
  }
  
  for (i in factor_list){
    data[,i] <- as.factor(data[,i])
  }  
  return(data)
}

combine_test <- function(data){
  origin_name <<- t(origin_name)
  for (i in factor_list){
    data[,i] <- as.character(data[,i])
  }
  
  index <- 1
  for (i in refer_list){
    data[,i] = sapply(as.character(data[,i]), function(value){
                      ifelse(value %in% refer_name[,index], value, 'other')})
    if ((index + 1) <= ncol(refer_name)){
      index <- index+1
    }
  }
  
  ind <- 1
  for (i in origin_list){
    data[,i] = sapply(as.character(data[,i]), function(value){
                      ifelse(value %in% origin_name[,ind], value, origin_name[1,ind])})
    if ((ind + 1) <= ncol(origin_name)){
      ind <- ind+1
    }
  }
  
  for (i in factor_list){
    data[,i] <- as.factor(data[,i])
  } 
  
  return (data)
}

# Change categorical variables to dummy variables
dummy_train <- function(data){
  label <- data$label
  data_dum <- model.matrix(label ~ . -1, data = data)
  data_dum <- cbind(data_dum, label)
  data_dum <- as.data.frame(data_dum)
  return(data_dum)
}
dummy_test <- function(data){
  data_dum <- model.matrix(~.-1, data = data)
  data_dum <- as.data.frame(data_dum)
  return(data_dum)
}

# Apply pre-process, combine, dummy function
train_drop = drop_miss_train(train)
test_drop = drop_miss_test(test)
Train <- pre_process(train_drop)
Test <- pre_process(test_drop)
TRAIN <- combine_train(Train, 5)
TEST <- combine_test(Test)
TOTAL <- rbind(TRAIN,TEST)
n_TRAIN <- nrow(TRAIN)
TRAIN = TOTAL[1:n_TRAIN,]
TEST = TOTAL[-(1:n_TRAIN),]

#Keep levels of factors the same in TRAIN and TEST
TOTAL_dum <- dummy_train(TOTAL)
TRAIN_dum <- TOTAL_dum[1:n_TRAIN,] 
TEST_dum <- TOTAL_dum[-(1:n_TRAIN),]

# Evaluation
# y = as.numeric(as.character(TEST$label))
# evaluation = function(y, pred){
#   N = length(pred)
#   LogLoss = -1/N * sum(y*log(pred)+(1-y)*log(1-pred))
#   return(LogLoss)
# }

# Weights
# y_t = as.numeric(as.character(TRAIN$label))
# weight_train = ifelse(y_t == 1, 20, 1)

proc.time() - time_start

###################################################################################
###############################   Modeling   ######################################
###################################################################################

# Logistic Regression
time_start = proc.time()
model_logit = glm(label~.-id-member_id, data = TRAIN, family = "binomial")
#library(klaR)
#model_bic = stepclass(as.matrix(TRAIN_dum[,!(colnames(TRAIN_dum) %in% c('label','id','member_id'))]), 
#                      as.factor(TRAIN_dum[,'label']), method = 'lda',fold = 3)
pred_logit = predict(model_logit, TEST, type = "response")
pred_logit = sapply(pred_logit, function(value){
  value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))
})
# score_logit = evaluation(y, pred_logit)
# score_logit
proc.time() - time_start

# Ridge
time_start = proc.time()
library(glmnet)
TRAIN_dum$label = as.factor(TRAIN_dum$label)
TEST_dum$label = as.factor(TEST_dum$label)
# Sample for cv
index = sample(1:nrow(TRAIN),floor(nrow(TRAIN)/2))
model_ridge = glmnet(as.matrix(TRAIN_dum[,!(colnames(TRAIN_dum) %in% c('label','id','member_id'))]), TRAIN_dum[,'label'], family = "binomial", alpha = 0)
cv_ridge = cv.glmnet(as.matrix(TRAIN_dum[index,!(colnames(TRAIN_dum) %in% c('label','id','member_id'))]), 
                     TRAIN_dum[index,'label'], family = "binomial", alpha = 0,
                     nfolds = 3, parallel = TRUE)
pred_ridge = predict(model_ridge, s=cv_ridge$lambda.1se, newx = as.matrix(TEST_dum[,!(colnames(TEST_dum) %in% c('label','id','member_id'))]),type = "response")
pred_ridge = sapply(pred_ridge, function(value){
                    value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))})
# score_ridge = evaluation(y, pred_ridge)
# score_ridge
proc.time() - time_start

# Lasso
time_start = proc.time()
model_lasso = glmnet(as.matrix(TRAIN_dum[,!(colnames(TRAIN_dum) %in% c('label','id','member_id'))]), TRAIN_dum[,'label'], family = "binomial", alpha = 1)
cv_lasso = cv.glmnet(as.matrix(TRAIN_dum[index,!(colnames(TRAIN_dum) %in% c('label','id','member_id'))]), 
                     TRAIN_dum[index,'label'], family = "binomial", alpha = 1,
                     nfolds = 3, parallel = TRUE)
pred_lasso = predict(model_lasso, s=cv_lasso$lambda.1se, newx = as.matrix(TEST_dum[,!(colnames(TEST_dum) %in% c('label','id','member_id'))]),type = "response")
pred_lasso = sapply(pred_lasso, function(value){
  value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))
})
# score_lasso = evaluation(y, pred_lasso)
# score_lasso
proc.time() - time_start

# Random Forest
time_start = proc.time()
library(randomForest)
TRAIN$label = as.factor(TRAIN$label)
TEST$label =as.factor(TEST$label)
model_rf = randomForest(label~.-id-member_id, data = TRAIN, ntree= 50, importance=TRUE)
pred_rf = predict(model_rf, TEST[,! (colnames(TEST) %in% 'label')], type = "prob")
pred_rf = sapply(pred_rf[,2], function(value){
                 value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))})
# score_rf = evaluation(y, pred_rf)
# score_rf
proc.time() - time_start

#GBM
time_start = proc.time()
library(gbm)
TRAIN$label = as.numeric(as.character(TRAIN$label))

model_gbm = gbm(label~.-id-member_id, data = TRAIN, n.trees = 100, shrinkage = 1, 
                bag.fraction = 0.5, distribution = "adaboost")
pred_gbm = predict(model_gbm, TEST[,! (colnames(TEST) %in% 'label')],n.trees = 100, type = "response")
pred_gbm = sapply(pred_gbm, function(value){
  value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))
})
# score_gbm = evaluation(y, pred_gbm)
# score_gbm
proc.time() - time_start

#XGBoost
time_start = proc.time()
library(xgboost)
model_xgboost = xgboost(data = as.matrix(TRAIN_dum[,!(colnames(TRAIN_dum) %in% c('label','id','member_id'))]), 
                        label = as.numeric(as.character(TRAIN_dum[, 'label'])), nrounds = 400, objective = "binary:logistic")
pred_xgboost = predict(model_xgboost, as.matrix(TEST_dum[,! (colnames(TEST_dum) %in% c('label','id','member_id'))]), type = "prob")
pred_xgboost = sapply(pred_xgboost, function(value){
  value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))
})
# score_xgboost = evaluation(y, pred_xgboost)
# score_xgboost
proc.time() - time_start

# # PCA&LDA
# time_start = proc.time()
# train_pca = TRAIN_dum[,-ncol(TRAIN_dum)]
# test_pca = TEST_dum[,-ncol(TEST_dum)]
# # Remove a constant/zero column
# train_pca = train_pca[,apply(train_pca, 2, var, na.rm=TRUE)!=0]
# test_pca = test_pca[,apply(train_pca, 2, var, na.rm=TRUE)!=0]
# library(caret)
# model_pca <- preProcess(train_pca, method="pca", pcaComp=38)
# fit_pca = predict(model_pca,newdata=train_pca)
# pred_pca = predict(model_pca,newdata=test_pca)
# # LDA
# library(MASS)
# train_label = TRAIN_dum[,ncol(TRAIN_dum)]
# train_lda = cbind(fit_pca,train_label)
# test_label = TEST_dum[,ncol(TEST_dum)]
# test_lda = pred_pca
# model_lda = lda(train_label~.,data=train_lda)
# # Fitted value
# fit_lda = predict(model_lda, newdata =train_lda,type='prob')
# # Prediction
# pred_lda = predict(model_lda, newdata =test_lda, type="prob")
# pred_lda=1-pred_lda$posterior
# pred_lda = sapply(pred_lda, function(value){
#   value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))
# })
# score_lda = evaluation(y, pred_lda)
# proc.time() - time_start
# 
# # Naive bayes
# # e1071 package cannot return probabilities
# time_start = proc.time()
# library(naivebayes)
# # The level in train and test dataset must be the same for categorical varaibles, so we have to use dummy coding 
# train_nb=TRAIN[,-ncol(TRAIN)]
# test_nb=TEST[,-ncol(TEST)]
# # Scale numerical variables 
# ind <- sapply(train_nb, is.numeric)
# train_nb[ind] <- lapply(train_nb[ind], scale)
# test_nb[ind] <- lapply(test_nb[ind], scale)
# train_label=TRAIN[,ncol(TRAIN)]
# # Naive bayes model 
# model_nb=naive_bayes(as.factor(train_label)~., data.frame(train_nb))
# fit_nb=predict(model_nb, newdata = train_nb, type="prob")
# pred_nb=predict(model_nb, newdata =test_nb,type='prob')
# pred_nb=pred_nb[,2]
# # Set NA as 0.0001  Accuray do not change
# pred_nb = sapply(pred_nb, function(value){
#   value = ifelse(value>0.999, 0.999, ifelse(value<0.001, 0.001, value))
# })
# pred_nb[is.na(pred_nb)]=0.001
# score_nb=evaluation(y, pred_nb)
# proc.time() - time_start

# Ensemble Model
time_start = proc.time()
pred_ensemble = rowMeans(cbind(pred_logit, pred_ridge, pred_lasso,
                                   pred_rf, pred_gbm, pred_xgboost), na.rm = TRUE)
# score_ensemble = evaluation(y, pred_ensemble)
# score_ensemble
proc.time() - time_start

###################################################################################
###############################   Output   ########################################
###################################################################################

pred1=data.frame("id"=TEST$id,"prob"=pred_rf)
pred2=data.frame("id"=TEST$id,"prob"=pred_xgboost)
pred3=data.frame("id"=TEST$id,"prob"=pred_ensemble)
write.table(pred1, row.names=FALSE, sep = ",", quote = FALSE, "mysubmission1.txt")
write.table(pred2, row.names=FALSE, sep = ",", quote = FALSE, "mysubmission2.txt")
write.table(pred3, row.names=FALSE, sep = ",", quote = FALSE, "mysubmission3.txt")

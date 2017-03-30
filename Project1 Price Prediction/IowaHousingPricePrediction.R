# Iowa Housing Data Prediction Code#
#Methods used: AIC, BIC, Ridge Regression, Lasso, GBM, XGBoost, Random Forest, PCA#

#setwd("~/Dropbox/STAT542/Project/Iowa/")

time_start = proc.time()

# Loading the training and testing data
#From (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
trainset = read.csv('train.csv')
testset = read.csv('test.csv')

                  # Part 1 Preprocessing for train and test datasets
#########################################################################################
# Missing values
#########################################################################################
data_total=rbind(trainset[, colnames(trainset)!="SalePrice"], testset[, colnames(testset)!="SalePrice"])
data_total=data_total[, colnames(data_total)!="Id"]

response_price=trainset[, "SalePrice"]

n_total = nrow(data_total)

# Checking no. of missing values for each variable in train and test set
numNA = colSums(apply(data_total, 2, is.na))
Missing = numNA[which(numNA != 0)]
miss=which(Missing/n_total > 0.6)  #If Missing values are more than 60% of total data, delete the variables.
data_type = sapply(data_total[,names(which(numNA != 0))], class)
cbind(Missing, data_type)

# PoolQC, MiscFeature, Alley, and Fence are missing most of their data so we drop them 
most_miss <- names(miss)
new_most_miss <- setdiff(names(data_total), most_miss)
data_total <- data_total[new_most_miss]

# Treat missing values in other categorical var as new level 'missing'
numNA = colSums(apply(data_total, 2, is.na))
Missing = numNA[which(numNA != 0)]
data_type = sapply(data_total[,names(which(numNA != 0))], class)
cat_var = names(Missing)[which(data_type == 'factor')]  # categorical variables

for (i in 1:length(cat_var)){
  data_total[, cat_var[i]] = as.character(data_total[, cat_var[i]])
  data_total[, cat_var[i]][is.na(data_total[,cat_var[i]])]<- "missing"
  data_total[, cat_var[i]] = as.factor(data_total[, cat_var[i]])
}

# Fill missing values for numerical predictors
numeric_var = names(Missing)[which(data_type != 'factor')]
numeric_var = numeric_var[numeric_var != "LotFrontage"]
data_total$LotFrontage[is.na(data_total$LotFrontage)] = median(na.exclude(data_total$LotFrontage))
for (i in 1:length(numeric_var)){
  tt = data_total[,numeric_var[i]]
  miss_obs = is.na(tt)
  data_total[,numeric_var[i]][miss_obs] = 0
}

# Deal with missing values in response
response_price[is.na(response_price)] = median(na.exclude(response_price))

######################################################################################
# Categorical variables with too many levels
######################################################################################
data1=data_total

format=sapply(data_total, class)
indices=which(format=="factor")

# Identify categorical variables that are stored as numeric values
## For tree methods, we can just treat them as numeric var. For regression, treat as categorical vars.
## But they are ordinal
pre_ordinal = data1[,c("MSSubClass","OverallQual","OverallCond")]

ind_ordinal = which(colnames(data1) %in% c("MSSubClass","OverallQual","OverallCond"))

indices = c(indices, ind_ordinal)

for (i in indices){
  data1[,i]=as.character(data1[,i])
}

row.num=dim(data_total)[1]

for (i in indices){
  aa=summary(as.factor(data1[,i]))
  threshold = 0.05*nrow(data1)  
  aa=aa[which(aa>threshold)]
  name.bb=names(aa)
  
  for (j in 1:row.num){
    value=as.character(data1[j,i])
    if (!(value %in% name.bb)) {data1[j,i]='other'}
  }
}

for (i in indices) {
  data1[,i]=as.factor(data1[,i])
}

data_linear=data1
##############################################################################################
# Log transformation
##############################################################################################
# Linear models generally work better with data that is not highly skewed.
# One way to reduce skewness is by a log transform. 

# Data_total
# Check the numbers of continuous and categorical features
data.type = sapply(data_total, class)
table(data.type)

data_total_log = data_total

# Determine skewness of response values
library(moments)
skewness(response_price)
# Transform SalePrice target to log form
response_price <- log(response_price + 1)
skewness(response_price)

# For numeric feature with excessive skewness, perform log transformation
# Pick up continuous features
int_var =  names(data_total_log)[which(data.type == 'integer')]
num_var =  names(data_total_log)[which(data.type == 'numeric')]
numeric_var = c(int_var, num_var)
# Determine skewness for each numeric feature
skewed_feats = sapply(data_total_log[, numeric_var], skewness)
# Only transform features that exceed a threshold = 0.75 for skewness
skewed_feats = numeric_var[which(skewed_feats > 0.75)]
for(j in skewed_feats) {
  data_total_log[, j] = log(data_total_log[, j] + 1)
}

# Data_linear
# Check the numbers of continuous and categorical features
data.type = sapply(data_linear, class)
table(data.type)
# Drop ID column
data_linear_log = data_linear

# For numeric feature with excessive skewness, perform log transformation
# Pick up continuous features
int_var =  names(data_total_log)[which(data.type == 'integer')]
num_var =  names(data_total_log)[which(data.type == 'numeric')]
numeric_var = c(int_var, num_var)
# Determine skewness for each numeric feature
skewed_feats = sapply(data_linear_log[, numeric_var], skewness)
# Only transform features that exceed a threshold = 0.75 for skewness
skewed_feats = numeric_var[which(skewed_feats > 0.75)]
for(j in skewed_feats) {
  data_linear_log[, j] = log(data_linear_log[, j] + 1)
}
##############################################################################################
# Seperate the predictors of training dataset and test dataset
##############################################################################################
trainset1 = cbind(data_total_log[1:nrow(trainset),], response_price) 
testset1 = data_total_log[-(1:nrow(trainset)), ]

trainset2 = cbind(data_linear_log[1:nrow(trainset),], response_price) 
testset2 = data_linear_log[-(1:nrow(trainset)), ]

# The datasets we finally use are data_linear_log(without Id column) 
# and data_total_log(without reducing levels of categorical vars and without Id column)
######################################################################################
#################   End of Preprocessing    ##########################################
######################################################################################

                       #Part 2: Modeling with train dataset
##############################################################################################
# Multiple linear regression
##############################################################################################
model_lm = lm(response_price ~ LotArea + Street + Condition1 + OverallQual + 
              OverallCond + YearBuilt + YearRemodAdd + BsmtFinSF1 + TotalBsmtSF + 
              CentralAir + X2ndFlrSF + GrLivArea + BsmtFullBath + FullBath + 
              KitchenAbvGr + KitchenQual + Functional + Fireplaces + GarageCars + 
              ScreenPorch + SaleCondition, data =trainset2)
# Fitted values
fitted_lm = predict(model_lm, newdata = trainset2)
fitted_lm = exp(fitted_lm) - 1
# Prediction
pred_lm = predict(model_lm, newdata= testset2)
pred_lm = exp(pred_lm) - 1
##############################################################################################
# Ridge and Lasso 
##############################################################################################
# Change categorical variables to dummy variables
Y = response_price
trainset2.dum = model.matrix(Y ~ . - 1, data = trainset2[, colnames(trainset2)!="response_price"])
testset2.dum = model.matrix(~.-1, data = testset2)

# Ridge regression
library(glmnet) 
model_ridge = glmnet(trainset2.dum, Y, alpha=0)

#CV error, lambda sequence set by glmnet
cv.out = cv.glmnet(trainset2.dum, Y, alpha=0)  

# lambda.min
bestlam.ridge = cv.out$lambda.min
# Fitted values
fitted_ridge = predict(model_ridge, s=bestlam.ridge, newx = trainset2.dum)
fitted_ridge = exp(fitted_ridge) - 1
# Prediction
pred_ridge = predict(model_ridge,s=bestlam.ridge ,newx=testset2.dum)
pred_ridge = exp(pred_ridge) - 1

# LASSO regression
model_lasso = glmnet(trainset2.dum, Y, alpha=1)

# CV error
cv.out = cv.glmnet(trainset2.dum, Y, alpha=1)

# lambda.min
bestlam.lasso = cv.out$lambda.min
# Fitted values
fitted_lasso = predict(model_lasso, s=bestlam.lasso, newx = trainset2.dum)
fitted_lasso = exp(fitted_lasso) - 1
# Prediction
pred_lasso = predict(model_lasso, s=bestlam.lasso, newx=testset2.dum)
pred_lasso = exp(pred_lasso) - 1

##############################################################################################
# Random forest 
##############################################################################################
library(randomForest)
set.seed(4547)
model_rf=randomForest(response_price~., data=trainset1)

# Fitted values
fitted_rf = predict(model_rf, newdata = trainset1)
fitted_rf = exp(fitted_rf) - 1
# Prediction
pred_rf = predict(model_rf, newdata = testset1)
pred_rf = exp(pred_rf) - 1

##############################################################################################
# GBM
##############################################################################################
library(gbm)
set.seed(8642)
model_gbm = gbm(response_price~., data=trainset1, distribution = "gaussian", 
                n.trees = 500, shrinkage = 0.05, bag.fraction = 0.7)

# Fitted values
fitted_gbm = predict(model_gbm, newdata = trainset1, n.trees = 500)
fitted_gbm = exp(fitted_gbm) - 1
# Prediction
pred_gbm = predict(model_gbm, newdata = testset1, n.trees = 500)
pred_gbm = exp(pred_gbm) - 1

##############################################################################################
# PCA
##############################################################################################
library(pls)
# Modeling using pcr
trainset_PCA = cbind(Y, as.data.frame(trainset2.dum))
model_pcr=pcr(Y~., data=trainset_PCA, validation="CV")

# Fitted values
fitted_pca = predict(model_pcr, data=trainset_PCA, ncomp = 120)
fitted_pca = exp(fitted_pca) - 1
# Prediction
pred_pca = predict(model_pcr, newdata = as.data.frame(testset2.dum), ncomp = 120)
pred_pca = exp(pred_pca) - 1

##############################################################################################
# XGboost
##############################################################################################
# Change categorical variables to dummy variables
Y = response_price
trainset1.dum = model.matrix(Y ~ . - 1, data = trainset1[, colnames(trainset1)!="response_price"])
testset1.dum = model.matrix(~.-1, data = testset1)

set.seed(100)
library(xgboost)
# Set nround = 300
model_xgb = xgboost(data = trainset1.dum, label = Y, max.depth = 10, eta = 0.03, nround = 300,gamma=0.1, objective = "reg:linear")
file.remove("xgboost.model")
# Fitted values
fitted_xgb = predict(model_xgb, newdata=trainset1.dum)
fitted_xgb = exp(fitted_xgb) - 1
# Prediction
pred_xgb = predict(model_xgb, testset1.dum) 
pred_xgb = exp(pred_xgb) - 1
##############################################################################################
# Ensemble Model
##############################################################################################
pred_ese=rowMeans(cbind(pred_lm,pred_ridge,pred_lasso,pred_rf,pred_gbm,pred_pca,pred_xgb))

######################################################################################
#################   End of modeling    ##########################################
######################################################################################

                  #Part 3: output
##############################################################################################
# Output
##############################################################################################
write.table(pred_lm, "mysubmission1.txt", sep=",", quote = FALSE)
write.table(pred_xgb, "mysubmission2.txt", sep=",",quote = FALSE)
write.table(pred_ese, "mysubmission3.txt", sep=",",quote = FALSE)

proc.time() - time_start

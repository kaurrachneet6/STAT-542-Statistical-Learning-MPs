# STAT-542 Statistical Learning, UIUC Machine Problems 
The repository contains the R coding projects for STAT 542 Statistical Learning course

**Project 1:** R code to predict the prices for the Iowa Housing Data

The houses prices data used is downloaded from Kaggle (https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It contains 1460 recent house sold records and 81 explanatory variables describing (almost) every aspect of these houses. 
The first prediction method used is linear regression with feature selection using AIC and BIC. Then after evaluating the performance of various advanced models applied to predict house sale prices, it is found that XGBoost and ensemble model are robust and well-performed, hence they are used as second and third prediction methods for the sale prices.

**Project 2:** Lending club loan data analysis

Create a model that predicts whether or not a loan will be default using the historical data. The historical loan data we use is downloaded from Kaggle. It contains 887379 records and 74 explanatory variables. In our data preprocessing procedure, we drop some irrelevant variables, dealing with some unusual values, fill missing values, combine levels of categorical variables and dummy coding. Then we apply various advanced models to predict the loan status. After evaluating their performance, we find that xgboot, random forest and ensemble model are robust and well-performed, so we set them as our three prediction methods.

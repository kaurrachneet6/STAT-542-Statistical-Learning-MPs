# STAT-542 Statistical Learning, UIUC Machine Problems 
The repository contains the R coding projects for STAT 542 Statistical Learning course

**Project 1: R code to predict the prices for the Iowa Housing Data**

The houses prices data used is downloaded from Kaggle (https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It contains 1460 recent house sold records and 81 explanatory variables describing (almost) every aspect of these houses. 
The first prediction method used is linear regression with feature selection using AIC and BIC. Then after evaluating the performance of various advanced models applied to predict house sale prices, it is found that XGBoost and ensemble model are robust and well-performed, hence they are used as second and third prediction methods for the sale prices.

**Project 2: Lending club loan data analysis**

Create a model that predicts whether or not a loan will be default using the historical data. The historical loan data we use is downloaded from Kaggle. It contains 887379 records and 74 explanatory variables. In our data preprocessing procedure, we drop some irrelevant variables, dealing with some unusual values, fill missing values, combine levels of categorical variables and dummy coding. Then we apply various advanced models to predict the loan status. After evaluating their performance, we find that xgboot, random forest and ensemble model are robust and well-performed, so we set them as our three prediction methods.

**Project 3: Movie Recommendation System**

Our goal is to build a movie recommender system based on the MovieLens 1M Dataset. 
In the train.dat, it contains about 60% rows of the ratings.dat from the MovieLens 1M dataset (of the same format). And in the test.csv, it contains about 20% of the user-movie pairs from the ratings.dat from the MovieLens 1M dataset. In our data preprocessing procedure, we get the rating matrix and the movie feature matrix at first. Then we apply content-based method and collaborative filtering method to predict the ratings.

**Project 4: Sentiment Analysis** 

The labeled training data set consists of 25,000 IMDB movie reviews, while the test data set contains 25,000 reviews without labels. In this project, we need to predict the label of the test data using sentiment analysis techniques.

**Project 5: Predict Walmart sales** 

The Walmart data we use is downloaded from Kaggle. There are 45 stores and 81 departments in this data set. Our goal is to predict the weekly sales in each store under each department. First, we fill missing values in our data set and plot our data to find their patterns. Then, we applied three models including Seasonal Naïve Method, Product Method and ARIMA Model to do prediction and evaluate their performance using WMAE.

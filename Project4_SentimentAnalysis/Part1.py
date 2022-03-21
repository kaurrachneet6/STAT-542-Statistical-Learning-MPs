# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup             
import re
import nltk
from nltk.corpus import stopwords 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from gensim import models

############read data############################  

labeled_train = pd.read_csv("labeledTrainData.tsv", header=0, \
                    delimiter="\t", quoting=3)

# Read the test data
testData = pd.read_csv("testData.tsv", header=0, delimiter="\t", \
                   quoting=3 )


labeled_train=labeled_train.dropna(axis=0,how='any')
#train,test = train_test_split(labeled_train, test_size = 0.25)
train=labeled_train
test=testData
#########################################################



##############preprocessing function########################
def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,'lxml').get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  
##############################################################




#############applying preprocessing function on train dataset####
num_reviews = train["review"].size
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
#drop nan in train 
train=train.dropna(axis=0,how='any')
for i in train.index:
    # If the index is evenly divisible by 1000, print a message
    #if( str(train["review"][i])!='nan'):
    #if(train['review'][i].isnull()==False&train['sentiment'][i].isnull()==False):
    clean_train_reviews.append( review_to_words( train["review"][i] ))
###############################################################    
    


###################apply preprocessing function on test data####################
print(test.shape)

#num_reviews = len(test["review"])
clean_test_reviews = [] 

for i in test.index:
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

##############################################################################


'''
###############create bag of words##############################
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
cv= CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000, \
                             ngram_range=(1,1),
                             max_df=0.5)
             


# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
##########train####
cv_train_features = cv.fit_transform(clean_train_reviews)
# Numpy arrays are easy to work with, so convert the result to an 
# array
cv_train_features = cv_train_features.toarray()
###########test######
cv_test_features = cv.transform(clean_test_reviews)
cv_test_features = cv_test_features.toarray()
#################################################################
'''



#################TF-IDF#####################################
tfidf_vectorizer = TfidfVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
tfidf_train_features = tfidf_vectorizer.fit_transform(clean_train_reviews)
tfidf_train_features = tfidf_train_features.toarray()
tfidf_vocab = tfidf_vectorizer.get_feature_names()  
tfidf_test_features = tfidf_vectorizer.transform(clean_test_reviews)
tfidf_test_features = tfidf_test_features.toarray() 
###############################################################

'''
###############random forest###########################
cv_rf = RandomForestClassifier(n_estimators = 100)  #100 trees
#cv
cv_rf = cv_rf.fit(cv_train_features, train["sentiment"] )
cv_rf_fit = cv_rf.predict_proba(cv_train_features)
cv_rf_pred = cv_rf.predict_proba(cv_test_features)
roc_cv_rf_fit=roc_auc_score(train['sentiment'], cv_rf_fit[:,1])
roc_cv_rf_pred=roc_auc_score(test['sentiment'], cv_rf_pred[:,1])
#tfidf
tfidf_rf = RandomForestClassifier(n_estimators = 100)  #100 trees
tfidf_rf = tfidf_rf.fit(tfidf_train_features, train["sentiment"] )
tfidf_rf_fit = tfidf_rf.predict_proba(tfidf_train_features)
tfidf_rf_pred = tfidf_rf.predict_proba(tfidf_test_features)
roc_tfidf_rf_fit=roc_auc_score(train['sentiment'], tfidf_rf_fit[:,1])
roc_tfidf_rf_pred=roc_auc_score(test['sentiment'], tfidf_rf_pred[:,1])
################################################################
'''

################### LASSO ###############################
'''
#cv
cv_lasso = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
#print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(model, X, y, cv=20, scoring='roc_auc'))

#print "Retrain on all training data, predicting test labels...\n"
cv_lasso.fit(cv_train_features, train["sentiment"] )
cv_lasso = cv_lasso.fit(cv_train_features, train["sentiment"] )

cv_lasso_pred = cv_lasso.predict_proba(cv_test_features)

roc_cv_lasso_pred=roc_auc_score(test['sentiment'], cv_lasso_pred[:,1])
'''
#tfidf
tfidf_lasso = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)

tfidf_lasso.fit(tfidf_train_features, train["sentiment"] )
tfidf_lasso_fit=tfidf_lasso.predict_proba(tfidf_train_features)[:,1]
tfidf_lasso_pred = tfidf_lasso.predict_proba(tfidf_test_features)[:,1]
#roc_tfidf_lasso_pred=roc_auc_score(test['sentiment'], tfidf_lasso_pred)
#################################################################


'''
###################lda+lasso##############################
cv = TfidfVectorizer(stop_words = 'english',
                     lowercase=True,
                     min_df=20,
                     max_df=0.6,
                     ngram_range=(1,1),
                     max_features=500)
                     
train_review= cv.fit_transform(train['review'])
test_review= cv.transform(test['review'])

num_topics=5
lda = LatentDirichletAllocation(n_topics=num_topics, max_iter=5,
                                learning_method='online', learning_offset=5.,
                                random_state=23).fit(train_review)

#lda_topics = tp.get_topics(cv, lda)
train_lda=lda.transform(train_review)
test_lda=lda.transform(test_review)


lasso = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)

lasso.fit(train_lda, train["sentiment"] )
lda_lasso_fit=lasso.predict_proba(train_lda)
lda_lasso_pred = lasso.predict_proba(test_lda)

########################################################################
'''
'''
#########################xgboost######################################
import xgboost as xgb
xg_train = xgb.DMatrix(tfidf_train_features, label=sentiment_train)
num_round = 10 #number of rounds for boosting
params = {'booster':'gbtree', 'max_depth': 10, 'objective':'binary:logistic', \
              'eval_metric': 'auc', 'silent': 1}
bst = xgb.train(params, xg_train, num_round) 

#Testing
xg_test = xgb.DMatrix(tfidf_test_features)
#create a submission
result = bst.predict(xg_test) #SOFT CLASSIFICATION
roc_tfidf_xgboost_fit=roc_auc_score(test['sentiment'], result)
#####################################################################
'''
############create submission file#######################################
output = pd.DataFrame( data={"id":test["id"], "sentiment":tfidf_lasso_pred} )
# comma-separated 
output.to_csv( "KaggleSubmission.csv", index=False, quoting=3 )
#output.to_csv( "C:/UIUC/class2017Spring/542/project4/Bag_of_Words_model.csv", index=False, quoting=3 )
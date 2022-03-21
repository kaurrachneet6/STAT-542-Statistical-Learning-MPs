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
from sklearn.linear_model import LogisticRegression
import operator
from collections import Counter
#from nltk import pos_tag
#import re

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
    #rgxs = re.compile(r"(JJ|NN|VBN|VBG)")
    #rgxs = re.compile(r"(JJ)")
    #ptgs = pos_tag(rem_stop_words)
    
    #meaningful_words = [tkn[0] for tkn in ptgs if re.match(rgxs, tkn[1])]
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
    clean_train_reviews.append( review_to_words( train["review"][i] ))
###############################################################    
    


###################apply preprocessing function on test data####################
print(test.shape)
clean_test_reviews = [] 

for i in test.index:
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )

##############################################################################



#################TF-IDF#####################################
tfidf_vectorizer = TfidfVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
tfidf_train_features = tfidf_vectorizer.fit_transform(clean_train_reviews)
tfidf_train_features = tfidf_train_features.toarray()
tfidf_vocab = tfidf_vectorizer.get_feature_names()  
tfidf_test_features = tfidf_vectorizer.transform(clean_test_reviews)
tfidf_test_features = tfidf_test_features.toarray() 
###############################################################



################### LASSO ###############################
lasso = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)
lasso.fit(tfidf_train_features, train["sentiment"] )
tfidf_lasso_fit=lasso.predict_proba(tfidf_train_features)[:,1]
tfidf_lasso_pred = lasso.predict_proba(tfidf_test_features)[:,1]
#roc_tfidf_lasso_pred=roc_auc_score(test['sentiment'], tfidf_lasso_pred)
#################################################################

########################create word lists##########################################
#total_class=lasso.predict(tfidf_test_features)
#total = test
#total['sentiment'] = total_class
total=pd.DataFrame(data={"review":clean_train_reviews,'sentiment':train['sentiment']})
pos_total=total.loc[total['sentiment'] == 1]
neg_total=total.loc[total['sentiment'] == 0]
pos_review=pos_total['review']
neg_review=neg_total['review']

####################positive words frequency list#############
cv= CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 1000, \
                             ngram_range=(1,1), \
                             binary=False,\
                             max_df=0.5)
             
cv_pos = cv.fit_transform(pos_review)
pos_words=cv.get_feature_names()
pos_counts = cv_pos.sum(axis=0).A1
pos_freq = dict(Counter(dict(zip(pos_words, pos_counts))))
######################################################################



#######################negative words frequency list##################
cv= CountVectorizer(analyzer = "word",   \
                             max_features = 1000, \
                             ngram_range=(1,1), \
                             binary=False,\
                             max_df=0.5,\
                             stop_words = 'english')
             
cv_neg = cv.fit_transform(neg_review)
neg_words=cv.get_feature_names()
neg_counts = cv_neg.sum(axis=0).A1
neg_freq = dict(Counter(dict(zip(neg_words, neg_counts))))


#####################get the difference between pos and neg word list###
diff1=pos_freq.keys()-neg_freq.keys()
positive=dict((k, pos_freq[k]) for k in diff1)
diff2=neg_freq.keys()-pos_freq.keys()
negative=dict((k, neg_freq[k]) for k in diff2)

sorted_pos = sorted(positive.items(), key=operator.itemgetter(1),reverse=True)
sorted_neg = sorted(negative.items(), key=operator.itemgetter(1),reverse=True)
#######################################################################


###################get final expos, pos, exneg, neg word lists###############
pos_boundary=int(len(sorted_pos)/2)
ex_pos_words=np.array(sorted_pos)[0:pos_boundary,0]
pos_words=np.array(sorted_pos)[pos_boundary+1:,0]

neg_boundary=int(len(sorted_neg)/2)
ex_neg_words=np.array(sorted_neg)[0:neg_boundary,0]
neg_words=np.array(sorted_neg)[neg_boundary+1:,0]

ex_pos_output = pd.DataFrame(data={"ex_pos":ex_pos_words})
pos_output = pd.DataFrame(data={"pos":pos_words})
ex_neg_output = pd.DataFrame(data={"ex_neg":ex_neg_words})
neg_output = pd.DataFrame(data={"neg":neg_words})
#################################################################



###################create word list file#################################
pos_output.to_csv( "positive.csv", index=False, quoting=3,header=None )
neg_output.to_csv( "negative.csv", index=False, quoting=3 ,header=None)
ex_pos_output.to_csv( "ex_positive.csv", index=False, quoting=3, header=None)
ex_neg_output.to_csv( "ex_negative.csv", index=False, quoting=3, header=None)


############create submission file#######################################
#output = pd.DataFrame( data={"id":test["id"],"sentiment":tfidf_lasso_pred} )
# comma-separated 
#output.to_csv( "C:/UIUC/class2017Spring/542/project4/Bag_of_Words_model.csv", index=False, quoting=3 )
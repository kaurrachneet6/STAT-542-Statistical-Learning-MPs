**Project 4:** Sentiment Analysis 

The labeled training data set consists of 25,000 IMDB movie reviews, while the test data set contains 25,000 reviews without labels. In this project, we need to predict the label of the test data using sentiment analysis techniques.

For preprocessing, first, we turn the reviews into a corpus. Then, remove the punctuation, html symbols, stopwords, numbers and extra blanks from the reviews, and turn all words to lower case. Next, transform the review data into a word matrix and adjust the sparsity of the matrix. During the transformation, we try different models, including Bag of Word, TF-IDF and LDA. Finally, turn the matrix and the label vector into a data frame. In this way we get the data frame for modeling.

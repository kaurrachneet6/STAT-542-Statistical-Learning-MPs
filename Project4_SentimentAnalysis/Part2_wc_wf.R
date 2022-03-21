#setwd("~/Downloads/")

# load data
require(data.table)
#trainData = fread(file="labeledTrainData.tsv", header = TRUE, sep = "\t", quote = "\"", stringsAsFactors = FALSE)
testData = fread(file="testData.tsv", header = TRUE, sep = "\t", quote = "\"", stringsAsFactors = FALSE)
#alldata <- rbind(trainData[,c(1,3)], testData)
#data = trainData[,c(1,3)]

pred = read.csv('KaggleSubmission.csv')
#neg = read.csv('negative.csv')
#pos = read.csv('positive.csv')
#ex_neg = read.csv('ex_negative.csv')
#ex_pos = read.csv('ex_positive.csv')
testData$sentiment = ifelse(pred[,2]>0.5,1,0)
data_1 = testData[sentiment==1][,-3]
data_0 = testData[sentiment==0][,-3]

# y = 1
library(tm)
data = data_1
full_text = Corpus(VectorSource(data$review))
full_text = tm_map(full_text, content_transformer(tolower))
full_text = tm_map(full_text, removeNumbers)
full_text = tm_map(full_text, removePunctuation)
full_text = tm_map(full_text, removeWords, stopwords("english"))
full_text = tm_map(full_text, stripWhitespace)
#word_freq = DocumentTermMatrix(full_text)
#word_freq = removeSparseTerms(word_freq, 0.95)

#tf-idf
word_tfidf = DocumentTermMatrix(full_text, control = list(weighting = weightTfIdf))
word_tfidf = removeSparseTerms(word_tfidf, 0.99)

#train <- as.data.frame(as.matrix(word_tfidf))
#train$y <- trainData$sentiment

test_1 <- as.data.frame(as.matrix(word_tfidf))

# y = 0
data = data_0
full_text = Corpus(VectorSource(data$review))
full_text = tm_map(full_text, content_transformer(tolower))
full_text = tm_map(full_text, removeNumbers)
full_text = tm_map(full_text, removePunctuation)
full_text = tm_map(full_text, removeWords, stopwords("english"))
full_text = tm_map(full_text, stripWhitespace)
#word_freq = DocumentTermMatrix(full_text)
#word_freq = removeSparseTerms(word_freq, 0.95)

#tf-idf
word_tfidf = DocumentTermMatrix(full_text, control = list(weighting = weightTfIdf))
word_tfidf = removeSparseTerms(word_tfidf, 0.99)

#train <- as.data.frame(as.matrix(word_tfidf))
#train$y <- trainData$sentiment

test_0 <- as.data.frame(as.matrix(word_tfidf))


# Visualization
library(wordcloud)
library(RColorBrewer)
# test
# y = 1
data = test_1
v_1 <- sort(colSums(data),decreasing=TRUE)

# y = 0
data = test_0
v_0 <- sort(colSums(data),decreasing=TRUE)

diff_1 = setdiff(names(v_1), names(v_0))
diff_0 = setdiff(names(v_0), names(v_1))
v_1_n = v_1[names(v_1) %in% diff_1]
v_0_n = v_0[names(v_0) %in% diff_0]
d_1 <- data.frame(word = names(v_1_n),freq=v_1_n)
d_0 <- data.frame(word = names(v_0_n),freq=v_0_n)

set.seed(1234)
wordcloud(words = d_1$word, freq = d_1$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
barplot(d_1[1:10,]$freq, las = 2, names.arg = d_1[1:10,]$word,
        col ="lightblue", main ="Most frequent words for sentiment = 1",
        ylab = "Word frequencies")
set.seed(1234)
wordcloud(words = d_0$word, freq = d_0$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
barplot(d_0[1:10,]$freq, las = 2, names.arg = d_0[1:10,]$word,
        col ="lightblue", main ="Most frequent words for sentiment = 0",
        ylab = "Word frequencies")
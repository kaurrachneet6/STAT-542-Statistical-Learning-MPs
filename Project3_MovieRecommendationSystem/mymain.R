#setwd('D:\\542 Project\\Recommender System\\ml-1m')
#setwd('~/Downloads/')

pkg = c('data.table', 'reshape2', 'matrixStats','recommenderlab')  # all packages you will use in your code
# Do not include unnecessary packages.
for (i in pkg){
  if (!require(i, character.only = TRUE)){
    install.packages(i)
  }
}

library(data.table)

#Rating matrix
ratings = readLines('train.dat')
ratings = gsub('::',',',ratings)
writeLines(ratings,'ratings11.csv')
ratings = fread('ratings11.csv', sep = ',', header=FALSE,stringsAsFactors = F)

colnames(ratings) = c('UserID','MovieID','Rating','Timestamp')
ratings = ratings[,-4]
ratings = as.data.frame(ratings)

#Read in the test dataset
test = read.csv('test.csv', sep = ',')

# #Draw sample for testing
# set.seed(543)
# ind = sample(1:nrow(ratings),nrow(ratings)/5)
# test = ratings[ind,]
# ratings = ratings[-ind,]
#####################CF based recommendation##########################################################
library(reshape2)
rating_m = acast(ratings, UserID ~ MovieID, value.var ='Rating') # This is the final rating matrix, rownames--UserID, colnames--MovieID
R = as(rating_m, 'realRatingMatrix')
R_m = normalize(R)
recommenderRegistry$get_entries(dataType = "realRatingMatrix")
#rec = Recommender(R, method = 'UBCF',param = list(normalize = 'Z-score', method = 'Cosine', nn =5))
#rec = Recommender(R,method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=5))
#rec = Recommender(R,method="IBCF", param=list(normalize = "Z-score",method="Jaccard"))
rec = Recommender(R,method="POPULAR")

recom = predict(rec, R, type = 'ratings')  # predict ratings. This may be slow.
rec_list = as(recom, 'list')  # each element are ratings of that user
##################################################################################################


############################content based recommendation##########################################
#Movie features matrix
movie_f = readLines('movies.dat')
movie_f = gsub('::','=', movie_f)
writeLines(movie_f, 'movie_1.dat')
movie_f = fread('movie_1.dat', sep='=',stringsAsFactors = F, header = F)

colnames(movie_f) = c('MovieID','Title','Genres')

genre = strsplit(movie_f$Genres,'[|]')
genre_uni = unique(unlist(genre))
genre_uni = sort(genre_uni)

movie_fm = matrix(0,nrow =nrow(movie_f),ncol=length(genre_uni)+1)
movie_fm[,1] = movie_f$MovieID
colnames(movie_fm) = c('MovieID',genre_uni)

movie_t1 = matrix(0,nrow = 0,ncol=length(genre_uni))

temp1 = sapply(movie_f$Genres,function(gen1){
  inclu = sapply(genre_uni,function(mt){
    t1 = grepl(mt,gen1)
  })
  movie_t1 <<- rbind(movie_t1,inclu)
})

movie_fm[,2:ncol(movie_fm)] = movie_t1 # This is the movie feature matrix, first col is MovieID

#Similarity (correlation matrix)
movie_trans = t(movie_fm)[-1,]
sim1 = cor(movie_trans)  # This is the correlation matrix of movie pairs, should be symmetric
colnames(sim1) = movie_fm[,1]
rownames(sim1) = movie_fm[,1]
#######################################################################################################


#######################################################################################################
#Seperate test into two parts: matched and unmatched
#For new movies, just use content-based for prediction
#For new users, we need to consider seperately
index = test$user %in% ratings$UserID
test_newU = test[!index,]
test_oldU = test[index,]

index2 = test_oldU$movie %in% ratings$MovieID
test_oo = test_oldU[index2,]
test_on = test_oldU[!index2,]



#####pred old user,old movie###############
test_oo$pred = NA

#pred all lines in test_oo
b = sapply(1:nrow(test_oo), function(i)
{
  userid <- test_oo[i,2]
  movieid<-test_oo[i,3]
  u1<-as.data.frame(rec_list[[which(names(rec_list)==as.character(userid))]])
  # Create a (second column) column-id in the data-frame u1 and populate it with row-names
  # Remember (or check) that rownames of u1 contain are by movie-ids
  # We use row.names() function
  u1$id<-row.names(u1)
  # Now access movie ratings in column 1 of u1
  x= u1[u1$id==movieid,1]
  if (length(x)==0)
  {
    test_oo$pred[i]=0
  }
  else
  {
    test_oo$pred[i]=x
  }
  return(test_oo[i,])
}
)

test_oo= as.data.frame(t(as.matrix(b)))
###########################################


#Create user groups
user = readLines('users.dat')
user = gsub('::',',',user)
user = read.table(text=user,sep = ',')
colnames(user) = c('UserID','Gender','Age','Occupation','ZipCode')
index_u = split(1:nrow(user), list(user$Age,user$Gender))
userid_loc = lapply(index_u, function(ind1){
  user_id = user[ind1,'UserID']
  r_names = rownames(rating_m)
  location = r_names[r_names %in% user_id]
  return(location)
})

predict_newU = function(ind2){
  u_id = test_newU[ind2,'user']
  info = user[user$UserID == u_id,c('Age','Gender')]
  info = paste(info[1,1],info[1,2],sep = '.')
  group_id = which(names(index_u) == info)
  
  m_id = as.character(test_newU[ind2,'movie'])
  if(m_id %in% colnames(rating_m)){
  temp_vec = rating_m[userid_loc[[group_id]], m_id]
  pred = mean(na.exclude(temp_vec))
  pred = ifelse(is.na(pred), 3, pred)} else {
    pred = 3
  }
  return(pred)
}

pred_n = sapply(1:nrow(test_newU), predict_newU)
test_newU$pred = pred_n


# Content-based
diff = setdiff(colnames(sim1), colnames(rating_m))
sim = sim1[!colnames(sim1) %in% diff,]

# Normalization
library(matrixStats)
normal = function(rating){
  rowM = rowMeans(rating, na.rm = TRUE)
  rowSd = apply(rating,1,sd,na.rm=TRUE)
  for(i in 1:ncol(rating)){
    rating[,i] = (rating[,i] - rowM)/rowSd
  }
  return(rating)
}
rowM = rowMeans(rating_m, na.rm = TRUE)
rowSd = apply(rating_m,1,sd,na.rm=TRUE)
rating_n = normal(rating_m)

## fill NA in rating_m with 0 and build a binary rating_m called rating_bi
rating_bi = rating_m
rating_bi[! is.na(rating_bi)] = 1
rating_bi[is.na(rating_bi)] = 0
rating_m[is.na(rating_m)] = 0
rating_n[is.na(rating_n)] = 0

## Calculate dot product as rating prediction
predRating = function(test, rating, ratingbi){
  user = as.character(test$user)
  movie = as.character(test$movie)
  id = test$ID
  pred = NULL
  for (i in 1:length(user)){
    row = rating[rownames(rating) == user[i],]
    row_1 = ratingbi[rownames(ratingbi) == user[i],]
    col = sim[, colnames(sim) == movie[i]]
    pro = row %*% col
    sum = row_1 %*% col
    rate = pro/sum
    pred = c(pred, rate)
  }
  pred = ifelse(pred>5,5,ifelse(pred<1,1,pred))
  return(pred)
}
predRating_n = function(test, rating, ratingbi){
  user = as.character(test$user)
  movie = as.character(test$movie)
  id = test$ID
  pred = NULL
  for (i in 1:length(user)){
    row = rating[rownames(rating) == user[i],]
    row_1 = ratingbi[rownames(ratingbi) == user[i],]
    col = sim[, colnames(sim) == movie[i]]
    pro = row %*% col
    sum = row_1 %*% col
    rate = pro/sum
    rate = rowSd[which(rownames(rating) == user[i])] * rate + rowM[which(rownames(rating) == user[i])]
    pred = c(pred, rate)
  }
  pred = ifelse(pred>5,5,ifelse(pred<1,1,pred))
  return(pred)
}
## Final rating predicton
#pred_m = predRating(test_oldU, rating_m, rating_bi)
pred_n = predRating_n(test_oldU, rating_n, rating_bi)
#test_oldU$rating = pred_m
test_oldU$pred = pred_n

#Extract old-user but new-movie records
test_on$pred = test_oldU[(test_oldU$ID %in% test_on$ID), 'pred']

##################################################################################

#Combine the results and output
#1. Content-based
total_test = rbind(test_oldU, test_newU)
total_test = total_test[order(total_test$ID),]
submission1 = data.frame(ID=total_test$ID, user=total_test$user, movie=total_test$movie,rating=total_test$pred)
write.csv(submission1, 'mysubmission1.csv',row.names = F)

#2. CF
test_oo1=data.frame(ID=as.numeric(test_oo$ID),user = as.numeric(test_oo$user),
                   movie = as.numeric(test_oo$movie),pred = as.numeric(test_oo$pred))
total_test = rbind(test_newU,test_on,test_oo1)
total_test = total_test[order(total_test$ID),]
submission2 = data.frame(ID=total_test$ID, user=total_test$user, movie=total_test$movie,rating=total_test$pred)
write.csv(submission2, 'mysubmission2.csv',row.names = F)

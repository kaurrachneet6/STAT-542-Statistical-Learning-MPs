#setwd("C:/UIUC/class2017Spring/542/project4")

#setwd("D:\\542 Project\\Sentiment Analysis")

#The test data should include predicted label as a column, with same format as train data
library(data.table)
myreview=fread("testData.tsv",
        header = T,  stringsAsFactors = F)
myreview = as.data.frame(myreview)
label_pre = read.csv("KaggleSubmission.csv", header = T, stringsAsFactors = F)
temp1 = cbind(myreview,label_pre$sentiment)
myreview = temp1[,c(1,3,2)]
colnames(myreview) = c('id','sentiment','review')
#myreview = myreview[1:10,]

# remove HTML tags
myreview[,3] = gsub("<.*?>", " ", myreview[,3])

#Read in word lists
wordlist.neg=read.csv("negative.csv", stringsAsFactors = F, header = F)[,1]
wordlist.pos=read.csv("positive.csv", stringsAsFactors = F, header = F)[,1]
wordlist.exneg=read.csv("ex_negative.csv", stringsAsFactors = F, header = F)[,1]
wordlist.expos=read.csv("ex_positive.csv", stringsAsFactors = F, header = F)[,1]

myfile = "sentiment_output.html"
if (file.exists(myfile)) file.remove(myfile)
n.review = dim(myreview)[1]
 
## create html file
write(paste("<html> \n", 
            "<head> \n",  
            "<style> \n",
            "@import \"textstyle1.css\"", 
            "</style>", 
            "</head> \n <body>\n"), file=myfile, append=TRUE)
write("<ul>", file=myfile, append=TRUE)

for(i in 1:n.review){
  write(paste("<li><strong>", myreview[i,1], 
              "</strong> sentiment =", myreview[i,2], 
              "<br><br>", sep=" "),
        file=myfile, append=TRUE)
  myreview[i,3] = sub(",", ", ", myreview[i,3])
  tmp = strsplit(myreview[i,3], " +")[[1]]
  
  tmp.copy = tmp
  nwords = length(tmp)
  
  pos=NULL;
  for(j in 1:length(wordlist.neg))
    pos = c(pos, grep(wordlist.neg[j], tmp, ignore.case = TRUE))
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"neg\">", 
                                   tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  pos=NULL;
  for(j in 1:length(wordlist.pos))
    pos = c(pos, grep(wordlist.pos[j], tmp, ignore.case = TRUE))
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"pos\">", 
                               tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  pos=NULL;
  for(j in 1:length(wordlist.exneg))
    pos = c(pos, grep(wordlist.exneg[j], tmp, ignore.case = TRUE))
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"exneg\">", 
                               tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  pos=NULL;
  for(j in 1:length(wordlist.expos))
    pos = c(pos, grep(wordlist.expos[j], tmp, ignore.case = TRUE))
  if (length(pos)>0) {
    for(j in 1:length(pos)){
      tmp.copy[pos[j]] = paste("<span class=\"expos\">", 
                               tmp.copy[pos[j]], "</span>", sep="")
    }
  }
  
  
  write( paste(tmp.copy, collapse = " "), file=myfile, append=TRUE)
  write("<br><br>", file=myfile, append=TRUE)
}

write("</ul> \n  </body> \n </html>", file=myfile, append=TRUE)



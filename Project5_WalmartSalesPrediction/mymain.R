# setwd("~/Dropbox/STAT542/Project/Walmart/")
# 
# t1 = proc.time()
# library(forecast)
# train = read.csv("train_r.csv")
# test = read.csv("test.csv")

# WMAE = matrix(0, nrow = 20, ncol = 3)

# for (t in 1:20)
# {
  #Add newtest data into train
  train<<-train
  test<<-test
  t<<-t
  if (t>1)
  {
    train = rbind(train, newtest)
  }

  #Preprocessing-train
  #Get unique dates, stores and departments as well as their numbers
  store_name = sort(unique(train$Store))
  store_num = length(store_name)
  
  dept_name = sort(unique(train$Dept))
  dept_num = length(dept_name)
  
  date_name = sort(as.Date(unique(train$Date)))
  date_num = length(date_name)
  
  #Create a dataframe with date and store name pairs
  data1 = data.frame(Date = rep(date_name, store_num), Store = rep(store_name, each = date_num))
  
  #Split the data by departments
  dept_id = split(1:nrow(train), train[, "Dept"])
  
  library(reshape)
  M = function(index)
  {
    data2 = train[index, c("Date","Store","Weekly_Sales")]
    data3 = merge(data1, data2, by = c("Date", "Store"), all.x = TRUE)
    data4 = cast(data3, Date ~ Store, value = "Weekly_Sales")
    invisible(as.data.frame(data4))
  }
  train_frame_dept = lapply(dept_id, M)
  
  id_test = paste(test$Date, test$Store, test$Dept, sep = '-')
  
  for (d in 1:dept_num)
  {
    #Fill missings for seasonal naive and product models
    train_dept1 <- train_frame_dept[[d]]
    for (i in 2:(store_num+1)){
      train_dept1[is.na(train_dept1[, i]) ,i] = mean(na.exclude(train_dept1[, i]))
      train_dept1[is.na(train_dept1[, i]) ,i] = 0
    }
    
    lastdate=train_frame_dept[[d]][nrow(train_frame_dept[[d]]),1]
    horizon=5
    
    
    n.comp=12
    train_dept=train_frame_dept[[d]]
    train_dept[is.na(train_dept)] = 0
    z = svd(train_dept[, 2:ncol(train_dept)], nu=n.comp, nv=n.comp)
    ss = diag(z$d[1:n.comp])
    train_dept[, 2:ncol(train)] = z$u %*% ss %*% t(z$v)
    
    for (s in 2:(store_num+1))
    {
      
      #Models for prediction
      #seasonal.naive
      pred <- train_dept1[nrow(train_dept1) - (52:1) + 1, s]
      pred.sales1 <- pred[1:horizon]

      #product
      pred <- train_dept1[nrow(train_dept1) - (52:1) + 1,]
      levels <- colMeans(pred[,2:ncol(pred)])
      profile <- rowMeans(pred[,2:ncol(pred)])
      overall <- mean(levels)
      pred <- matrix(profile, ncol=1) %*% matrix(levels, nrow=1)
      pred <- pred / overall
      pred.sales2 <- pred[1:horizon, s-1]
      
      
      # #fit arima model
      train.temp.sales=train_dept[,s]
      #train.temp.sales[train.temp.sales==0]<-NA
      ##############missing vlaue#################
      if(sum(is.na(train.temp.sales)) > length(train.temp.sales)/3){
        if(s==(store_num)+1){
          a=cbind(train_frame_dept[[d]][(length(train.temp.sales)-4):length(train.temp.sales),s-2],train_frame_dept[[d]][(length(train.temp.sales)-4):length(train.temp.sales),s-1])
          pred.sales3=apply(a,1,mean,na.rm=TRUE)
          pred.sales3[is.na(pred.sales3)]<-mean(pred.sales3,na.rm=TRUE)
          pred.sales3[is.na(pred.sales3)]<-0
        }else{
          a=cbind(train_frame_dept[[d]][(length(train.temp.sales)-4):length(train.temp.sales),s-1],train_frame_dept[[d]][(length(train.temp.sales)-4):length(train.temp.sales),s+1])
          pred.sales3=apply(a,1,mean,na.rm=TRUE)
          pred.sales3[is.na(pred.sales3)]<-mean(pred.sales3,na.rm=TRUE)
          pred.sales3[is.na(pred.sales3)]<-0
        }
        #####################arima#######################
      }else{
        train.temp.sales[is.na(train.temp.sales)]<- mean(train.temp.sales, na.rm = TRUE)
        ts <- ts(train.temp.sales, frequency=52)
        model <- auto.arima(ts, ic='bic')
        fc <- forecast(model, h=horizon)
        pred.sales3 <- as.numeric(fc$mean)
      }
      
      
      #Store the predicted values into the test dataset
      pred.date=seq(as.Date(lastdate),by="week", length=6) 
      pred.date=pred.date[-1]
      store_n = as.numeric(store_name[s-1])
      dept_n = as.numeric(names(train_frame_dept)[d])
      temp = data.frame(Date = as.character(pred.date), Store = rep(store_n,5), Dept = rep(dept_n,5), 
                         Weekly_Pred1 = pred.sales1, Weekly_Pred2 = pred.sales2, Weekly_Pred3 = pred.sales3)
      temp[is.na(temp)] = 0
      id_pred = paste(temp$Date, temp$Store,temp$Dept, sep = '-')
      index1 = match(id_pred, id_test)
      test[na.exclude(index1), c('Weekly_Pred1', 'Weekly_Pred2','Weekly_Pred3')] = temp[!is.na(index1),c('Weekly_Pred1', 'Weekly_Pred2', 'Weekly_Pred3')]
    }
  }
  test$Date = as.character(test$Date)
  test=test[,-1]
  test<<-test
  #newtest: sales data for this month; taking the same format as "train".
  # tmp.filename = paste('test', t, '.csv', sep=''); #Note that the names of the new test set for each month must be "xxxt.csv" 
  # newtest = read.csv(tmp.filename)
  
  #Match the predicted values in "test" with the "newtest" dataset for the current month by Date, Store, Department
  #Note that "test" contains all observations from 2010-2 to 2012-10, but it does not have "Weekly_Sales" variable.
  # eva_set = merge(newtest, test, by=c("Dept","Date","Store"),all.x = TRUE)
  # eva_set$Weekly_Pred1[is.na(eva_set$Weekly_Pred1)] = 0 
  # eva_set$Weekly_Pred2[is.na(eva_set$Weekly_Pred2)] = 0 
  # eva_set$Weekly_Pred3[is.na(eva_set$Weekly_Pred3)] = 0 
  

  #Evaluation code(function of newtest and predicted values)
  #weightss = ifelse(eva_set$IsHoliday.x, 5, 1)
#   WMAE[t,1] = 1/sum(weightss)*sum(weightss*abs(eva_set$Weekly_Sales.x - eva_set$Weekly_Pred1), na.rm=TRUE)
#   WMAE[t,2] = 1/sum(weightss)*sum(weightss*abs(eva_set$Weekly_Sales.x - eva_set$Weekly_Pred2), na.rm=TRUE)
#   WMAE[t,3] = 1/sum(weightss)*sum(weightss*abs(eva_set$Weekly_Sales.x - eva_set$Weekly_Pred3), na.rm=TRUE)
# }

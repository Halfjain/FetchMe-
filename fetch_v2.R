




rm(list = ls()) # remove all the variables from the workspace
library(forecast)
require(NTS)
library(Hmisc)
library(neuralnet)
ft<-read.csv(file="C:/Users/kbl00/Dropbox/Teaching/PM/Spring 2019/BUAL 5600/Fetchme/fetch.csv", header=TRUE, sep=",")

we<-read.csv(file="C:/Users/kbl00/Dropbox/Teaching/PM/Spring 2019/BUAL 5600/Fetchme/weather.csv", header=TRUE, sep=",")


ft$wc<-we$PRCP
ft$wc.d<-0
ft$wc.d[ft$wc>0]<-1

ft$wc.t1<-0
ft$wc.t2<-0
ft$wc.t3<-0
ft$wc.t2[ft$wc>0 & ft$wc<=0.07]<-1
ft$wc.t3[ft$wc>0.07]<-1



date<-c("M","T","W","R","F","SA","SU")
ft$date<-c(rep(date, 52), "M")
ft$time<-seq(1,365,1)
ft$time2<-ft$time*ft$time
ft$Mon<-0
ft$Mon[ft$date=="M"]<-1
ft$Tue<-0
ft$Tue[ft$date=="T"]<-1
ft$Wed<-0
ft$wed[ft$date=="W"]<-1
ft$Thu<-0
ft$Thu[ft$date=="R"]<-1
ft$Fri<-0
ft$Fri[ft$date=="F"]<-1
ft$Sat<-0
ft$Sat[ft$date=="SA"]<-1
ft$Sun<-0
ft$Sun[ft$date=="SU"]<-1
ft$logy<-log(ft$Total.Order+1)




# create lagged variabl
ft$lag.y1<-Lag(ft$Total.Order, +1)
ft$lag.y2<-Lag(ft$Total.Order, +2)
ft$lag.y3<-Lag(ft$Total.Order, +3)

ft$lag.logy<-Lag(ft$logy, +1)
ft$lag.wc.d<-Lag(ft$wc.d, +1)

# change column name
colnames(ft)[colnames(ft)=="Average.Dinner.Order"] <- "adr" 
#"Average.Dinner.Order"=old name; "adr"= new name


Total.Order<-ts(ft$Total.Order, freq=7)
plot(Total.Order, xlab="Time", ylab="Total Order",
     ylim=c(0,175), bty="l")
hist(ft$Total.Order)

log.Total.Order<-ts(ft$logy, freq=7)
ts.plot(log.Total.Order, xlab="Time", ylab="Total Order",bty="l")
hist(ft$logy)



## extract a set to train the NN
trainset <- ft[1:285, ];
trainset<-trainset[2:285,]
## select the test set; forecasting 80 observations
testset <- ft[286:365, ]





# regression with a linear trend 
fm0<-lm(Total.Order~time, data=trainset)
summary(fm0)
yhat0<-predict(fm0, new=testset)
er0 <- yhat0-testset$Total.Order 
mean(er0^2); mean(abs(er0))

plot.ts(trainset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm0), lwd=2)

plot.ts(testset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm0, new=testset), lwd=1, lty=2, col="blue")



# regression with a linear trend and weather condition
fm0<-lm(Total.Order~lag.y1+wc.d+lag.wc.d, data=trainset)
summary(fm0)
yhat0<-predict(fm0, new=testset)
er0 <- yhat0-testset$Total.Order 
mean(er0^2); mean(abs(er0))








# regression with a quadratic trend 
fm1<-lm(Total.Order~time+time2, data=trainset)
summary(fm1)
yhat1<-predict(fm1, new=testset)
er1 <- yhat1-testset$Total.Order
mean(er1^2); mean(abs(er1))

plot.ts(trainset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm1), lwd=2)

plot.ts(testset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm1, new=testset), lwd=1, lty=2, col="blue")



# regression with a lagged depdent 
fm2<-lm(Total.Order~lag.y1, data=trainset)
summary(fm2)
yhat2<-predict(fm2, new=testset)
er2 <- yhat2-testset$Total.Order
mean(er2^2); mean(abs(er2))

plot.ts(ft$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm2), lwd=2, col="red")

plot.ts(testset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm2, new=testset), lwd=1, lty=2, col="blue")



# regression with a lagged dependent + time trend + dates
fm3<-lm(Total.Order~lag.y1+time+time2+as.factor(date), data=trainset)
summary(fm3)
yhat3<-predict(fm3, new=testset)
er3 <- yhat3-testset$Total.Order
mean(er3^2); mean(abs(er3))

plot.ts(ft$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm3), lwd=2, col="blue")
lines(predict(fm2), lwd=2, col="red")

plot.ts(testset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm3, new=testset), lwd=1, lty=2, col="blue")




# regression with a lagged dependent + dates
fm4<-lm(Total.Order~lag.y1+as.factor(date), data=trainset)
summary(fm4)
yhat4<-predict(fm4, new=testset)
er4 <- yhat4-testset$Total.Order
mean(er4^2); mean(abs(er4))

plot.ts(ft$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm4), lwd=2, col="green")
lines(predict(fm3), lwd=2, col="blue")
lines(predict(fm2), lwd=2, col="red")


plot.ts(testset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm4, new=testset), lwd=1, lty=2, col="blue")



# regression with trend terms and date dummies
fm5<-lm(Total.Order~time+time2+as.factor(date), data=trainset)
summary(fm5)
yhat5<-predict(fm5, new=testset)
er5 <- yhat5-testset$Total.Order
mean(er5^2); mean(abs(er5))

plot.ts(ft$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm5), lwd=2, col="blue")


plot.ts(testset$Total.Order, xlab="Time", ylab="Total Order",
        ylim=c(0,175), bty="l")
lines(predict(fm5, new=testset), lwd=1, lty=2, col="blue")



### [Optional model]Machine Learning ###
# if we are trying to predict the outcome of continuous (or counts) using neural networks,
# we must scale our outcome variable. A standard approach is to scale the inputs to have mean 0 and a 
# variance of 1 (standardization). However, the problem is the neural networks using scaled values will obviously
# have a different intercept, smaller MSE (and MAE), than the unscaled originals if the original mean values
# were not zero. Futhermore, it is hard to interpret the estimates. Thus, I don't recommen the neural networks
# in this project. 

# for special credits, you are welcome to try the neural network and then discuss its findings. 



nn1<-neuralnet(Total.Order ~lag.y1+time+time2+as.factor(date),
               trainset, hidden = 4, lifesign = "minimal", 
                       linear.output = TRUE, threshold = 0.1)
# you can't use the "as.factor" function in neuralnet. So, pleas try the R syntax below:

nn1<-neuralnet(Total.Order ~ time+Mon+Tue+Wed+Thu+Sat+Sun,
               trainset, hidden = 4, lifesign = "minimal", 
               linear.output = TRUE, threshold = 0.1)



## plot the NN
plot(nn1, rep = "best");



## test the resulting output
temp_test <- subset(testset, select = c("time","Mon","Tue","Wed","Thu","Sat","Sun"));
forecast.results <- compute(nn1, temp_test);

results <- data.frame(actual = testset$Total.Order, prediction=forecast.results$net.result)
head(results);

nn.e<-results$actual-results$prediction;
mean(nn.e^2); mean(abs(nn.e))






nn2<-neuralnet(logy ~ lag.logy+time+Mon+Tue+Wed+Thu+Sat+Sun,
               trainset, hidden = 1, linear.output = TRUE)

## plot the NN
plot(nn2, rep = "best");



## test the resulting output
temp_test <- subset(testset, select = c("lag.logy","time","Mon","Tue","Wed","Thu","Sat","Sun"));
forecast.results <- compute(nn2, temp_test);

results <- data.frame(actual = testset$logy, prediction=forecast.results$net.result)
head(results);

nn.e2<-results$actual-results$prediction;
mean(nn.e2^2); mean(abs(nn.e2))
# please keep in mind that, these MSE and MAE are a lot lower than above models' MSE and MAE because 
# of log-transformation (scale changes)








#import the dataset
dataset=read.csv('Position_Salaries.csv')
dataset=dataset[2:3]

#Spliting dataset in testset and training set
#library(caTools)
#set.seed(123)
#split=sample.split(dataset$Purchased,SplitRatio = 0.8)
#training_set=subset(dataset,split==TRUE)
#test_set=subset(dataset,split==FALSE)

#Feature Scalling not needed in Trees

# training_set[,2:3]=scale(training_set[,2:3])
# test_set[,2:3]=scale(test_set[,2:3])


#Fitting Decision Tree Regression model to the dataset
#create regressor here !
regressor=rpart(formula= Salary ~ . ,data=dataset,
                control =rpart.control(minsplit = 1) ) #control does splits control



#Predicting a new result 
#data frame is to create specific level 
y_pred=predict(regressor,data.frame(Level=6.5)) 

#Visualising the Decision Tree Regression Model results (for higher resolution and smoother curve)
#
x_grid=seq(min(dataset$Level),max(dataset$Level),0.01)


ggplot()+
  geom_point(aes(x=dataset$Level ,y= dataset$Salary),
             color='red') +
  geom_line(aes(x=x_grid ,y= predict(regressor, newdata = data.frame(Level=x_grid))),
            color='blue') +
  ggtitle('Truth or Bluff(Regression Model)') +
  xlab('Level') +
  ylab('Salary')

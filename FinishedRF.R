install.packages("caret")       # For sampling the dataset 
install.packages("caret", dependencies = TRUE)

library(caret)

smp_size <- floor(0.80 * nrow(completedData))

train_ind <- sample(seq_len(nrow(completedData)), size = smp_size)

train <- completedData[train_ind, ]

test <- completedData[-train_ind, ]

head(train)
head(test)


set.seed(51)

train$Bot <- factor(train$Bot, ordered = TRUE)

table(train[,c('Bot', 'Following')])

#train$Bot <- Bot

rf_model<-train(Bot~.,data=train,method="rf",
                trControl=trainControl(method="cv",number=5),
                prox=TRUE,allowParallel=TRUE)
print(rf_model)

print(rf_model$finalModel)

test$Bot <- factor(test$Bot, ordered = TRUE)
#Bot <- as.factor(test$Bot)


predictions = predict(rf_model, newdata = test)
#str(test)
#str(predictions)
confusionMatrix(predictions, test$Bot)


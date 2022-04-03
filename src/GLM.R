library(MASS)

K.FOLD = 3
data <- read.csv("F:/github/Pei-Yu_project/data/Dwelling_asbestos_fixed.csv",
                 TRUE,",")

# data pre-processing
data <- na.omit(data.frame(data$Pipe.insulation, 
                   data$EPC.category,
                   data$EPC.type,
                   data$Construction.year,
                   data$Renovation.year,
                   data$Number.of.floors,
                   data$Floor.area,
                   data$Number.of.basements,
                   data$Number.of.stairwells,
                   data$Number.of.apartments,
                   data$Exhaust,
                   data$Balanced,
                   data$Balanced.with.heat.exchanger,
                   data$Exhaust.with.heat.pump,
                   data$Natural.ventilation,
                   data$Gbg))

data$data.Number.of.basements <-as.factor(data$data.Number.of.basements)
data$kfold.group <- ceiling(runif(nrow(data), min=0, max=K.FOLD))

# perform a stepwise regression to find out the best model
glm.model <- glm(data.Pipe.insulation~data.EPC.category + data.EPC.type + 
               data.Construction.year + data.Renovation.year + 
               data.Number.of.floors + data.Floor.area + data.Number.of.basements + 
               data.Number.of.stairwells + data.Number.of.apartments + 
               data.Exhaust + data.Balanced + data.Balanced.with.heat.exchanger + 
               data.Exhaust.with.heat.pump + data.Natural.ventilation + 
               data.Gbg, family=binomial, data=data)

stepfit <- stepAIC(glm.model)


# K-fold cross validation based on the model we found
percision <- c()
recall <- c()
accuracy <- c()

for (group in c(1:K.FOLD)){
  train_data <- subset(data, kfold.group != group)
  test_data <- subset(data, kfold.group == group)
  
  model <- glm(stepfit$formula, family=binomial, data=train_data)
  
  predict_proba <- predict(stepfit, newdata = test_data, type='response')
  TP <- 0
  FP <- 0
  TN <- 0
  FN <- 0
  
  for (i in c(1:length(predict_proba))){
    if (test_data$data.Pipe.insulation[i] == 1 & predict_proba[i] > 0.5){
      TP <- TP + 1
    }
    else if (test_data$data.Pipe.insulation[i] == 0 & predict_proba[i] > 0.5){
      FP <- FP + 1
    }
    else if (test_data$data.Pipe.insulation[i] == 0 & predict_proba[i] <= 0.5){
      TN <- TN + 1
    }
    else{
      FN <- FN + 1
    }
  }
  percision[group] = TP / (TP + FP)
  recall[group] = TP / (TP + FN)
  accuracy = (TP + TN) / (TP + TN + FP + FN)
}


print(paste("percision: ", mean(percision)))
print(paste("recall: ", mean(recall)))
print(paste("accuracy: ", mean(accuracy)))






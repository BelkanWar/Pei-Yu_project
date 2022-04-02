data <- read.csv("F:/github/Pei-Yu_project/data/Dwelling_asbestos_fixed.csv",
                 TRUE,",")

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

glm.model <- glm(data.Pipe.insulation~data.EPC.category + data.EPC.type + 
               data.Construction.year + data.Renovation.year + 
               data.Number.of.floors + data.Floor.area + data.Number.of.basements + 
               data.Number.of.stairwells + data.Number.of.apartments + 
               data.Exhaust + data.Balanced + data.Balanced.with.heat.exchanger + 
               data.Exhaust.with.heat.pump + data.Natural.ventilation + 
               data.Gbg, family=binomial, data=data)

stepfit <- stepAIC(glm.model)

predict_proba <- predict(stepfit, type='response')
TP <- 0
FP <- 0
TN <- 0
FN <- 0

for (i in c(1:length(predict_proba))){
  if (data$data.Pipe.insulation[i] == 1 & predict_proba[i] > 0.5){
    TP <- TP + 1
  }
  else if (data$data.Pipe.insulation[i] == 0 & predict_proba[i] > 0.5){
    FP <- FP + 1
  }
  else if (data$data.Pipe.insulation[i] == 0 & predict_proba[i] <= 0.5){
    TN <- TN + 1
  }
  else{
    FN <- FN + 1
  }
}

print('precision: ')
print(TP / (TP + FP))
print('recall: ')
print(TP / (TP + FN))


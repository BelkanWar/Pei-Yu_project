import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import torch
import math
import numpy as np
from sklearn import metrics

# parameters
EPOCH = 600
LEARNING_RATE = 0.06
target = 'Pipe insulation'
explain = ['EPC category',
           'EPC type',
           'Construction year',
           'Renovation year',
           'Number of floors',
           'Floor area',
           'Number of basements',
           'Number of stairwells',
           'Number of apartments',
           'Exhaust','Balanced',
           'Balanced with heat exchanger',
           'Exhaust with heat pump',
           'Natural ventilation',
           'Gbg']
usecols = [target] + explain

AUC = []
F1 = []


data = utils.read_data('../data/Dwelling_asbestos_fixed.csv', usecols)

for train_data, test_data in utils.data_prepare_and_split(data, 3):
    train_loader = utils.prepare_tensor(train_data, target)
    test_loader = utils.prepare_tensor(test_data, target)

    # modeling
    model = utils.Model(len(set(data['EPC category'])), len(set(data['EPC type'])))
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96, verbose=True)

    for epoch in range(1, EPOCH+1):
        for batch_data in train_loader:
            prediction = model(batch_data)
            loss = loss_func(prediction, batch_data[0].unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # scheduler.step()
            print(f"training loss: {float(loss.data)}")
    
    model.eval()

    for batch_data in test_loader:
        predict_proba = model(batch_data).detach().numpy().reshape((-1))
        predict = np.array([0 if i <0.5 else 1 for i in predict_proba])
        y_true = batch_data[0].numpy().astype(int)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, predict_proba, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if math.isnan(auc) == False:
            AUC.append(auc)
        F1.append(metrics.f1_score(y_true, predict))
        print('predict: ',predict)
        print('true:    ', y_true)
        print('')


print(AUC)
print(np.average(AUC))
print(F1)
print(np.average(F1))



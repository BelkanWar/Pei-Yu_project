import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import utils
import torch
import numpy as np

# parameters
EPOCH = 50
LEARNING_RATE = 0.05
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
           'Natural ventilation']
usecols = [target] + explain


data = utils.read_data('../data/Dwelling_asbestos_fixed.csv', usecols)
tensor_Dataset = utils.prepare_tensor(data, target)
train_loader, test_loader = utils.dataset_split(tensor_Dataset, len(data))


# modeling
model = utils.Model(len(set(data['EPC category'])), len(set(data['EPC type'])))
loss_func = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

for epoch in range(1, EPOCH+1):
    for batch_data in train_loader:
        prediction = model(batch_data)
        loss = loss_func(prediction, batch_data[0].unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"training loss: {float(loss.data)}")


import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold

def integer_encoding_dict(input):
    return {string:idx for idx, string in enumerate(list(set(input)))}

def read_data(path, usecols):
    data = pd.read_csv(path, usecols = usecols)

    data = data.dropna()
    return data

def data_prepare_and_split(data, n_fold):
    EPC_cate_to_idx_dict = integer_encoding_dict(data['EPC category'])
    EPC_type_to_idx_dict = integer_encoding_dict(data['EPC type'])

    data['EPC category'] = [EPC_cate_to_idx_dict[i] for i in data['EPC category']]
    data['EPC type'] = [EPC_type_to_idx_dict[i] for i in data['EPC type']]

    for key in list(data):
        data[key] = np.array(data[key])
    
    kf = KFold(n_splits=n_fold, shuffle=True)
    output = []

    for train_idx, test_idx in kf.split(data[key]):
        train_data, test_data = {}, {}
        for key in data:
            train_data[key] = np.array(data[key])[train_idx]
            test_data[key] = np.array(data[key])[test_idx]

        output.append([train_data, test_data])

    return output


def prepare_tensor(data, target):
    data[target] = [1. if int(i) == 0 else 0. for i in data[target]]
    tensor_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(list(data[target])),
        torch.LongTensor(list(data['EPC category'])),
        torch.LongTensor(list(data['EPC type'])),
        torch.FloatTensor(list(data['Construction year'])),
        torch.FloatTensor(list(data['Renovation year'])),
        torch.FloatTensor(list(data['Number of floors'])),
        torch.FloatTensor(list(data['Number of basements'])),
        torch.FloatTensor(list(data['Number of stairwells'])),
        torch.FloatTensor(list(data['Number of apartments'])),
        torch.FloatTensor(list(data['Exhaust'])),
        torch.FloatTensor(list(data['Balanced'])),
        torch.FloatTensor(list(data['Balanced with heat exchanger'])),
        torch.FloatTensor(list(data['Exhaust with heat pump'])),
        torch.FloatTensor(list(data['Natural ventilation'])),
        torch.FloatTensor(list(data['Gbg']))
    )

    data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=1000,  shuffle=True)
    return data_loader


class Model(torch.nn.Module):
    def __init__(self, epc_cate_count, epc_type_count):
        super(Model, self).__init__()
        epc_cate_hidden_dim = 2
        epc_type_hidden_dim = 2
        self.epc_cate_embed = torch.nn.Embedding(epc_cate_count, epc_cate_hidden_dim)
        self.epc_type_embed = torch.nn.Embedding(epc_type_count, epc_type_hidden_dim)
        self.bn0 = torch.nn.BatchNorm1d(12+epc_cate_hidden_dim+epc_type_hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.bn2 = torch.nn.BatchNorm1d(5)
        self.linear1 = torch.nn.Linear(12+epc_cate_hidden_dim+epc_type_hidden_dim,10)
        self.linear2 = torch.nn.Linear(10,5)
        self.linear3 = torch.nn.Linear(5,1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x_input):
        epc_cate_vector = self.epc_cate_embed(x_input[1])
        epc_type_vector = self.epc_type_embed(x_input[2])
        linear_vector = torch.cat((epc_cate_vector, epc_type_vector, x_input[3].unsqueeze(1),
            x_input[4].unsqueeze(1), x_input[5].unsqueeze(1), x_input[6].unsqueeze(1),
            x_input[7].unsqueeze(1), x_input[8].unsqueeze(1), x_input[9].unsqueeze(1),
            x_input[10].unsqueeze(1), x_input[11].unsqueeze(1), x_input[12].unsqueeze(1),
            x_input[13].unsqueeze(1), x_input[14].unsqueeze(1)), dim=1)
        linear_vector = F.relu(self.bn0(linear_vector))
        linear_vector = F.relu(self.bn1(self.linear1(linear_vector)))
        linear_vector = F.relu(self.bn2(self.linear2(linear_vector)))
        linear_vector = F.relu(self.linear3(linear_vector))
        linear_vector = self.sigmoid(linear_vector)

        return linear_vector

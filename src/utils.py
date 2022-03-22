import pandas as pd
import torch
import torch.nn.functional as F

def integer_encoding_dict(input):
    return {string:idx for idx, string in enumerate(list(set(input)))}

def read_data(path, usecols):
    data = pd.read_csv(path, usecols = usecols)

    data = data.dropna()
    return data

def prepare_tensor(data, target):
    EPC_cate_to_idx_dict = integer_encoding_dict(data['EPC category'])
    EPC_type_to_idx_dict = integer_encoding_dict(data['EPC type'])

    tensor_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(list(data[target])),
        torch.LongTensor([EPC_cate_to_idx_dict[i] for i in data['EPC category']]),
        torch.LongTensor([EPC_type_to_idx_dict[i] for i in data['EPC type']]),
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
        torch.FloatTensor(list(data['Natural ventilation']))
    )

    return tensor_dataset

def dataset_split(tensor_dataset, dp_count):

    train_dataset, test_dataset = torch.utils.data.random_split(tensor_dataset, 
        [dp_count-int(dp_count/8), int(dp_count/8)])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = 1000, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = 1000)

    return train_loader, test_loader


class Model(torch.nn.Module):
    def __init__(self, epc_cate_count, epc_type_count):
        super(Model, self).__init__()
        epc_cate_hidden_dim = 6
        epc_type_hidden_dim = 6
        self.epc_cate_embed = torch.nn.Embedding(epc_cate_count, epc_cate_hidden_dim)
        self.epc_type_embed = torch.nn.Embedding(epc_type_count, epc_type_hidden_dim)
        self.bn0 = torch.nn.BatchNorm1d(11+epc_cate_hidden_dim+epc_type_hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(10)
        self.bn2 = torch.nn.BatchNorm1d(5)
        self.linear1 = torch.nn.Linear(11+epc_cate_hidden_dim+epc_type_hidden_dim,10)
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
            x_input[13].unsqueeze(1)), dim=1)
        linear_vector = F.relu(self.bn0(linear_vector))
        linear_vector = F.relu(self.bn1(self.linear1(linear_vector)))
        linear_vector = F.relu(self.bn2(self.linear2(linear_vector)))
        linear_vector = F.relu(self.linear3(linear_vector))
        linear_vector = self.sigmoid(linear_vector)

        return linear_vector

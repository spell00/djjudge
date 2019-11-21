import torch
from torch import nn


class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layers = []
        self.bns = []
        in_channels = [1, 20, 40, 60, 60, 60]
        out_channels = [20, 40, 60, 60, 60, 1]
        kernel_sizes = [5, 5, 5, 5, 5, 1]
        strides = [3, 3, 3, 3, 3, 1]
        self.relu = torch.nn.ReLU()
        for ins, outs, ksize, stride in zip(in_channels, out_channels, kernel_sizes, strides):
            self.layers += [torch.nn.Conv1d(in_channels=ins, out_channels=outs, kernel_size=ksize, stride=stride)]
            self.bns += [nn.BatchNorm1d(num_features=outs)]
        self.dense1 = torch.nn.Linear(in_features=1233, out_features=1)
        self.dropout = nn.Dropout(0.2)
        self.layers = nn.ModuleList(self.layers)

    def random_init(self):
        for i in range(len(self.layers)):
            nn.init.kaiming_uniform_(self.layers[i].weight)
            nn.init.constant_(self.layers[i].bias, 0)
        nn.init.kaiming_uniform_(self.dense1.weight)
        nn.init.constant_(self.dense1.bias, 0)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.relu(x)
            # x = self.dropout(x)
            # x = self.bn1(x)
        x = x.squeeze()
        x = self.dense1(x)
        log_probs = torch.sigmoid(x)

        return log_probs

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

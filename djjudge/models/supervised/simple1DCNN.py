import torch
from torch import nn


class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layers = []
        self.bns = []
        self.lns = []
        self.pooling_layers = []
        in_channels = [1, 64, 128, 256, 512, 1024]
        out_channels = [64, 128, 256, 512, 1024, 1]
        kernel_sizes = [5, 5, 5, 5, 5, 1]
        strides = [3, 3, 3, 3, 3, 1]
        self.pooling = [0, 0, 0, 0, 0, 0]
        self.relu = torch.nn.ReLU()
        i = 0
        for ins, outs, ksize, stride in zip(in_channels, out_channels, kernel_sizes, strides):
            self.layers += [torch.nn.Conv1d(in_channels=ins, out_channels=outs, kernel_size=ksize, stride=stride)]
            if self.pooling[i] == 1:
                self.pooling_layers += [torch.nn.AdaptiveAvgPool1d(output_size=2)]
            else:
                self.pooling_layers += [None]
            self.bns += [nn.BatchNorm1d(num_features=outs).cuda()]
            # self.lns += [nn.LayerNorm(normalized_shape=None).cuda()]
            i += 1
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
            if self.pooling[i] == 1:
                x = self.pooling_layers[i](x)
            if i == len(self.layers) - 1:
                x = self.bns[i](x)
        x = x.squeeze()
        x = self.dense1(x)
        x = torch.sigmoid(x)

        return x

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ConvResnet(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv1d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)
        self.linear1 = torch.nn.Linear(in_features=256, out_features=32)
        self.linear2 = torch.nn.Linear(in_features=32, out_features=1)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.prelu = torch.nn.PReLU()
        self.hardtanh = nn.Hardtanh(min_val=-0.5, max_val=1.5)
        self.dropout = nn.Dropout()
        #self.dropout2d = nn.Dropout2d()

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input):
        x = self.blocks(input)
        x = self.dropout(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.linear1(x.view(-1, 256))
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        # x = self.hardtanh(x)

        # The output will be in the interval [-0.5, 1.5], even though the range of the scores are in [0.0, 1.0].
        # The number 1.5 is completely arbitrary.
        return x

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

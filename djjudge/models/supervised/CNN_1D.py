import torch
#from ..utils.stochastic import GaussianSample
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"
import torch

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        epsilon = epsilon.to(device)

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # y = x.T * beta + std * epsilon
        # mu is x.T * beta
        # y = mu _ std * epsilon
        y = (mu).addcmul(std, epsilon)
        assert y.shape[1] == 1
        return y


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features).to(device)
        self.log_var = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var

    def mle(self, x):
        return self.mu(x)


class Simple1DCNN(torch.nn.Module):
    def __init__(self,
                 activation,
                 is_bns,
                 is_dropouts,
                 final_activation=None,
                 drop_val=0.5,
                 is_bayesian=False,
                 random_node="output"
                 ):
        super(Simple1DCNN, self).__init__()
        if is_bayesian:
            if random_node == "output":
                self.GaussianSample = GaussianSample(1, 1)
            elif(random_node == "last"):
                self.GaussianSample = GaussianSample(1233, 1233)
        self.is_bayesian = is_bayesian

        self.activation = activation.to(device)
        self.is_bns = is_bns
        self.is_dropouts = is_dropouts
        self.final_activation = final_activation
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
            self.layers += [torch.nn.Conv1d(in_channels=ins, out_channels=outs, kernel_size=ksize, stride=stride).to(device)]
            if self.pooling[i] == 1:
                self.pooling_layers += [torch.nn.AdaptiveAvgPool1d(output_size=2).to(device)]
            else:
                self.pooling_layers += [None]
            self.bns += [nn.BatchNorm1d(num_features=outs).to(device)]
            # self.lns += [nn.LayerNorm(normalized_shape=None).to(device)]
            i += 1
        self.dense1 = torch.nn.Linear(in_features=1233, out_features=1).to(device)
        self.dropout = nn.Dropout(drop_val)
        self.layers = nn.ModuleList(self.layers)

    def random_init(self, init_method):
        for i in range(len(self.layers)):
            init_method(self.layers[i].weight)
            nn.init.constant_(self.layers[i].bias, 0)
        init_method(self.dense1.weight)
        nn.init.constant_(self.dense1.bias, 0)

    def forward(self, x):
        for i in range(len(self.layers)):
            #if i == len(self.layers) - 1:
            #    x = self.bns[i-1](x)
            x = self.dropout(x)
            x = self.layers[i](x)
            x = self.activation(x)
            if self.pooling[i] == 1:
                x = self.pooling_layers[i](x)
        x = x.squeeze()
        x = self.dense1(x)
        if self.is_bayesian:
            x, _, _ = self.GaussianSample.float()(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        x = self.final_activation(x)

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
    def __init__(self,
                 in_channel,
                 channel,
                 n_res_block,
                 n_res_channel,
                 stride,
                 activation,
                 dense_layers_sizes,
                 is_bns,
                 is_dropouts,
                 final_activation=None,
                 drop_val=0.5,
                 is_bayesian=False,
                 random_node="output"
                 ):
        super().__init__()
        self.is_bayesian = is_bayesian
        if is_bayesian:
            if random_node == "output":
                self.GaussianSample = GaussianSample(1, 1)
            elif(random_node == "last"):
                self.GaussianSample = GaussianSample(dense_layers_sizes[-2], dense_layers_sizes[-2])
        if stride == 4:
            blocks = [
                nn.Conv1d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
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
        self.is_dropouts = is_dropouts
        self.dropout = [[] for _ in dense_layers_sizes]
        self.bns = [[] for _ in dense_layers_sizes]
        self.linears = [[] for _ in dense_layers_sizes]
        self.bn0 = torch.nn.BatchNorm1d(dense_layers_sizes[0])
        self.is_bns = is_bns
        self.blocks = nn.Sequential(*blocks)
        for i in range(len(dense_layers_sizes)-1):
            self.linears[i] = torch.nn.Linear(in_features=dense_layers_sizes[i], out_features=dense_layers_sizes[i+1]).to(device)
            if self.is_bns[i] == 1:
                self.bns[i] = torch.nn.BatchNorm1d(dense_layers_sizes[i]).to(device)
            else:
                self.bns[i] = None
            if self.is_dropouts[i] == 1:
                self.dropout[i] = nn.Dropout(drop_val).to(device)
            else:
                self.dropout[i] = None

        self.activation = activation
        self.final_activation = final_activation

    def random_init(self, init_method=nn.init.xavier_normal_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, random_node):
        x = self.blocks(input)
        x = x.view(-1, 256)
        if self.is_bns[0]:
            x = self.bns[0](x)
        x = self.activation(x)
        for i, (dense, bn, is_bn, is_drop) in enumerate(zip(self.linears, self.bns, self.is_bns, self.is_dropouts)):
            if is_drop:
                x = self.dropout[i](x)
            # TODO linear layers are not turning to float16
            if random_node == "last" and i == len(self.bns)-2:
                x, _, _ = self.GaussianSample.float()(x)

            x = dense(x.float())
            if i < len(self.bns)-2:
                if self.is_bns[i + 1]:
                    x = self.bns[i + 1](x)
                x = self.activation(x)

        if self.is_bayesian:
            if self.final_activation is not None:
                x = self.final_activation(x)
            # TODO GaussianSample turning to float16 (half), but x is float32 (float)
            if random_node == "output":
                y, _, _ = self.GaussianSample.float()(x)
        else:
            if self.final_activation is not None:
                y = self.final_activation(x)
        return y

    def mle_forward(self, input, random_node):
        x = self.blocks(input)
        x = x.view(-1, 256)
        if self.is_bns[0]:
            x = self.bns[0](x)
        for i, (dense, bn, is_bn, is_drop) in enumerate(zip(self.linears, self.bns, self.is_bns, self.is_dropouts)):
            # linear layers are not turning to float16
            if is_drop:
                x = self.dropout[i](x)
            if random_node == "last" and i == len(self.bns)-2:
                x, _, _ = self.GaussianSample.float()(x)
            x = dense(x.float())
            if i < len(self.bns)-2:
                if self.is_bns[i + 1]:
                    x = self.bns[i + 1](x)
                x = self.activation(x)

        assert self.is_bayesian
        # GaussianSample turning to float16 (half), but x is float32 (float)
        # y = self.GaussianSample.float().mle(x)
        if random_node == "output":
            y, _, _ = self.GaussianSample.float()(x)
        if self.final_activation is not None:
            y = self.final_activation(y)
        return y

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

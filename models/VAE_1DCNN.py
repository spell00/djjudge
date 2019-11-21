import torch
from utils.stochastic import GaussianSample
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from utils.distributions import log_gaussian, log_standard_gaussian
from flow import NormalizingFlows, linIAF, PlanarNormalizingFlow
from utils.masked_layer import GatedConv1d, GatedConvTranspose1d

class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparametrize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparametrize(mu, log_var), mu, log_var


class Autoencoder1DCNN(torch.nn.Module):
    def __init__(self, z_dim=100, flow_type="nf", n_flows=2, gated=True):
        super(Autoencoder1DCNN, self).__init__()
        self.conv_layers = []
        self.deconv_layers = []
        self.bns = []
        in_channels = [1, 20, 40, 60, 60, 60]
        out_channels = [20, 40, 60, 60, 60, 1]
        kernel_sizes = [5, 5, 5, 5, 5, 1]
        strides = [3, 3, 3, 3, 3, 1]
        self.GaussianSample = GaussianSample(z_dim, z_dim)
        self.relu = torch.nn.ReLU()
        for ins, outs, ksize, stride in zip(in_channels, out_channels, kernel_sizes, strides):
            if not gated:
                self.conv_layers += [torch.nn.Conv1d(in_channels=ins, out_channels=outs, kernel_size=ksize,
                                                     stride=stride)]
            else:
                self.conv_layers += [GatedConv1d(in_channels=ins, out_channels=outs, kernel_size=ksize, stride=stride,
                                                 padding=0)]

            self.bns += [nn.BatchNorm1d(num_features=outs)]

        for ins, outs, ksize, stride in zip(reversed(out_channels), reversed(in_channels),
                                            reversed(kernel_sizes), reversed(strides)):
            if not gated:
                self.deconv_layers += [torch.nn.ConvTranspose1d(in_channels=ins, out_channels=outs,
                                                            kernel_size=ksize, stride=stride)]
            else:
                self.deconv_layers += [GatedConvTranspose1d(in_channels=ins, out_channels=outs, kernel_size=ksize,
                                                            stride=stride, padding=0)]

            self.bns += [nn.BatchNorm1d(num_features=outs)]

        self.dense1 = torch.nn.Linear(in_features=1233, out_features=z_dim)
        self.dense2 = torch.nn.Linear(in_features=z_dim, out_features=1233)
        self.dropout = nn.Dropout(0.2)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = PlanarNormalizingFlow(in_features=z_dim)

    def random_init(self):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, mu, log_var):
        if self.flow_type == "nf" and self.n_flows > 0:
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        else:
            qz = log_gaussian(z, mu, log_var)
        pz = log_standard_gaussian(z)

        kl = qz - pz

        return kl

    def encoder(self, x):
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            x = self.relu(x)
            # x = self.dropout(x)
            # x = self.bn1(x)
        x = x.squeeze()
        z = self.dense1(x)
        z = self.relu(z)

        return z

    def decoder(self, z):
        x = self.dense2(z).unsqueeze(1)
        for i in range(len(self.deconv_layers)):
            x = self.deconv_layers[i](x)
            if i < len(self.deconv_layers) - 1:
                x = self.relu(x)
            # x = self.dropout(x)
            # x = self.bn1(x)
        # TODO there is 139 / 300000 loss at the end, close enough?

        return x

    def forward(self, x):

        x = self.encoder(x)
        z, mu, log_var = self.GaussianSample(x)

        # Kullback-Leibler Divergence
        kl = self._kld(z, mu, log_var)

        x = self.decoder(z)
        log_probs = torch.tanh(x)
        return log_probs, kl

    def sample(self, z, y=None):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

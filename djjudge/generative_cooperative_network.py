import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from djjudge.models.supervised.CNN_1D import *
from djjudge.data_preparation.load_wavs_as_tensor import Wave2tensor
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
from djjudge.utils.utils import create_missing_folders
from matplotlib import pyplot as plt
from matplotlib import pylab
import wave
from djjudge.utils.CycleAnnealScheduler import CycleScheduler

checkpoint_path="classif_ckpt/cnn_corr_bayesian_v3_convresnet_kaiming_uniform__PReLU(num_parameters=1)_None_[1, 1]_[1, 1]_[128, 1]_last"

# ##### DATA: Target data and generator input data

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

# ##### MODELS: Generator model and discriminator model

def performance_per_score(predicted_values, results_path, filename="scores_performance"):
    create_missing_folders(results_path + "/plots/")
    fig2, ax21 = plt.subplots()
    predicted_values = np.array(predicted_values)

    plt.plot(predicted_values)

    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()
    del predicted_values, results_path


if __name__ == "__main__":
    n_steps = 10000000
    minibatch_size = 6
    g_input_size = 128
    lr = 1e-4
    judge = ConvResnet(in_channel=1,
                       channel=256,
                       n_res_block=4,
                       n_res_channel=256,
                       stride=4,
                       dense_layers_sizes=[256, 128, 1],
                       is_bns=[1, 1],
                       is_dropouts=[1, 1],
                       activation=nn.PReLU(),
                       final_activation=None,
                       drop_val=0.5,
                       is_bayesian=True,
                       random_node="last"
                       ).to(device)

    generator = DeconvResnet(in_channel=1,
                       channel=256,
                       n_res_block=4,
                       n_res_channel=256,
                       stride=4,
                       dense_layers_sizes=[128, 256],
                       is_bns=[1, 1],
                       is_dropouts=[1, 1],
                       activation=nn.PReLU(),
                       final_activation=torch.sigmoid_,
                       drop_val=0.5,
                       is_bayesian=False,
                       random_node="last"
                       ).to(device)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    model_for_loading = checkpoint_dict['model']
    judge.load_state_dict(model_for_loading)
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    criterion = nn.MSELoss()
    judge.to(device)
    gi_sampler = get_generator_input_sampler()
    optimizer = optim.AdamW(generator.parameters(), lr=lr, amsgrad=True)
    trace = []
    local_trace = []
    lr_schedule = CycleScheduler(optimizer, lr, n_iter=n_steps)
    for i in range(n_steps):
        generator.zero_grad()
        gen_input = Variable(gi_sampler(minibatch_size, g_input_size)).cuda()
        compositions = generator(gen_input)
        scores = judge(compositions, random_node='last')[0].cuda()
        g_error = criterion(scores.view(-1), torch.ones_like(scores).cuda().view(-1))
        g_error.backward()
        optimizer.step()  # Only optimizes G's parameters
        ge = g_error.item()
        local_trace += [ge]
        lr_schedule.step()
        if i % 100 == 0:
            print(i, np.mean(local_trace))
            trace += [np.mean(local_trace)]
            local_trace = []
            write('audio/GCN/' + str(i) + '.wav', rate=22050,
                  data=compositions.detach().cpu().numpy()[0].reshape([1, 1, -1]))
            performance_per_score(trace, "figures", filename="GCN_scores_trace")
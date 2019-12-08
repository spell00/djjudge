import torch

from sim.djjudge.train_cnn import train
from torch import nn
# from djjudge.models.supervised.CNN_1D import *


# DONT EXEC FROM HERE, exec from main !
if __name__ == "__main__":
    training_folders = [
        "/home/simon/Desktop/MIR/genres/blues/wav",
        "/home/simon/Desktop/MIR/genres/classical/wav",
        "/home/simon/Desktop/MIR/genres/country/wav",
        "/home/simon/Desktop/MIR/genres/disco/wav",
        "/home/simon/Desktop/MIR/genres/hiphop/wav",
        "/home/simon/Desktop/MIR/genres/jazz/wav",
        "/home/simon/Desktop/MIR/genres/metal/wav",
        "/home/simon/Desktop/MIR/genres/pop/wav",
        "/home/simon/Desktop/MIR/genres/reggae/wav",
        "/home/simon/Desktop/MIR/genres/rock/wav",
    ]
    scores = [
        "/home/simon/Desktop/MIR/genres/blues/scores.csv",
        "/home/simon/Desktop/MIR/genres/classical/scores.csv",
        "/home/simon/Desktop/MIR/genres/country/scores.csv",
        "/home/simon/Desktop/MIR/genres/disco/scores.csv",
        "/home/simon/Desktop/MIR/genres/hiphop/scores.csv",
        "/home/simon/Desktop/MIR/genres/jazz/scores.csv",
        "/home/simon/Desktop/MIR/genres/metal/scores.csv",
        "/home/simon/Desktop/MIR/genres/pop/scores.csv",
        "/home/simon/Desktop/MIR/genres/reggae/scores.csv",
        "/home/simon/Desktop/MIR/genres/rock/scores.csv",
    ]
    output_directory = "."

    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(training_folders,
          scores,
          output_directory,
          batch_size=6,
          epochs=1000,
          epochs_per_checkpoint=1,
          learning_rate=1e-3,
          fp16_run=False,
          checkpoint_name="classif_ckpt/cnn_corr_bayesian_v2",
          is_bns=[1, 1],
          is_dropouts=[1, 1],
          activation=nn.PReLU(),
          final_activation=None,
          noise=0.02,
          loss_type=torch.nn.MSELoss,
          factor=1.1,
          flat_extrems=False,
          model_type="convresnet",
          is_bayesian=True,
          init_method=nn.init.kaiming_normal_,

          )

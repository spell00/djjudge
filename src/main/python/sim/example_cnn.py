import torch

from sim.djjudge.train_cnn import train
from torch import nn
# from djjudge.models.supervised.CNN_1D import *
from fbs_runtime.application_context.PyQt5 import ApplicationContext


def main(ctx):
    appctx = ctx
    training_folders = [
        appctx.get_resource("MIR/genres/blues/wav"),
        appctx.get_resource("MIR/genres/classical/wav"),
        appctx.get_resource("MIR/genres/country/wav"),
        appctx.get_resource("MIR/genres/disco/wav"),
        appctx.get_resource("MIR/genres/hiphop/wav"),
        appctx.get_resource("MIR/genres/jazz/wav"),
        appctx.get_resource("MIR/genres/metal/wav"),
        appctx.get_resource("MIR/genres/pop/wav"),
        appctx.get_resource("MIR/genres/reggae/wav"),
        appctx.get_resource("MIR/genres/rock/wav")
    ]
    scores = [
        appctx.get_resource("MIR/genres/blues/scores.csv"),
        appctx.get_resource("MIR/genres/classical/scores.csv"),
        appctx.get_resource("MIR/genres/country/scores.csv"),
        appctx.get_resource("MIR/genres/disco/scores.csv"),
        appctx.get_resource("MIR/genres/hiphop/scores.csv"),
        appctx.get_resource("MIR/genres/jazz/scores.csv"),
        appctx.get_resource("MIR/genres/metal/scores.csv"),
        appctx.get_resource("MIR/genres/pop/scores.csv"),
        appctx.get_resource("MIR/genres/reggae/scores.csv"),
        appctx.get_resource("MIR/genres/rock/scores.csv")
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


# DONT EXEC FROM HERE, exec from main !
if __name__ == "__main__":
    main(ApplicationContext())
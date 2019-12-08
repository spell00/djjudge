import torch

from sim.djjudge.models.supervised.CNN_1D import *
from sim.djjudge.train_cnn import train


# DONT EXEC FROM HERE, exec from main !
if __name__ == "__main__":
    training_folders = [
        "C:/Users/simon/Documents/MIR/genres/blues/wav",
        "C:/Users/simon/Documents/MIR/genres/classical/wav",
        "C:/Users/simon/Documents/MIR/genres/country/wav",
        "C:/Users/simon/Documents/MIR/genres/disco/wav",
        "C:/Users/simon/Documents/MIR/genres/hiphop/wav",
        "C:/Users/simon/Documents/MIR/genres/jazz/wav",
        "C:/Users/simon/Documents/MIR/genres/metal/wav",
        "C:/Users/simon/Documents/MIR/genres/pop/wav",
        "C:/Users/simon/Documents/MIR/genres/reggae/wav",
        "C:/Users/simon/Documents/MIR/genres/rock/wav",
    ]
    scores = [
        "C:/Users/simon/Documents/MIR/genres/blues/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/classical/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/country/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/disco/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/hiphop/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/jazz/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/metal/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/pop/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/reggae/scores.csv",
        "C:/Users/simon/Documents/MIR/genres/rock/scores.csv",
    ]
    output_directory = "C:/Users/simon/djjudge/"

    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(training_folders,
          scores,
          output_directory,
          batch_size=8,
          epochs=100000,
          epochs_per_checkpoint=1,
          learning_rate=1e-3,
          fp16_run=False,
          checkpoint_name=None)

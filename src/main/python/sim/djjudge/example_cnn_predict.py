import torch

from sim.djjudge.models.supervised.CNN_1D import *
from sim.djjudge.prediction import predict


# DONT EXEC FROM HERE, exec from main !
if __name__ == "__main__":
    spotify = [
        "/home/simon/Desktop/spotify/potpourri",
    ]
    output_directory = "C:/Users/simon/djjudge/"

    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    predict(predict_folders=spotify,
            batch_size=1,
            fp16_run=False,
            final_activation=torch.nn.Hardtanh(min_val=-0.4, max_val=1.4),
            activation=torch.nn.PReLU(),
            checkpoint_name="classif_ckpt/cnn_corr_convresnet_kaiming_uniform__PReLU(num_parameters=1)_Hardtanh(min_val=-1.4, max_val=1.4)_[0, 0]_[1, 1]")

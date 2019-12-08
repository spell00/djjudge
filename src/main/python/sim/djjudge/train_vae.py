import torch.nn as nn
import argparse
import os
import torch
import numpy as np

from tensorboardX import SummaryWriter
from scipy.io.wavfile import write
from sim.djjudge.utils.CycleAnnealScheduler import CycleScheduler
from torch.utils.data import DataLoader
from sim.djjudge.data_preparation.load_wavs_as_tensor import Wave2tensor
from sim.djjudge.models.unsupervised.VAE_1DCNN import Autoencoder1DCNN
from sim.djjudge.utils.plot_performance import plot_performance
from sim.djjudge.utils.plot_waves import plot_waves
from sim.djjudge.utils.utils import create_missing_folders


training_folders = [
    "C:/Users/simon/Documents/MIR/genres/hiphop/wav/",
    "C:/Users/simon/Documents/MIR/genres/jazz/wav/",
    "C:/Users/simon/Documents/MIR/genres/rock/wav/",
    "C:/Users/simon/Documents/MIR/genres/metal/wav/",
    "C:/Users/simon/Documents/MIR/genres/classical/wav/",
    "C:/Users/simon/Documents/MIR/genres/reggae/wav/",
    "C:/Users/simon/Documents/MIR/genres/pop/wav/",
    "C:/Users/simon/Documents/MIR/genres/disco/wav/",
    "C:/Users/simon/Documents/MIR/genres/country/wav/",
    "C:/Users/simon/Documents/MIR/genres/blues/wav/",
    "C:/Users/simon/Documents/spotify/potpourri"
]
output_directory = "C:/Users/simon/djjudge/checkpoints/"


def load_checkpoint(checkpoint_path, model, optimizer, z_dim, gated, in_channels, out_channels, kernel_sizes,
                    strides, dilatations, name="vae_1dcnn"):
    # if checkpoint_path
    losses_recon = {
        "train": [],
        "valid": [],
    }
    kl_divs = {
        "train": [],
        "valid": [],
    }
    losses = {
        "train": [],
        "valid": [],
    }

    if name not in os.listdir(checkpoint_path):
        print("Creating checkpoint...")
        save_checkpoint(model, optimizer, learning_rate=None, epoch=0, checkpoint_path=checkpoint_path,
                        z_dim=z_dim, gated=gated, losses=losses, kl_divs=kl_divs, losses_recon=losses_recon,
                        in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_sizes, strides=strides,
                        dilatations=dilatations, name=name)
    checkpoint_dict = torch.load(checkpoint_path + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    losses_recon = checkpoint_dict['losses_recon']
    kl_divs = checkpoint_dict['kl_divs']
    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(
        checkpoint_path, epoch))
    return model, optimizer, epoch, losses, kl_divs, losses_recon


def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path, z_dim, gated, losses,
                    kl_divs, losses_recon, in_channels, out_channels, kernel_sizes, strides, dilatations,
                    name="vae_1dcnn"):
    model_for_saving = Autoencoder1DCNN(flow_type="nf", n_flows=10, z_dim=z_dim, gated=gated,
                                        in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_sizes,
                                        strides=strides, dilatations=dilatations).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'losses': losses,
                'kl_divs': kl_divs,
                'losses_recon': losses_recon,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + name)


def generate(model):
    pass


def train(name,
          z_dim,
          in_channels,
          out_channels,
          kernel_sizes,
          strides,
          dilatations,
          batch_size=16,
          epochs=100,
          learning_rate=1e-3,
          fp16_run=False,
          checkpoint_path=None,
          epochs_per_checkpoint=1,
          gated=True,
          n_flows=10):
    create_missing_folders('audio/' + name)
    torch.manual_seed(42)
    model = Autoencoder1DCNN(z_dim, in_channels, out_channels, kernel_sizes, strides, dilatations, flow_type="nf",
                             n_flows=n_flows, gated=gated).cuda()
    model.random_init()
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, amsgrad=False)
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # Load checkpoint if one exists
    epoch = 0
    if checkpoint_path is not None:
        model, optimizer, epoch, losses, kl_divs, losses_recon = load_checkpoint(checkpoint_path, model, optimizer,
                                                                                 z_dim, gated=gated,
                                                                                 in_channels=in_channels,
                                                                                 out_channels=out_channels,
                                                                                 kernel_sizes=kernel_sizes,
                                                                                 strides=strides,
                                                                                 dilatations=dilatations,
                                                                                 name=name)

    train_set = Wave2tensor(training_folders, scores_files=None, segment_length=30000)

    valid_set = Wave2tensor(training_folders, scores_files=None, segment_length=30000, valid=True)

    train_loader = DataLoader(train_set, num_workers=0,
                              shuffle=True,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    valid_loader = DataLoader(valid_set, num_workers=0,
                              shuffle=True,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    logger = SummaryWriter('logs')
    epoch_offset = max(1, epoch)
    lr_schedule = CycleScheduler(optimizer, learning_rate, n_iter=epochs * len(train_loader))
    losses = {
        "train": [],
        "valid": [],
    }
    kl_divs = {
        "train": [],
        "valid": [],
    }
    losses_recon = {
        "train": [],
        "valid": [],
    }
    running_abs_error = {
        "train": [],
        "valid": [],
    }
    shapes = {
        "train": len(train_set),
        "valid": len(valid_set),
    }
    for epoch in range(epoch_offset, epochs):
        model.train()
        train_losses = []
        train_abs_error = []
        train_kld = []
        train_recons = []
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            audio, targets, sampling_rate = batch
            audio = torch.autograd.Variable(audio).cuda()
            reconstruct, kl = model(audio.unsqueeze(1))
            reconstruct = reconstruct.squeeze()
            if reconstruct.shape[1] < audio.shape[1]:
                audio = audio[:, :reconstruct.shape[1]]
            elif reconstruct.shape[1] > audio.shape[1]:
                reconstruct = reconstruct[:, :audio.shape[1]]

            loss_recon = criterion(reconstruct, audio).sum()
            kl_div = torch.mean(kl)
            loss = loss_recon + kl_div
            train_losses += [loss.item()]
            train_kld += [kl_div.item()]
            train_recons += [loss_recon.item()]
            train_abs_error += [float(torch.mean(torch.abs_(reconstruct - audio.cuda())).item())]

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                del scaled_loss
            else:
                loss.backward()
            optimizer.step()
            lr_schedule.step()
            logger.add_scalar('training_loss', loss, i + len(train_loader) * epoch)
            del loss, kl
        losses["train"] += [np.mean(train_losses)]
        kl_divs["train"] += [np.mean(train_kld)]
        losses_recon["train"] += [np.mean(train_recons)]
        running_abs_error["train"] += [np.mean(train_abs_error)]

        if epoch % epochs_per_checkpoint == 0:
            print("Epoch: {}:\tTrain Loss: {:.5f} , kld: {:.3f} , recon: {:.3f}".format(epoch,
                                                                                        np.mean(losses["train"]),
                                                                                        np.mean(kl_divs["train"]),
                                                                                        np.mean(losses_recon["train"])))
            write('audio/' + name + '/train_original_example.wav', rate=int(sampling_rate[0]),
                  data=audio[0].detach().cpu().numpy())
            write('audio/' + name + '/train_reconstruction_example.wav', rate=int(sampling_rate[0]),
                  data=reconstruct[0].detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/train/" + name + '/',
                       filename="wave_train" + str(epoch))
        model.eval()
        valid_losses = []
        valid_kld = []
        valid_recons = []
        valid_abs_error = []
        for i, batch in enumerate(valid_loader):
            audio, targets, sampling_rate = batch
            audio = audio.cuda()
            reconstruct, kl = model(audio.unsqueeze(1))
            reconstruct = reconstruct.squeeze()
            if reconstruct.shape[1] < audio.shape[1]:
                audio = audio[:, :reconstruct.shape[1]]
            elif reconstruct.shape[1] > audio.shape[1]:
                reconstruct = reconstruct[:, :audio.shape[1]]
            loss_recon = criterion(reconstruct, audio.cuda()[:, :reconstruct.shape[1]]).sum()
            kl_div = torch.mean(kl)
            loss = loss_recon + kl_div
            valid_losses += [loss.item()]
            valid_kld += [kl_div.item()]
            valid_recons += [loss_recon.item()]
            valid_abs_error += [float(torch.mean(torch.abs_(reconstruct - audio.cuda())).item())]
            logger.add_scalar('training loss', np.log2(loss.item()), i + len(train_loader) * epoch)
        losses["valid"] += [np.mean(valid_losses)]
        kl_divs["valid"] += [np.mean(valid_kld)]
        losses_recon["valid"] += [np.mean(valid_recons)]
        running_abs_error["valid"] += [np.mean(valid_abs_error)]

        if epoch % epochs_per_checkpoint == 0:
            save_checkpoint(model, optimizer, learning_rate, epoch, output_directory, z_dim=z_dim, gated=gated,
                            name=name, losses=losses, kl_divs=kl_divs, losses_recon=losses_recon,
                            in_channels=in_channels, out_channels=out_channels, kernel_sizes=kernel_sizes,
                            strides=strides, dilatations=dilatations
                            )
            print("Epoch: {}:\tValid Loss: {:.5f} , kld: {:.3f} , recon: {:.3f}".format(epoch,
                                                                                        np.mean(losses["valid"]),
                                                                                        np.mean(kl_divs["valid"]),
                                                                                        np.mean(losses_recon["valid"])))
            write('audio/' + name + '/valid_reconstruction_example.wav', rate=int(sampling_rate[0]),
                  data=reconstruct[0].detach().cpu().numpy())
            write('audio/' + name + '/valid_original_example.wav', rate=int(sampling_rate[0]),
                  data=audio[0].detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/valid/" + name + '/',
                       filename="wave_valid" + str(epoch))
            generated_sound = model.sample(torch.rand(1, z_dim).cuda())
            write('audio/' + name + '/generated_sample_example.wav', rate=int(sampling_rate[0]),
                  data=generated_sound.detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/valid/" + name + '/',
                       filename="wave_valid" + str(epoch))
        plot_performance(loss_total=losses, losses_recon=losses_recon, kl_divs=kl_divs, shapes=shapes,
                         results_path="figures",
                         filename="training_loss_trace_" + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    in_channels = [1, 64, 64, 64, 128, 128, 128]
    out_channels = [64, 64, 64, 128, 128, 128, 1]
    kernel_sizes = [3, 3, 3, 3, 3, 3, 1]
    strides = [2, 2, 2, 2, 2, 2, 1, 1]
    dilatations = [1, 2, 4, 8, 16, 32, 64]
    paddings = [1, 2, 4, 8, 16, 32, 64]
    dilatations_deconv = [1, 2, 4, 8, 16, 32, 64]
    z_dim = 100
    n_flows = 10
    n_epochs = 100000
    bs = 16
    epochs_per_checkpoint = 1
    gated = True
    checkpoint_path = "checkpoints/"
    train("vae_1dcnn_nogated3" + str(n_flows) + "_" + str(z_dim),
          z_dim,
          in_channels,
          out_channels,
          kernel_sizes,
          strides,
          dilatations,
          batch_size=bs,
          epochs=n_epochs,
          checkpoint_path=checkpoint_path,
          epochs_per_checkpoint=epochs_per_checkpoint,
          gated=gated,
          n_flows=n_flows,
          fp16_run=False)

from CycleAnnealScheduler import CycleScheduler
from torch.utils.data import DataLoader
from load_wavs_as_tensor import Wave2tensor
import torch.nn as nn
import argparse
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from models.VAE_1DCNN import Autoencoder1DCNN
from utils.plot_performance import plot_performance
from utils.plot_waves import plot_waves

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
    "C:/Users/simon/Documents/MIR/genres/blues/wav/",
    "C:/Users/simon/Documents/spotify/potpourri"
]
output_directory = "C:/Users/simon/djjudge/checkpoints/"
from scipy.io.wavfile import write


def load_checkpoint(checkpoint_path, model, optimizer, z_dim, gated, name="vae_1dcnn"):
    # if checkpoint_path
    if name not in os.listdir(checkpoint_path):
        print("Creating checkpoint...")
        save_checkpoint(model, optimizer, learning_rate=None, epoch=0, checkpoint_path=checkpoint_path,
                        z_dim=z_dim, gated=gated, train_losses=[], valid_losses=[], name="vae_1dcnn")
    checkpoint_dict = torch.load(checkpoint_path + name, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    train_losses = checkpoint_dict['train_losses']
    valid_losses = checkpoint_dict['valid_losses']
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, iteration, train_losses, valid_losses


def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path, z_dim, gated, train_losses,
                    valid_losses, name="vae_1dcnn"):
    model_for_saving = Autoencoder1DCNN(flow_type="nf", n_flows=10, z_dim=z_dim, gated=gated).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'iteration': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + name)


def generate(model):
    pass


def train(name, z_dim=100,
          batch_size=16,
          epochs=100,
          learning_rate=1e-4,
          fp16_run=False,
          checkpoint_path=None,
          epochs_per_checkpoint=1,
          gated=True):
    torch.manual_seed(42)
    model = Autoencoder1DCNN(flow_type="nf", n_flows=10, z_dim=z_dim, gated=gated).cuda()
    model.random_init()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, amsgrad=True)
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # Load checkpoint if one exists
    epoch = 0
    if checkpoint_path is not None:
        model, optimizer, epoch, train_losses, valid_losses = load_checkpoint(checkpoint_path, model, optimizer,
                                                                              z_dim, gated=gated, name=name)

    train_set = Wave2tensor(training_folders, scores_files=None, segment_length=300000)

    valid_set = Wave2tensor(training_folders, scores_files=None, segment_length=300000, valid=True)

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
    running_abs_error = {
        "train": [],
        "valid": [],
    }
    shapes = {
        "train": len(train_set),
        "valid": len(valid_set),
    }
    losses["train"] = train_losses
    losses["valid"] = valid_losses
    for epoch in range(epoch_offset, epochs):
        model.train()
        train_losses = []
        train_abs_error = []
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            audio, targets, sampling_rate = batch
            audio = torch.autograd.Variable(audio).cuda()
            reconstruct, kl = model(audio.unsqueeze(1))
            reconstruct = reconstruct.squeeze()
            audio = audio[:, :reconstruct.shape[1]]
            loss = criterion(reconstruct, audio) + torch.mean(kl).item()
            train_losses += [loss.item()]
            train_abs_error += [float(torch.mean(torch.abs_(reconstruct - audio.cuda())).item())]

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                del scaled_loss
            else:
                loss.backward()
            optimizer.step()
            lr_schedule.step()
            logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
            del loss, kl
        losses["train"] += [np.mean(train_losses)]
        running_abs_error["train"] += [np.mean(train_abs_error)]

        if epoch % epochs_per_checkpoint == 0:
            print("Epoch: {}:\tRunning Loss: {:.5f}, {:.3f}".format(epoch, np.mean(losses["train"]),
                                                                    np.mean(np.log2(losses["train"])),
                                                                    torch.std(reconstruct)))
            write('train_original_example.wav', rate=int(sampling_rate[0]), data=audio[0].detach().cpu().numpy())
            write('train_reconstruction_example.wav', rate=int(sampling_rate[0]),
                  data=reconstruct[0].detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/train",
                       filename="wave_train" + str(epoch))
        model.eval()
        valid_losses = []
        valid_abs_error = []
        for i, batch in enumerate(valid_loader):
            audio, targets, sampling_rate = batch
            audio = audio.cuda()
            reconstruct, kl = model(audio.unsqueeze(1))
            reconstruct = reconstruct.squeeze()
            audio = audio[:, :reconstruct.shape[1]]
            loss = criterion(reconstruct, audio.cuda()[:, :reconstruct.shape[1]]) + torch.mean(kl).item()
            reduced_loss = loss.item()
            valid_losses += [loss.item()]
            valid_abs_error += [float(torch.mean(torch.abs_(reconstruct - audio.cuda())).item())]
            logger.add_scalar('training loss', np.log2(reduced_loss), i + len(train_loader) * epoch)
        losses["valid"] += [np.mean(valid_losses)]
        running_abs_error["valid"] += [np.mean(valid_abs_error)]
        if epoch % epochs_per_checkpoint == 0:
            save_checkpoint(model, optimizer, learning_rate, epoch, output_directory, z_dim=z_dim, gated=gated,
                            name=name, train_losses=losses['train'], valid_losses=losses['valid'])
            print("Epoch: {}:\tValid Loss: {:.5f}, {:.3f}".format(epoch, np.mean(losses["valid"]),
                                                                  np.mean(np.log2(losses["valid"])),
                                                                  torch.std(reconstruct)))
            write('valid_reconstruction_example.wav', rate=int(sampling_rate[0]),
                  data=reconstruct[0].detach().cpu().numpy())
            write('valid_original_example.wav', rate=int(sampling_rate[0]), data=audio[0].detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/valid",
                       filename="wave_valid" + str(epoch))
            generated_sound = model.sample(torch.rand(1, 100).cuda())
            write('generated_sample_example.wav', rate=int(sampling_rate[0]), data=generated_sound.detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/valid",
                       filename="wave_valid" + str(epoch))
        plot_performance(loss_total=losses, accuracy=None, shapes=shapes, results_path="figures",
                         filename="training_loss_trace_" + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(name="vae_1dcnn",
          z_dim=100,
          batch_size=16,
          epochs=100000,
          checkpoint_path=output_directory,
          epochs_per_checkpoint=1,
          gated=False)

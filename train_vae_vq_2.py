from utils.CycleAnnealScheduler import CycleScheduler
from torch.utils.data import DataLoader
from data_preparation.load_wavs_as_tensor import Wave2tensor
import torch.nn as nn
import argparse
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
from models.unsupervised.VQVAE2 import VQVAE
from models.unsupervised.wavenet.wavenet import WaveNetEncoder
from utils.plot_performance import plot_performance
from utils.plot_waves import plot_waves
from utils.utils import create_missing_folders
import torchaudio

training_folders = [
    #"C:/Users/simon/Documents/MIR/genres/hiphop/wav/",
    #"C:/Users/simon/Documents/MIR/genres/jazz/wav/",
    #"C:/Users/simon/Documents/MIR/genres/rock/wav/",
    #"C:/Users/simon/Documents/MIR/genres/metal/wav/",
    #"C:/Users/simon/Documents/MIR/genres/classical/wav/",
    #"C:/Users/simon/Documents/MIR/genres/reggae/wav/",
    #"C:/Users/simon/Documents/MIR/genres/pop/wav/",
    #"C:/Users/simon/Documents/MIR/genres/disco/wav/",
    #"C:/Users/simon/Documents/MIR/genres/country/wav/",
    #"C:/Users/simon/Documents/MIR/genres/blues/wav/",
    "C:/Users/simon/Documents/spotify/potpourri"
]
output_directory = "C:/Users/simon/djjudge/checkpoints/"
from scipy.io.wavfile import write

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_wavenet_checkpoint(checkpoint_path, model, name="wavenet"):
    # if checkpoint_path
    checkpoint_dict = torch.load(checkpoint_path + name, map_location='cpu')
    # epoch = checkpoint_dict['epoch']
    model_for_loading = checkpoint_dict
    model.load_state_dict(model_for_loading.state_dict())
    # print("Loaded checkpoint '{}' (epoch {})".format(
    #    checkpoint_path, epoch))
    return model


def load_checkpoint(checkpoint_path, model, optimizer, z_dim, num_embeddings, name="vae_1dcnn"):
    # if checkpoint_path
    losses_recon = {
        "train": [],
        "valid": [],
    }
    vq_losses = {
        "train": [],
        "valid": [],
    }
    kl_losses = {
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
                        z_dim=z_dim, losses=losses, vq_losses=vq_losses, losses_recon=losses_recon,
                        name=name, num_embeddings=num_embeddings)
    checkpoint_dict = torch.load(checkpoint_path + name, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    # warmup_count = checkpoint_dict['warmup_count']
    # no_kld_count = checkpoint_dict['no_kld_count']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    losses_recon = checkpoint_dict['losses_recon']
    vq_losses = checkpoint_dict['vq_losses']
    # kl_losses = checkpoint_dict['kl_losses']
    losses = checkpoint_dict['losses']
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    # model.warmup_count = warmup_count
    # model.no_kld_count = no_kld_count
    return model, optimizer, epoch, losses, vq_losses, kl_losses, losses_recon


def save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path, z_dim, losses,
                    vq_losses, losses_recon, num_embeddings, name="vae_1dcnn"):
    model_for_saving = VQVAE(in_channel=1,
                             channel=128,
                             n_res_block=4,
                             n_res_channel=32,
                             embed_dim=z_dim,
                             n_embed=num_embeddings).to(device)
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'losses': losses,
                'vq_losses': vq_losses,
                # 'kl_losses': kl_losses,
                'warmup_count': model.warmup_count,
                # 'no_kld_count': model.no_kld_count,
                'losses_recon': losses_recon,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path + name)


def generate(model):
    pass


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    if length > 1:
        samples = np.stack(samples, axis=0)
    else:
        samples = samples[0]
    return samples


def transform(audio,
              name,
              z_dim,
              num_embeddings,
              checkpoint_path=None):
    model = VQVAE(in_channel=1,
                  channel=128,
                  n_res_block=4,
                  n_res_channel=32,
                  embed_dim=z_dim,
                  n_embed=num_embeddings).to(device)
    model.eval()

    model, optimizer, epoch, losses, vq_losses, kl_losses, losses_recon = load_checkpoint(
        checkpoint_path,
        model,
        None,
        z_dim,
        num_embeddings=num_embeddings,
        name=name)

    # Take only the part that fits in the model
    reconstruct, vq_loss = model(audio[:, :122761].unsqueeze(1))

    write('audio/' + name + '/train_original_example.wav', rate=22050,
          data=audio[0].detach().cpu().numpy())
    write('audio/' + name + '/train_reconstruction_example.wav', rate=22050,
          data=reconstruct[0].detach().cpu().numpy())
    plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
               wave2=audio[0].detach().cpu().numpy(),
               results_path="figures/train/" + name + '/',
               filename="wave_train" + str(epoch))


def train(name,
          z_dim,
          num_embeddings,
          batch_size=16,
          epochs=100,
          learning_rate=1e-3,
          fp16_run=False,
          checkpoint_path=None,
          epochs_per_checkpoint=1,
          is_generate=False):
    create_missing_folders('audio/' + name)
    torch.manual_seed(42)

    train_set = Wave2tensor(training_folders, scores_files=None, segment_length=122761)

    valid_set = Wave2tensor(training_folders, scores_files=None, segment_length=122761, valid=True)

    train_loader = DataLoader(train_set, num_workers=4,
                              shuffle=True,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    valid_loader = DataLoader(valid_set, num_workers=4,
                              shuffle=True,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    model = VQVAE(in_channel=1,
                  channel=128,
                  n_res_block=4,
                  n_res_channel=32,
                  embed_dim=z_dim,
                  n_embed=num_embeddings,
                  ).to(device)
    model.random_init()
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # Load checkpoint if one exists
    epoch = 0
    if checkpoint_path is not None:
        model, optimizer, epoch, losses, vq_losses, kl_losses, losses_recon = load_checkpoint(
            checkpoint_path, model,
            optimizer,
            z_dim,
            num_embeddings=num_embeddings,
            name=name)

    # Get shared output_directory ready
    logger = SummaryWriter('logs')
    epoch_offset = max(1, epoch)
    lr_schedule = CycleScheduler(optimizer, learning_rate, n_iter=epochs * len(train_loader))
    losses = {
        "train": [],
        "valid": [],
    }
    vq_losses = {
        "train": [],
        "valid": [],
    }
    # kl_losses = {
    #   "train": [],
    #    "valid": [],
    # }
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
        train_vq_loss = []
        # train_kl_loss = []
        train_recons = []
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            audio, targets, sampling_rate = batch

            # The upsampling returns batch_sizex1x295233 dimensions (1 channel, mono sound); all the extra inputs are
            # put to zeros so it should not affect the output

            # audio[:, 28033:] = torch.zeros(audio.shape[1] - 28033).to(device)
            audio = torch.autograd.Variable(audio).to(device)
            # reconstruct, vq_loss, kl = model(audio.unsqueeze(1))
            reconstruct, vq_loss = model(audio.unsqueeze(1))
            reconstruct = reconstruct.squeeze()
            # kl = torch.mean(kl)
            # if model.warmup_count < model.warmup:
            #    kl = (model.warmup_count / model.warmup) * kl
            # if model.no_kld_count < model.no_kld_epochs:
            #    kl = 0 * kl

            audio = audio[:, :reconstruct.shape[1]]

            loss_recon = criterion(reconstruct, audio.to(device)).sum()
            loss = loss_recon + vq_loss  # + kl
            train_losses += [loss.item()]
            train_vq_loss += [vq_loss.item()]
            # train_kl_loss += [kl.item()]
            train_recons += [loss_recon.item()]
            train_abs_error += [float(torch.mean(torch.abs_(reconstruct - audio.to(device))).item())]

            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                del scaled_loss
            else:
                loss.backward()
            optimizer.step()
            lr_schedule.step()
            lr = optimizer.param_groups[0]['lr']
            logger.add_scalar('training_loss', loss, i + len(train_loader) * epoch)
            del targets, vq_loss, loss, loss_recon
        losses["train"] += [np.mean(train_losses)]
        vq_losses["train"] += [np.mean(train_vq_loss)]
        # kl_losses["train"] += [np.mean(train_kl_loss)]
        losses_recon["train"] += [np.mean(train_recons)]
        running_abs_error["train"] += [np.mean(train_abs_error)]
        # if model.no_kld_count < model.no_kld_epochs:
        #    model.no_kld_count += 1
        #    print("no kld loss:", float(model.no_kld_count / model.no_kld_epochs))
        # elif model.warmup_count < model.warmup:
        #    model.warmup_count += 1
        #    print("Warmup:", float(model.warmup_count / model.warmup))

        if epoch % epochs_per_checkpoint == 0:
            print("Epoch: {}:\tTrain Loss: {:.6f} , vq_loss: {:.6f} , recon: {:.6f}, lr: ".format(epoch,
                                                                                                  np.mean(
                                                                                                      losses["train"]),
                                                                                                  np.mean(vq_losses[
                                                                                                              "train"]),
                                                                                                  np.mean(
                                                                                                      losses_recon[
                                                                                                          "train"]),
                                                                                                  lr))
            write('audio/' + name + '/train_original_example.wav', rate=int(sampling_rate[0]),
                  data=audio[0].detach().cpu().numpy())
            write('audio/' + name + '/train_reconstruction_example.wav', rate=int(sampling_rate[0]),
                  data=reconstruct[0].detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/train/" + name + '/',
                       filename="wave_train" + str(epoch))
        del reconstruct, audio
        model.eval()
        valid_losses = []
        valid_vq_loss = []
        valid_recons = []
        valid_abs_error = []

        for i, batch in enumerate(valid_loader):
            audio, targets, sampling_rate = batch
            audio = audio.to(device)
            # reconstruct, vq_loss, kl = model(audio.unsqueeze(1))
            reconstruct, vq_loss = model(audio.unsqueeze(1))
            # kl = torch.mean(kl)
            reconstruct = reconstruct.squeeze()
            audio = audio[:, :reconstruct.shape[1]]
            loss_recon = criterion(reconstruct, audio.to(device)[:, :reconstruct.shape[1]]).sum()
            loss = loss_recon + vq_loss  # + kl
            valid_losses += [loss.item()]
            valid_vq_loss += [vq_loss.item()]
            valid_recons += [loss_recon.item()]
            valid_abs_error += [float(torch.mean(torch.abs_(reconstruct - audio.to(device))).item())]
            logger.add_scalar('training loss', np.log2(loss.item()), i + len(train_loader) * epoch)
            del targets, vq_loss, loss, loss_recon
        losses["valid"] += [np.mean(valid_losses)]
        vq_losses["valid"] += [np.mean(valid_vq_loss)]
        losses_recon["valid"] += [np.mean(valid_recons)]
        running_abs_error["valid"] += [np.mean(valid_abs_error)]

        if epoch % epochs_per_checkpoint == 0:
            save_checkpoint(model, optimizer, learning_rate, epoch, output_directory, z_dim=z_dim,
                            name=name, losses=losses, vq_losses=vq_losses, losses_recon=losses_recon,
                            num_embeddings=num_embeddings, kl_losses=None)

            print("Epoch: {}:\tValid Loss: {:.6f} , vq_loss: {:.6f} , recon: {:.6f} ".format(epoch,
                                                                                             np.mean(losses["valid"]),
                                                                                             np.mean(
                                                                                                 vq_losses["valid"]),
                                                                                             np.mean(
                                                                                                 losses_recon["valid"]),
                                                                                             lr))
            out_dir = 'audio/' + name + '/' + str(epoch)
            create_missing_folders(out_dir)
            write(out_dir + '/valid_reconstruction_example.wav', rate=int(sampling_rate[0]),
                  data=reconstruct[0].detach().cpu().numpy())
            write(out_dir + '/valid_original_example.wav', rate=int(sampling_rate[0]),
                  data=audio[0].detach().cpu().numpy())
            plot_waves(wave1=reconstruct[0].detach().cpu().numpy(),
                       wave2=audio[0].detach().cpu().numpy(),
                       results_path="figures/valid/" + name + '/',
                       filename="wave_valid" + str(epoch))
            if is_generate:
                model_top = WaveNetEncoder(layers=10,
                                           blocks=8,
                                           dilation_channels=16,
                                           residual_channels=16,
                                           skip_channels=512,
                                           end_channels=512,
                                           output_length=4092,
                                           kernel_size=5,
                                           dtype=torch.FloatTensor,
                                           bias=True,
                                           condition_dim=0)
                model_top = load_wavenet_checkpoint("C:/Users/simon/djjudge/snapshots/", model_top,
                                                    name="potpourri_4092_2019-11-26_20-52-35")
                model_top.eval()
                # sample_top = generate_audio(model_top,
                #                             length=1,
                #                             temperatures=[0.5])
                random_input1 = torch.randn([1, 256, model_top.receptive_field + model_top.output_length])
                sample_top = model_top(random_input1)
                del model_top, random_input1
                model_bottom = WaveNetEncoder(layers=10,
                                              blocks=8,
                                              dilation_channels=16,
                                              residual_channels=16,
                                              skip_channels=512,
                                              end_channels=512,
                                              output_length=8184,
                                              kernel_size=5,
                                              dtype=torch.FloatTensor,
                                              bias=True,
                                              condition_dim=0)

                model_bottom = load_wavenet_checkpoint("C:/Users/simon/djjudge/snapshots/", model_bottom,
                                                       name="potpourri_8184_2019-11-27_01-03-30")
                model_bottom.eval()
                random_input2 = torch.randn([1, 256, model_bottom.receptive_field + model_bottom.output_length])
                sample_bottom = model_bottom(random_input2)
                del model_bottom, random_input2

                # sample_bottom = generate_audio(model_bottom,
                #                                length=1,
                #                                temperatures=[0.5])

                model.eval()
                generated_sound = model.decode_code(torch.Tensor(sample_top).unsqueeze(0).to(device),
                                                    torch.Tensor(sample_bottom).unsqueeze(0).to(device))
                write(out_dir + '/generated_sample_example.wav', rate=int(sampling_rate[0]),
                      data=generated_sound.detach().cpu().numpy())
                plot_waves(wave1=generated_sound[0][0].detach().cpu().numpy(),
                           wave2=generated_sound[0][0].detach().cpu().numpy(),
                           results_path="figures/generate/" + name + '/',
                           filename="wave_generated" + str(epoch))

        plot_performance(loss_total=losses, losses_recon=losses_recon, kl_divs=vq_losses, shapes=shapes,
                         results_path="figures",
                         filename="training_loss_trace_" + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.backends.cudnn.enabled
    z_dim = 256
    num_embeddings = 512
    n_epochs = 100000
    bs = 4
    epochs_per_checkpoint = 1
    checkpoint_path = "checkpoints/"
    train("vae_vq_gated_" + str(num_embeddings) + str(z_dim),
          z_dim,
          num_embeddings=num_embeddings,
          batch_size=bs,
          epochs=n_epochs,
          checkpoint_path=checkpoint_path,
          epochs_per_checkpoint=epochs_per_checkpoint)

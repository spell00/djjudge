from models.supevised.simple1DCNN import Simple1DCNN
from utils.CycleAnnealScheduler import CycleScheduler
from torch.utils.data import DataLoader
from data_preparation.load_wavs_as_tensor import Wave2tensor
import torch.nn as nn
import argparse
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from utils.utils import create_missing_folders

someModel = None
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

def plot_performance(running_loss, valid_loss, results_path, filename):
    fig2, ax21 = plt.subplots(figsize=(20, 20))
    ax21.plot(running_loss, 'b-', label='Train')  # plotting t, a separately
    ax21.plot(valid_loss, 'r-', label='Valid')  # plotting t, a separately
    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    # pylab.show()
    create_missing_folders(results_path + "/plots/")
    try:
        pylab.savefig(results_path + "/plots/" + filename)
    except:
        pass
    plt.close()


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})".format(
        checkpoint_path, iteration))
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    model_for_saving = Simple1DCNN().cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(batch_size=8,
          epochs=100,
          learning_rate=1e-4,
          fp16_run=False,
          checkpoint_path=None,
          epochs_per_checkpoint=1):
    torch.manual_seed(42)
    model = Simple1DCNN().cuda()
    model.random_init()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, amsgrad=True)
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path is not None:
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model, optimizer)
        iteration += 1  # next iteration is iteration + 1

    train_set = Wave2tensor(training_folders, scores, segment_length=300000)

    valid_set = Wave2tensor(training_folders, scores, segment_length=300000, valid=True)

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
    logger = SummaryWriter(os.path.join(output_directory, 'logs'))
    epoch_offset = max(1, int(iteration / len(train_loader)))
    lr_schedule = CycleScheduler(optimizer, learning_rate, n_iter=epochs * len(train_loader))
    train_losses = []
    train_mean_diffs = []
    valid_mean_diffs = []
    valid_losses = []
    for epoch in range(epoch_offset, epochs):
        running_loss = []
        val_loss = []
        model.train()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            audio, targets, sampling_rate = batch
            audio = torch.autograd.Variable(audio).cuda()
            outputs = model(audio.unsqueeze(1)).squeeze()
            loss = criterion(outputs, targets.cuda())
            reduced_loss = loss.item()
            running_loss += [reduced_loss]
            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            lr_schedule.step()
            logger.add_scalar('training_loss', reduced_loss, i + len(train_loader) * epoch)
        train_losses += [np.mean(running_loss)]
        train_mean_diffs += [float(torch.mean(torch.abs_(outputs-targets.cuda())))]

        if epoch % epochs_per_checkpoint == 0:
            print("Epoch: {}:\tTrain Loss: {:.3f}, {:.3f}".format(epoch, train_losses[-1],
                                                                  train_mean_diffs[-1],
                                                                  torch.std(outputs)))

        model.eval()
        valid_loss = []
        for i, batch in enumerate(valid_loader):
            audio, targets, sampling_rate = batch
            audio = audio.cuda()
            outputs = model(audio.unsqueeze(1)).squeeze()
            loss = criterion(outputs, targets.cuda())
            val_loss += [loss.item()]
            reduced_loss = loss.item()
            valid_loss += [loss.item()]
            logger.add_scalar('training loss', np.log2(reduced_loss), i + len(train_loader) * epoch)
        valid_losses += [np.mean(val_loss)]
        valid_mean_diffs += [float(torch.mean(torch.abs_(outputs-targets.cuda())))]

        if epoch % epochs_per_checkpoint == 0:
            checkpoint_path = "{}/cnn{}".format(output_directory, iteration)
            save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path)
            print("Epoch: {}:\tValid Loss: {:.3f}, {:.3f}".format(epoch, valid_losses[-1],
                                                                  valid_mean_diffs[-1],
                                                                  torch.std(outputs)))

        plot_performance(train_losses, valid_losses,
                         results_path="figures",
                         filename="training_MSEloss_trace_classification")
        plot_performance(train_mean_diffs, valid_mean_diffs,
                         results_path="figures",
                         filename="training_mean_abs_diff_trace_classification")

def test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(batch_size=8, epochs=100000)

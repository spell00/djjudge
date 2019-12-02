from djjudge.models.supervised.CNN_1D import Simple1DCNN, ConvResnet
from .utils.CycleAnnealScheduler import CycleScheduler
from torch.utils.data import DataLoader
from .data_preparation.load_wavs_as_tensor import Wave2tensor
import torch.nn as nn
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from .utils.utils import create_missing_folders
import math


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def boxplots_genres(scores, results_path, filename="boxplots_genres", offset=100):
    create_missing_folders(os.getcwd() + "/" + results_path + "/plots/")
    fig2, ax21 = plt.subplots()

    scores_sorted_lists = [sorted(rand_jitter(np.array(scores[i * offset:(i + 1) * offset]))) for i in range(10)]
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    colors = ['darkorange', 'cornflowerblue', 'darkviolet', 'chocolate', 'yellowgreen', 'lightseagreen',
              'forestgreen', 'crimson', 'coral', 'wheat']
    box = ax21.boxplot(scores_sorted_lists, vert=0, patch_artist=True, labels=genres)  # plotting t, a separately
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()
    del scores, scores_sorted_lists


def performance_per_score(predicted_values, target_values, results_path, n, filename="scores_performance", valid=False):
    create_missing_folders(results_path + "/plots/")
    fig2, ax21 = plt.subplots()
    predicted_values = np.array(predicted_values)
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    colors = ['darkorange', 'cornflowerblue', 'darkviolet', 'chocolate', 'yellowgreen', 'lightseagreen',
              'forestgreen', 'crimson', 'coral', 'wheat']

    target_values = np.array(target_values)
    # ax21.set_ylim([0, 1])
    # ax21.set_xlim([0, 1])
    if not valid:
        plt.scatter(target_values, rand_jitter(predicted_values), facecolors='none', edgecolors="r")
    else:
        for i, (c, genre) in enumerate(zip(colors, genres)):
            plt.scatter(target_values[i * n:(i + 1) * n], rand_jitter(predicted_values[i * n:(i + 1) * n]),
                        facecolors='none', edgecolors=c)

    ax21.hlines(np.mean(predicted_values), xmin=0, xmax=1, colors='b', label='Predicted values average')
    ax21.hlines(np.mean(target_values), xmin=0, xmax=1, colors='k', label='Target values average')
    plt.plot(np.unique(target_values),
             np.poly1d(np.polyfit(target_values, predicted_values, 1))(np.unique(target_values)), label="Best fit")
    ident = [0.0, 1.0]
    ax21.plot(ident, ident, color="g", label='Identity line')
    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()
    del predicted_values, target_values, results_path


def plot_data_distribution(train_scores, valid_scores, results_path, filename="scores_data_distribution"):
    create_missing_folders(results_path + "/plots/")
    fig2, ax21 = plt.subplots()

    scores_train = sorted(train_scores)
    scores_valid = sorted(valid_scores)

    ax21.plot(scores_train, 'b--', label='Train')  # plotting t, a separately
    ax21.plot(scores_valid, 'r--', label='Valid')  # plotting t, a separately
    ax21.hlines(np.mean(train_scores), xmin=0, xmax=900, colors='b', label='Train mean')
    ax21.hlines(np.mean(valid_scores), xmin=0, xmax=900, colors='r', label='Valid mean')
    # ax21.vlines(500, ymin=0, ymax=1, colors='k')
    ax21.set_xlabel('epochs')
    ax21.set_ylabel('Loss')
    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()


def plot_performance(running_loss, valid_loss, results_path, filename):
    create_missing_folders(results_path + "/plots/")
    fig2, ax21 = plt.subplots()
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
    print("importing checkpoint from", checkpoint_path)
    assert os.path.isfile(checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    return model, optimizer, epoch


def load_checkpoint_for_test(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}')".format(checkpoint_path))
    return model


def save_checkpoint(model, optimizer, learning_rate, epoch, filepath, channel, n_res_block, n_res_channel,
                    stride, dense_layers_size, is_bns, is_dropout, activation, final_activation, dropval):
    print("Saving model and optimizer state at epoch {} to {}".format(
        epoch, filepath))
    model_for_saving = ConvResnet(in_channel=1,
                                  channel=channel,
                                  n_res_block=n_res_block,
                                  n_res_channel=n_res_channel,
                                  stride=stride,
                                  dense_layers_sizes=dense_layers_size,
                                  is_bns=is_bns,
                                  is_dropouts=is_dropout,
                                  activation=activation,
                                  final_activation=final_activation,
                                  drop_val=dropval).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'epoch': epoch,
                'val_loss': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def mse_corr(x, targets):
    arr = []
    for score, t in zip(x, targets):
        arr += [torch.abs_(score * (0.4 - t))]
    return torch.stack(arr).cuda()


def nlll(x):
    ent = []
    for freq in x:
        if freq > 0:
            ent += [-freq * math.log(freq, 2)]
        else:
            ent += [torch.Tensor([0])[0].cuda()]
        del freq
    return torch.stack(ent)


def test(
        training_folders,
        scores,
        output_directory,
        checkpoint_path=None, ):
    torch.manual_seed(42)
    model = Simple1DCNN().cuda()
    model.random_init()
    criterion = nn.MSELoss()
    # Load checkpoint if one exists
    epoch = 0
    if checkpoint_path is not None:
        model, optimizer = load_checkpoint_for_test(checkpoint_path, model)
        epoch += 1  # next epoch is epoch + 1

    valid_set = Wave2tensor(training_folders, scores, segment_length=300000, valid=True)
    valid_loader = DataLoader(valid_set, num_workers=0,
                              shuffle=False,
                              batch_size=1,
                              pin_memory=False,
                              drop_last=False)

    model.eval()
    valid_abs = []
    for i, batch in enumerate(valid_loader):
        audio, targets, sampling_rate = batch
        audio = audio.cuda()
        outputs = model(audio.unsqueeze(1)).squeeze()
        valid_abs += [torch.mean(torch.abs_(outputs - targets.cuda())).item()]
    boxplots_genres(valid_abs, results_path="figures/boxplots", filename="boxplot_valid_performance_per_genre",
                    offset=20)
    del valid_abs


def train(training_folders,
          scores,
          output_directory,
          batch_size=8,
          epochs=100,
          learning_rate=1e-3,
          fp16_run=False,
          checkpoint_path=None,
          epochs_per_checkpoint=50,
          channel=256,
          n_res_block=4,
          n_res_channel=256,
          stride=4,
          activation=torch.sigmoid,
          dense_layers_sizes=[32, 1],
          is_bns=[1, 1],
          is_dropouts=[0, 0],
          final_activation=None,
          drop_val=0.5,
          loss_type=nn.MSELoss
          ):
    torch.manual_seed(42)
    dense_layers_sizes = [channel] + dense_layers_sizes
    model = ConvResnet(in_channel=1,
                       channel=channel,
                       n_res_block=n_res_block,
                       n_res_channel=n_res_channel,
                       stride=stride,
                       dense_layers_sizes=dense_layers_sizes,
                       is_bns=is_bns,
                       is_dropouts=is_dropouts,
                       activation=activation,
                       final_activation=final_activation,
                       drop_val=drop_val
                       ).cuda()
    model.random_init()
    criterion = loss_type()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, amsgrad=True)
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # Load checkpoint if one exists
    epoch = 0
    if checkpoint_path is not None:
        print("Getting checkpoint at", checkpoint_path)
        model, optimizer, epoch = load_checkpoint(checkpoint_path, model, optimizer)
        epoch += 1  # next epoch is epoch + 1

    all_set = Wave2tensor(training_folders, scores, segment_length=300000, all=True, valid=False)
    boxplots_genres(all_set.scores, results_path="figures")
    del all_set
    train_set = Wave2tensor(training_folders, scores, segment_length=300000)

    valid_set = Wave2tensor(training_folders, scores, segment_length=300000, valid=True)

    plot_data_distribution(train_set.scores, valid_set.scores, results_path="figures")
    train_loader = DataLoader(train_set, num_workers=0,
                              shuffle=True,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)
    valid_loader = DataLoader(valid_set, num_workers=0,
                              shuffle=False,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=False)

    # Get shared output_directory ready
    # logger = SummaryWriter(os.path.join(output_directory, 'logs'))
    epoch_offset = max(1, epoch)
    lr_schedule = CycleScheduler(optimizer, learning_rate, n_iter=epochs * len(train_loader))
    losses = {
        "train": {
            "abs_error": [],
            "mse": []
        },
        "valid": {
            "abs_error": [],
            "mse": [],
        }
    }
    model.cuda()
    for epoch in range(epoch_offset, epochs):
        loss_list = {
            "train": {
                "abs_error": [],
                "mse": [],
                "targets_list": [],
                "outputs_list": [],
            },
            "valid": {
                "valid_abs_all": [],
                "abs_error": [],
                "mse": [],
                "targets_list": [],
                "outputs_list": [],
            }
        }

        model.train()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            audio, targets, sampling_rate = batch
            audio = torch.autograd.Variable(audio).cuda()
            outputs = model(audio.unsqueeze(1)).squeeze()

            noise = torch.rand(size=[len(targets)]).normal_() * 0.01
            targets = targets.cuda() # + noise.cuda()

            mse_loss = criterion(outputs, targets)
            loss = mse_loss
            # logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
            train_abs = torch.mean(torch.abs_(outputs - targets.cuda()))
            loss_list["train"]["outputs_list"].extend(outputs.detach().cpu().numpy())
            loss_list["train"]["targets_list"].extend(targets.detach().cpu().numpy())
            loss_list["train"]["mse"] += [mse_loss.item()]
            loss_list["train"]["abs_error"] += [mse_loss.item()]
            if fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            lr_schedule.step()
            del loss, outputs, targets, train_abs, audio, mse_loss  # , energy_loss
        losses["train"]["mse"] += [float(np.mean(loss_list["train"]["mse"]))]
        losses["train"]["abs_error"] += [float(np.mean(loss_list["train"]["abs_error"]))]
        performance_per_score(loss_list["train"]["outputs_list"], loss_list["train"]["targets_list"],
                              results_path="figures",
                              filename="scores_performance_train.png", n=80)
        del loss_list["train"]["outputs_list"], loss_list["train"]["targets_list"]
        if epoch % epochs_per_checkpoint == 0:
            print("Epoch: {}:\tTrain Loss: {:.3f}, Energy loss: {:.3f}".format(epoch, losses["train"]["mse"][-1],
                                                                               losses["train"]["abs_error"][-1]),
                  )

        model.eval()
        for i, batch in enumerate(valid_loader):
            audio, targets, sampling_rate = batch
            audio = audio.cuda()
            outputs = model(audio.unsqueeze(1)).squeeze()
            # energy_loss = shannon_entropy(outputs, targets.cuda()).mean()
            mse_loss = criterion(outputs, targets.cuda())
            loss = mse_loss
            # logger.add_scalar('training loss', np.log2(loss.item()), i + len(train_loader) * epoch)
            loss_list["valid"]["outputs_list"].extend(outputs.detach().cpu().numpy())
            loss_list["valid"]["targets_list"].extend(targets.detach().cpu().numpy())
            loss_list["valid"]["mse"] += [mse_loss.item()]
            loss_list["valid"]["abs_error"] += [mse_loss.item()]
            loss_list["valid"]["valid_abs_all"].extend(torch.abs_(outputs - targets.cuda()).detach().cpu().numpy())
            del loss, audio, outputs, targets  # , energy_loss
        losses["valid"]["mse"] += [float(np.mean(loss_list["valid"]["mse"]))]
        losses["valid"]["abs_error"] += [float(np.mean(loss_list["valid"]["abs_error"]))]
        boxplots_genres(loss_list["valid"]["valid_abs_all"], results_path="figures",
                        filename="boxplot_valid_performance_per_genre_valid.png", offset=20)

        if epoch % epochs_per_checkpoint == 0:
            checkpoint_path = "{}/classif_ckpt/cnn".format(output_directory)
            save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_path, channel, n_res_block,
                            n_res_channel, stride, dense_layers_sizes, is_bns, is_dropouts, activation,
                            final_activation, drop_val)
            print("Epoch: {}:\tValid Loss: {:.3f}, Energy loss: {:.3f}".format(epoch,
                                                                               losses["valid"]["mse"][-1],
                                                                               losses["valid"]["abs_error"][-1]))

        plot_performance(losses["train"]["mse"], losses["valid"]["mse"], results_path="figures",
                         filename="training_MSEloss_trace_classification")
        plot_performance(losses["train"]["abs_error"], losses["valid"]["abs_error"], results_path="figures",
                         filename="training_mean_abs_diff_trace_classification")
        performance_per_score(loss_list["valid"]["outputs_list"], loss_list["valid"]["targets_list"],
                              results_path="figures",
                              filename="scores_performance_valid.png", n=20, valid=True)
        del loss_list


"""
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    num_gpus = torch.cuda.device_count()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(batch_size=16,
          epochs=100000,
          epochs_per_checkpoint=1,
          learning_rate=1e-3,
          fp16_run=True,
          checkpoint_path="classif_ckpt/cnn")

"""

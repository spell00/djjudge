from .models.supervised.CNN_1D import Simple1DCNN, ConvResnet
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

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def boxplots_genres(scores, results_path, filename="boxplots_genres", offset=100):
    create_missing_folders(results_path + "/plots/")
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


def performance_per_score(predicted_values, target_values, results_path, n, filename="scores_performance", valid=False,
                          noise=False):
    create_missing_folders(results_path + "/plots/")
    fig2, ax21 = plt.subplots()
    predicted_values = np.array(predicted_values)
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    colors = ['darkorange', 'cornflowerblue', 'darkviolet', 'chocolate', 'yellowgreen', 'lightseagreen',
              'forestgreen', 'crimson', 'coral', 'wheat']

    target_values = np.array(target_values)
    # ax21.set_ylim([0, 1])
    # ax21.set_xlim([0, 1])
    if not valid and not noise:
        plt.scatter(rand_jitter(target_values), predicted_values, facecolors='none', edgecolors="r")
    elif not valid and noise:
        plt.scatter(target_values, predicted_values, facecolors='none', edgecolors="r")
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


def load_checkpoint(checkpoint_path, model, optimizer, fp16_run=False):
    print("importing checkpoint from", checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    return model, optimizer, epoch


def load_checkpoint_for_test(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}')".format(checkpoint_path))
    return model


def save_checkpoint(model, optimizer, learning_rate, epoch, filepath, channel, n_res_block, n_res_channel,
                    stride, dense_layers_size, is_bns, is_dropout, activation, final_activation, dropval, model_type,
                    is_bayesian):
    print("Saving model and optimizer state at epoch {} to {}".format(
        epoch, filepath))
    if model_type == "convresnet":
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
                                      drop_val=dropval,
                                      is_bayesian=is_bayesian).to(device)
    else:
        model_for_saving = Simple1DCNN(
            activation=activation,
            final_activation=final_activation,
            drop_val=dropval,
            is_bns=is_bns,
            is_dropouts=is_dropout,
            is_bayesian=is_bayesian
        )

    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'epoch': epoch,
                'val_loss': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def train(training_folders,
          scores,
          output_directory,
          batch_size=8,
          epochs=100,
          learning_rate=1e-3,
          fp16_run=False,
          checkpoint_name=None,
          epochs_per_checkpoint=50,
          channel=256,
          n_res_block=4,
          n_res_channel=256,
          stride=4,
          activation=torch.relu,
          dense_layers_sizes=[32, 1],
          is_bns=[1, 1],
          is_dropouts=[1, 1],
          final_activation=None,
          drop_val=0.5,
          loss_type=nn.MSELoss,
          init_method=nn.init.kaiming_normal_,
          noise=0.,
          average_score=0.4,
          factor=2,
          flat_extrems=False,
          model_type="convresnet",
          is_bayesian=False
          ):
    if noise > 0.:
        is_noisy = True
    else:
        is_noisy = False

    if len(str(activation).split(" ")) > 1:
        activation_string = str(activation).split(" ")[2]
    else:
        activation_string = str(activation).split('torch.')[0]

    if len(str(final_activation).split(" ")) > 1:
        final_activation_string = str(final_activation).split(" ")[2]
    else:
        final_activation_string = str(final_activation).split('torch.')[0]

    checkpoint_complete_name = "{}_{}_{}_{}_{}_{}_{}".format(checkpoint_name, model_type,
                                                             str(init_method).split(" ")[1],
                                                             activation_string,
                                                             final_activation_string,
                                                             str(is_bns), str(is_dropouts))

    torch.manual_seed(42)
    dense_layers_sizes = [channel] + dense_layers_sizes
    if model_type == "convresnet":
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
                           drop_val=drop_val,
                           is_bayesian=is_bayesian
                           ).to(device)
    else:
        model = Simple1DCNN(
            activation=activation,
            final_activation=final_activation,
            drop_val=drop_val,
            is_bns=is_bns,
            is_dropouts=is_dropouts,
            is_bayesian=is_bayesian
        )
    model.random_init(init_method=init_method)
    criterion = loss_type()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, amsgrad=True)

    # Load checkpoint if one exists
    epoch = 0
    if checkpoint_name is not None:
        print("Getting checkpoint at", checkpoint_complete_name)
        try:
            model, optimizer, epoch = load_checkpoint(checkpoint_complete_name, model, optimizer, fp16_run)
        except:
            print("No model found by the given name. Making a new model...")
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
    lr_schedule = CycleScheduler(optimizer, learning_rate, n_iter=(epochs - epoch_offset) * len(train_loader))
    losses = {
        "train": {
            "abs_error": [],
            "mse": []
        },
        "valid": {
            "abs_error": [],
            "mse": [],
            "mse_loss_mle": [],
        }
    }
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    print(model.parameters)

    for epoch in range(epoch_offset, epochs):
        loss_list = {
            "train": {
                "abs_error": [],
                "mse": [],
                "targets_list": [],
                "outputs_list": [],
                "outputs_list2": [],
            },
            "valid": {
                "valid_abs_all": [],
                "abs_error": [],
                "mse": [],
                "targets_list": [],
                "outputs_list": [],
                "outputs_list_mle": [],
                "mse_loss_mle": []
            }
        }

        model.train()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            audio, targets, sampling_rate = batch
            audio = torch.autograd.Variable(audio).to(device)
            outputs = model(audio.unsqueeze(1)).squeeze()
            outputs_original = outputs.clone().detach()
            noisy = torch.rand(len(targets)).normal_() * noise
            targets += noisy

            targets = targets.to(device)

            """
            If predicted values are <= targets and the targets are also <= 0, then no correction
            If predicted values are >= targets and the targets are also >= 1, then no correction

            There are surely better songs than those we used, which reach the max of 1.0, so the prediction 
            can predict better scores. If the scores if 1.3 but the real label is 1.0, we do not really want to 
            correct this because it will make the classifier more conservative, which is not wanted

            """
            lt_zero = torch.where((targets <= 0.) & (outputs <= targets))[0]
            gt_one = torch.where((targets >= 1.) & (outputs >= targets))[0]
            if flat_extrems:
                if len(lt_zero) > 0:
                    print("Corrected some smaller than 0 values")
                    outputs[lt_zero] = targets[lt_zero].clone().detach()
                if len(gt_one) > 0:
                    print("Corrected some larger than 1 values")
                    outputs[gt_one] = targets[gt_one].clone().detach()

            """
            If predicted values are >= targets and the targets are also >= 0.4 (the average score):
                the squared difference |target-0.4|*(target - output)**2 is added to the actual score to give a corrected scores

            If predicted values are <= targets and the targets are also <= 0.4 (the average score):
                the squared difference |target-0.4|*(target - output)**2 is substracted to the actual score to give a corrected scores

            This correction must be greater at the extremes than at the average scores, where it should not change 
                the score at all, this is why it is weighted by |target-0.4|*


            """

            lt_avg = torch.where((targets < average_score) & (outputs < targets))[0]
            gt_avg = torch.where((targets > average_score) & (outputs > targets))[0]
            if factor <= 1.0:
                if len(lt_avg) > 0:
                    # O* = O + [ ( T - O) - ( ( T + ( O - T ) ^ 2 )  * ( 1 - | avg - T | ) ) ]
                    outputs[lt_avg] = outputs[lt_avg].clone().detach() + (
                                (targets[lt_avg].clone().detach() - outputs[lt_avg].clone().detach()) - ((torch.abs(
                            outputs[lt_avg].clone().detach() - targets[lt_avg].clone().detach()).log1p_() * factor) * (
                                                                                                                     1 - torch.abs(
                                                                                                                 targets[
                                                                                                                     lt_avg].clone().detach() - average_score))))  # / average_score
                if len(gt_avg) > 0:
                    # O* = O - [ ( O - T) - ( ( T + ( O - T ) ^ 2 )  * ( 1 - | avg - T | ) ) ]
                    outputs[gt_avg] = outputs[gt_avg].clone().detach() - (
                                (outputs[gt_avg].clone().detach() - targets[gt_avg].clone().detach()) - ((torch.abs(
                            outputs[gt_avg].clone().detach() - targets[gt_avg].clone().detach()).log1p_() * factor) * (
                                                                                                                     1 - torch.abs(
                                                                                                                 targets[
                                                                                                                     gt_avg].clone().detach() - average_score))))  # / average_score

            mse_loss = criterion(outputs, targets)
            loss = mse_loss
            # logger.add_scalar('training_loss', loss.item(), i + len(train_loader) * epoch)
            # train_abs = torch.mean(torch.abs_(outputs - targets.to(device)))
            loss_list["train"]["outputs_list"].extend(outputs_original.detach().cpu().numpy())
            loss_list["train"]["outputs_list2"].extend(outputs.detach().cpu().numpy())
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
            del loss, outputs, targets, audio, mse_loss, noisy, outputs_original  # , energy_loss
        losses["train"]["mse"] += [float(np.mean(loss_list["train"]["mse"]))]
        losses["train"]["abs_error"] += [float(np.mean(loss_list["train"]["abs_error"]))]

        performance_per_score(loss_list["train"]["outputs_list"], loss_list["train"]["targets_list"],
                              results_path="figures",
                              filename="scores_performance_train.png", n=80, noise=is_noisy)
        performance_per_score(loss_list["train"]["outputs_list2"], loss_list["train"]["targets_list"],
                              results_path="figures",
                              filename="scores_performance_train_corrected.png", n=80, noise=is_noisy)
        del loss_list["train"]["outputs_list"], loss_list["train"]["targets_list"]
        if epoch % epochs_per_checkpoint == 0:
            print("Epoch: {}:\tTrain Loss: {:.3f}, Energy loss: {:.3f}".format(epoch, losses["train"]["mse"][-1],
                                                                               losses["train"]["abs_error"][-1]),
                  )

        model.eval()
        for i, batch in enumerate(valid_loader):
            audios, targets, sampling_rate = batch
            audios = torch.stack([audio.to(device) for audio in audios]).view(3, -1, 300000)

            outputs = [model(audio.unsqueeze(1)).squeeze() for audio in audios]

            mse_losses = torch.stack([criterion(out, targets.to(device)) for out in outputs])
            mse_loss = torch.min(mse_losses)
            argmin = int(torch.argmin(mse_losses))
            abs = torch.abs_(outputs[argmin] - targets.to(device))
            loss_list["valid"]["outputs_list"].extend(outputs[argmin].detach().cpu().numpy())
            loss_list["valid"]["targets_list"].extend(targets.detach().cpu().numpy())
            loss_list["valid"]["mse"] += [mse_loss.item()]
            loss_list["valid"]["valid_abs_all"].extend(abs.detach().cpu().numpy())

            del mse_loss, mse_losses, outputs, argmin, abs  # , energy_loss

            outputs_mle = [model.mle_forward(audio.unsqueeze(1)).squeeze() for audio in audios]
            mse_losses_mle = torch.stack([criterion(out, targets.to(device)) for out in outputs_mle])
            mse_loss_mle = torch.min(mse_losses_mle)
            argmin_mle = int(torch.argmin(mse_losses_mle))
            loss_list["valid"]["outputs_list_mle"].extend(outputs_mle[argmin_mle].detach().cpu().numpy())
            loss_list["valid"]["mse_loss_mle"] += [mse_loss_mle.item()]
            del mse_loss_mle, mse_losses_mle, audios, outputs_mle, targets, argmin_mle  # , energy_loss
        losses["valid"]["mse"] += [float(np.mean(loss_list["valid"]["mse"]))]
        losses["valid"]["mse_loss_mle"] += [float(np.mean(loss_list["valid"]["mse_loss_mle"]))]
        boxplots_genres(loss_list["valid"]["valid_abs_all"], results_path="figures",
                        filename="boxplot_valid_performance_per_genre_valid.png", offset=20)

        if epoch % epochs_per_checkpoint == 0:
            save_checkpoint(model, optimizer, learning_rate, epoch, checkpoint_complete_name, channel, n_res_block,
                            n_res_channel, stride, dense_layers_sizes, is_bns, is_dropouts, activation,
                            final_activation, drop_val, model_type, is_bayesian)
            print("Epoch: {}:\tValid Loss: {:.3f}, Energy loss: {:.3f}".format(epoch,
                                                                               losses["valid"]["mse"][-1],
                                                                               losses["valid"]["mse_loss_mle"][-1]))

        plot_performance(losses["train"]["mse"], losses["valid"]["mse"], results_path="figures",
                         filename="training_MSEloss_trace_classification")
        # plot_performance(losses["train"]["mse_loss_mle"], losses["valid"]["mse_loss_mle"], results_path="figures",
        #                  filename="training_mean_abs_diff_trace_classification")
        performance_per_score(loss_list["valid"]["outputs_list"], loss_list["valid"]["targets_list"],
                              results_path="figures",
                              filename="scores_performance_valid.png", n=20, valid=True)
        performance_per_score(loss_list["valid"]["outputs_list_mle"], loss_list["valid"]["targets_list"],
                              results_path="figures",
                              filename="scores_performance_valid_mle.png", n=20, valid=True)
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
          checkpoint_path="classif_ckpt/cnn_corr_normal_init")

"""

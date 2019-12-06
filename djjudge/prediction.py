from .models.supervised.CNN_1D import Simple1DCNN, ConvResnet
from torch.utils.data import DataLoader
from .data_preparation.load_wavs_as_tensor import Wave2tensor
import torch.nn as nn
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from .utils.utils import create_missing_folders


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


def performance_per_score(predicted_values, results_path, filename="scores_performance"):
    create_missing_folders(results_path + "/plots/")
    fig2, ax21 = plt.subplots()
    predicted_values = np.array(predicted_values)

    plt.plot(np.sort(predicted_values))

    ax21.hlines(np.mean(predicted_values), xmin=0, xmax=1, colors='b', label='Predicted values average')
    handle, label = ax21.get_legend_handles_labels()
    ax21.legend(handle, label)
    fig2.tight_layout()
    pylab.savefig(results_path + "/plots/" + filename)
    plt.close()
    del predicted_values, results_path


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


def load_checkpoint(checkpoint_path, model, fp16_run=False):
    print("importing checkpoint from", checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, opt_level='O2')

    return model, epoch


def predict(predict_folders,
            batch_size=8,
            fp16_run=False,
            checkpoint_name=None,
            channel=256,
            n_res_block=4,
            n_res_channel=256,
            stride=4,
            activation=torch.relu,
            dense_layers_sizes=[32, 1],
            is_bns=[0, 0],
            is_dropouts=[0, 0],
            final_activation=None,
            drop_val=0.5,
            init_method=nn.init.kaiming_normal_,
            model_type="convresnet"
            ):
    checkpoint_complete_name = "{}_{}_{}_{}_{}_{}_{}".format(checkpoint_name, model_type,
                                                             str(init_method).split(" ")[1],
                                                             str(activation).split('torch.')[0],
                                                             str(final_activation).split('torch.')[0],
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
                           drop_val=drop_val
                           ).cuda()
    else:
        model = Simple1DCNN(
            activation=activation,
            final_activation=final_activation,
            drop_val=drop_val,
            is_bns=is_bns,
            is_dropouts=is_dropouts
        )
    if fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(model, opt_level='O2')

    # Load checkpoint if one exists
    if checkpoint_name is not None:
        print("Getting checkpoint at", checkpoint_complete_name)
        try:
            model, epoch = load_checkpoint("../" + checkpoint_name, model, fp16_run)
        except:
            print("No model found by the given name. Making a new model...")
    predict_set = Wave2tensor(predict_folders, scores_files=None, segment_length=300000, all=True, valid=False, pred=True)
    songs_list = predict_set.audio_files
    # boxplots_genres(predict_set.scores, results_path="figures")

    loader = DataLoader(predict_set, num_workers=0,
                        shuffle=False,
                        batch_size=batch_size,
                        pin_memory=False,
                        drop_last=False)
    # Get shared output_directory ready
    # logger = SummaryWriter(os.path.join(output_directory, 'logs'))

    loss_list = {
        "predict": {
            "abs_error": [],
            "mse": [],
            "targets_list": [],
            "outputs_list": [],
            "outputs_list2": [],
        },
    }

    model.eval()
    predictions = []
    for i, batch in enumerate(loader):
        print(i, "/", len(loader))
        model.zero_grad()
        audio, _, sampling_rate = batch
        audio = torch.autograd.Variable(audio).cuda()
        preds = model(audio.unsqueeze(1)).squeeze()
        predictions += [preds.detach().cpu().numpy()]
        loss_list["predict"]["outputs_list"] += [predictions[-1]]

        del audio
    predictions = np.stack(predictions)
    best_scores, best_songs = torch.topk(torch.Tensor(predictions), 10)
    best_predictions_names = [songs_list[s] for s in best_songs.numpy().tolist()]
    print(best_predictions_names, best_songs)

    performance_per_score(predictions, results_path='figures', filename="scores_performance_predicted.png")
    del loss_list

    plot_data_distribution(predictions, results_path="figures")

from djjudge.models.supervised.CNN_1D import *
from djjudge.utils.log_likelihood import calculate_likelihood
from torch import nn
from torch.utils.data import DataLoader
from djjudge.data_preparation.load_wavs_as_tensor import Wave2tensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
from djjudge.utils.utils import create_missing_folders

spotify = ["/home/simon/Desktop/spotify/potpourri"]
checkpoint_path="classif_ckpt/cnn_corr_bayesian_v3_convresnet_kaiming_uniform__PReLU(num_parameters=1)_None_[1, 1]_[1, 1]_[128, 1]_last"
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


if __name__ == "__main__":
    model = ConvResnet(in_channel=1,
                       channel=256,
                       n_res_block=4,
                       n_res_channel=256,
                       stride=4,
                       dense_layers_sizes=[256, 128, 1],
                       is_bns=[1, 1],
                       is_dropouts=[1, 1],
                       activation=nn.PReLU(),
                       final_activation=None,
                       drop_val=0.5,
                       is_bayesian=True,
                       random_node="last"
                       ).to(device)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = checkpoint_dict['epoch']
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading)
    print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, epoch))
    predict_set = Wave2tensor(spotify, scores_files=None, segment_length=300000, all=True, valid=False, pred=True)
    songs_list = predict_set.audio_files
    # boxplots_genres(predict_set.scores, results_path="figures")

    loader = DataLoader(predict_set, num_workers=0,
                        shuffle=False,
                        batch_size=1,
                        pin_memory=False,
                        drop_last=False)
    model.to(device)
    nll, log_likelihoods, ys = calculate_likelihood(loader, model, S=2, MB=2)
    print("Total Negative log-likelihood:", nll)

    worst_predictions = np.stack(np.flip(log_likelihoods))
    worst_predictions, worst_predictions_songs = torch.topk(torch.Tensor(worst_predictions), 10, largest=True)
    worst_predictions_names = [songs_list[s] for s in worst_predictions_songs.numpy().tolist()]
    dataframe_worst_predictions = pd.DataFrame(np.concatenate((worst_predictions.detach().cpu().numpy().reshape(-1, 1),
                                         np.array([name.split("/")[-1] for name in worst_predictions_names]).reshape(-1, 1)), 1),
                                         columns=["scores", "songname"])
    print(worst_predictions_names, worst_predictions_songs)

    performance_per_score(worst_predictions, results_path='figures', filename="worst_nll_predicted.png")

    best_predictions = np.stack(log_likelihoods)

    # largest log-likelihood are smaller
    best_predictions, best_predictions_songs = torch.topk(torch.Tensor(best_predictions), 10, largest=False)
    best_predictions_names = [songs_list[s] for s in best_predictions_songs.numpy().tolist()]
    dataframe_best_predictions = pd.DataFrame(np.concatenate((best_predictions.detach().cpu().numpy().reshape(-1, 1),
                                         np.array([name.split("/")[-1] for name in best_predictions_names]).reshape(-1, 1)), 1),
                                         columns=["scores", "songname"])
    print(best_predictions_names, best_predictions_songs)

    performance_per_score(best_predictions, results_path='figures', filename="best_nll_predicted.png")

    ys = torch.stack(ys)
    best_scores, best_songs = torch.topk(torch.mean(ys, 1).view(-1), 10, largest=True, sorted=True)
    best_scores_names = [songs_list[s] for s in best_songs.detach().cpu().numpy().tolist()]
    dataframe_best_scores = pd.DataFrame(np.concatenate((best_scores.detach().cpu().numpy().reshape(-1, 1),
                                         np.array([name.split("/")[-1] for name in best_scores_names]).reshape(-1, 1)), 1),
                                         columns=["scores", "songname"])
    print(dataframe_best_scores)
    performance_per_score(best_scores.detach().cpu(), results_path='figures', filename="best_scores_nll_performance_predicted.png")

    worst_scores, worst_songs = torch.topk(torch.mean(ys, 1).view(-1), 10, largest=False, sorted=True)
    worst_scores_names = [songs_list[s] for s in best_songs.detach().cpu().numpy().tolist()]
    dataframe_worst_scores = pd.DataFrame(np.concatenate((worst_scores.detach().cpu().numpy().reshape(-1, 1),
                                         np.array([name.split("/")[-1] for name in worst_scores_names]).reshape(-1, 1)), 1),
                                         columns=["scores", "songname"])
    print(dataframe_worst_scores)

    performance_per_score(worst_scores.detach().cpu(), results_path='figures', filename="worst_scores_nll_performance_predicted.png")

    performance_per_score(torch.mean(ys, 1).view(-1).sort()[0].detach().cpu(), results_path='figures', filename="all_scores_predicted.png")

    performance_per_score(torch.Tensor(log_likelihoods).view(-1).sort()[0].detach().cpu(), results_path='figures', filename="all_nll_performance_predicted.png")

    dataframe_worst_predictions.to_csv("dataframe_worst_predictions")
    dataframe_best_predictions.to_csv("dataframe_best_predictions")
    dataframe_best_scores.to_csv("dataframe_best_scores")
    dataframe_worst_scores.to_csv("dataframe_worst_scores")

    # TODO put GUI here!
    # TODO inputs:
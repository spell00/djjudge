import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from utils.utils import create_missing_folders


def plot_waves(wave1,
               wave2,
               results_path,
               filename="NoName"):
    """

    :param wave1:
    :param wave2:
    :param results_path:
    :param filename:
    :param verbose:
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))
    axes[0].plot(wave1, "b")
    axes[0].set_xlabel("Reconstructed")
    #axes[0].set_ylim([-1.1, 1.1])
    axes[1].plot(wave2, "r")
    axes[1].set_xlabel("Original")
    #axes[1].set_ylim([-1.1, 1.1])
    fig.tight_layout()
    create_missing_folders(results_path + "/waves/")
    pylab.savefig(results_path + "/waves/" + filename)
    plt.close()

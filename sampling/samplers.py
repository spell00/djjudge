#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

# Taken directly from  https://github.com/idiap/importance-sampling/blob/master/importance_sampling/samplers.py
from blinker import signal
import numpy as np


def _get_dataset_length(dset, default=1):
    """Return the dataset's training data length and in case the dataset is
    uncountable return a defalt value."""
    try:
        return len(dset.train_data)
    except RuntimeError:
        return default


class BaseSampler(object):
    """BaseSampler denotes the interface for all the samplers.
    Samplers should provide the rest of the program with data points to train
    on and corresponding relative weights."""

    def __init__(self, dataset, reweighting):
        self.dataset = dataset
        self.reweighting = reweighting

    def _slice_data(self, x, y, idxs):
        if isinstance(x, (list, tuple)):
            return [xi[idxs] for xi in x], y[idxs]
        else:
            return x[idxs], y[idxs]

    def _send_messages(self, idxs, xy, w, predicted_scores):
        signal("is.sample").send({
            "idxs": idxs,
            "xy": xy,
            "w": w,
            "predicted_scores": predicted_scores
        })

    def _get_samples_with_scores(self, batch_size):
        """Child classes should implement this method.
        Arguments
        ---------
        batch_size: int
                    Return at least that many samples

        Return
        ------
        idxs: array
              The indices of some samples in the dataset
        scores: array or None
                The predicted importance scores for the corresponding idxs or
                None for uniform sampling
        xy: tuple or None
            Optionally return the data for the corresponding idxs
        """
        raise NotImplementedError()

    def sample(self, batch_size):
        # Get the importance scores of some samples
        idxs1, scores, xy = self._get_samples_with_scores(batch_size)

        # Sample from the available ones
        p = scores / scores.sum() if scores is not None else None
        idxs2 = np.random.choice(len(idxs1), batch_size, p=p)
        w = self.reweighting.sample_weights(idxs2, scores)

        # Make sure we have the data
        if xy is None:
            xy = self.dataset.train_data[idxs1[idxs2]]
        else:
            x, y = xy
            xy = self._slice_data(x, y, idxs2)

        scores = scores[idxs2] if scores is not None else np.ones(batch_size)
        self._send_messages(idxs1[idxs2], xy, w, scores)
        return idxs1[idxs2], xy, w

    def update(self, idxs, results):
        pass


class ModelSampler(BaseSampler):
    """ModelSampler uses a model to score the samples and then performs
    importance sampling based on those scores.
    It can be used to implement several training pipelines where the scoring
    model is separately trained or is sampled from the main model or is the
    main model."""
    def __init__(self, dataset, reweighting, model, large_batch=1024,
                 forward_batch_size=128):
        self.model = model
        self.large_batch = large_batch
        self.forward_batch_size = forward_batch_size
        self.N = _get_dataset_length(dataset, default=1)

        super(ModelSampler, self).__init__(dataset, reweighting)

    def _get_samples_with_scores(self, batch_size):
        assert batch_size < self.large_batch

        # Sample a large number of points in random and score them
        idxs = np.random.choice(self.N, self.large_batch)
        x, y = self.dataset.train_data[idxs]
        scores = self.model.score(x, y, batch_size=self.forward_batch_size)

        return (
            idxs,
            scores,
            (x, y)
        )


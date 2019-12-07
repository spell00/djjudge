import torch
from scipy.io.wavfile import read
import os
import random
import torch
import torch.utils.data
import pandas as pd

training_files = "C:/Users/simon/Documents/MIR/genres/jazz/train_files.txt"
MAX_WAV_VALUE = 32768.0


def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def folder_to_list(folder_path):
    """
    Takes a text file of filenames and makes a list of filenames
    """

    files = os.listdir(folder_path)
    files = [folder_path + "/" + f for f in files]
    return files


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class Wave2tensor(torch.utils.data.Dataset):
    """
    All folders must be in the same parent folder
    """
    def __init__(self, folders, scores_files, segment_length, valid=False, all=False, pred=False):
        train_lim = None
        scores = None
        files_list = None
        self.audio_files = []
        self.scores = []
        self.valid = valid
        for i, folder in enumerate(folders):
            if scores_files is not None:
                try:
                    scores = pd.read_csv(scores_files[i], header=None)[1].to_numpy()
                except:
                    scores = pd.read_csv(scores_files[i], header=None, sep="\t")[1].to_numpy()

            files_list = folder_to_list(folder)
            train_lim = int(0.8 * len(files_list))
            if not valid and not all:
                files_list = files_list[:train_lim]
                self.audio_files.extend(files_list)
                if scores_files is not None:
                    scores = scores[:train_lim]
                    self.scores.extend(scores)
            elif not all:
                files_list = files_list[train_lim:]
                self.audio_files.extend(files_list)
                if scores_files is not None:
                    scores = scores[train_lim:]
                    self.scores.extend(scores)
            elif pred:
                self.audio_files.extend(files_list)
            else:
                self.audio_files.extend(files_list)
                self.scores.extend(scores)

        files_list = files_list[train_lim:]
        self.audio_files.extend(files_list)
        if scores_files is None:
            self.scores = None
        self.segment_length = segment_length

    def __getitem__(self, index):
        # Read audio
        filename = self.audio_files[index]
        if self.scores is not None:
            targets = self.scores[index]
        else:
            targets = -1
        audio, sampling_rate = load_wav_to_torch(filename)

        self.sampling_rate = sampling_rate

        # Take segment
        if not self.valid:
            if audio.size(0) >= self.segment_length:
                max_audio_start = audio.size(0) - self.segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:audio_start+self.segment_length]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data
            audio = audio / MAX_WAV_VALUE

        else:
            max_audio_start = audio.size(0) - self.segment_length
            audios = []
            for audio_start in [0, int(max_audio_start/2), max_audio_start]:
                audios += [audio[audio_start:audio_start+self.segment_length] / MAX_WAV_VALUE]
            audio = audios
        return audio, targets, sampling_rate

    def __len__(self):
        return len(self.audio_files)

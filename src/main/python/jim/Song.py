from mido import MidiFile

from jim import musicfunctions
from jim import utils


class Song(object):
    def __init__(self,data_path, chosen_idx_melody=[]):
        self.data_path = data_path
        self.mid = MidiFile(data_path)
        #MICRO SECONDS PER BEAT
        self.tempo = utils.findTempo(self.mid)
        #BEAT PER MINUTE
        self.BPM = utils.convertTempoToBPM(self.tempo)
        self.ticks_per_beat = self.mid.ticks_per_beat 
        self.instruments_list, self.idx_drums, self.channels = \
            musicfunctions.findInstrumentsAndDrumsAndChannels(self.mid)
        self.notes, aux_channels, aux_instruments_list = \
            musicfunctions.findNotes(self.mid, self.channels, self.instruments_list)
        if len(aux_channels) > 0:
            self.channels = aux_channels
            self.instruments_list = aux_instruments_list
            self.idx_drums = [self.channels.index(9)]
        self.idx_melody = musicfunctions.findMelody(self.notes, self.instruments_list, chosen_idx_melody)
        if len(chosen_idx_melody) != 0:
            self.idx_melody = chosen_idx_melody

from midiutil.MidiFile import MIDIFile
from jim import musicfunctions
from jim import utils
from jim import constants
import numpy as np


class SongGenerator(object):
    def __init__(self, song):
        self.BPM = song.BPM
        self.ticks_per_beat = song.ticks_per_beat
        self.song = song
        self.tracks = []
        self.channels = []
        self.next_channel = 0
        self.track_notes = []
        self.track_instruments = []

    def addDrums(self):
        if len(self.song.idx_drums) > 0:
            drums = self.song.notes[self.song.idx_drums[0]]
            self.addInstrument(drums, 0, 9)

    def addMelody(self):
        print("ADD MELODY ")
        for i in range(len(self.song.idx_melody)):
            voices = self.song.notes[self.song.idx_melody[i]]
            voices_add = musicfunctions.createChorus(voices, octaves_augmentation=1)
            channel = self.song.channels[self.song.idx_melody[i]]
            voices_add_regularized = utils.deleteDuration0(
                musicfunctions.rythmRegularization(voices, voices_add, 100, constants.DIFFICULTY_LVL))
            # print(voices_add_regularized)
            instrument_number = self.song.instruments_list[self.song.idx_melody[i]] - 1
            instrument_number = 0
            self.addInstrument(voices_add_regularized, instrument_number, channel)

    def addAccompaniment(self):
        for idx in range(len(self.song.notes)):
            if (idx not in self.song.idx_melody) & (idx not in self.song.idx_drums):
                voices = self.song.notes[idx]
                voices = utils.deleteDuration0(voices)
                channel = self.song.channels[idx]
                instrument_number = self.song.instruments_list[idx] - 1
                self.addInstrument(voices, instrument_number, channel)

    def addInstrument(self, notes, instrument_number, channel):
        self.tracks += [len(self.tracks)]
        self.track_instruments += [instrument_number]
        track = self.tracks[len(self.tracks) - 1]
        self.channels += [channel]
        self.track_notes += [notes]

    def generateSong(self):
        print("Generating music ...")
        print("Melody : " + str(self.song.idx_melody) + " = " + str(utils.convertInstrumentNumbers2InstrumentName([self.song.instruments_list[i] for i in self.song.idx_melody])))
        print("Final Tracks : " + str(self.tracks))
        print("Final Channels : " + str(self.channels))
        print("Final midi Instruments : " + str(self.track_instruments))
        time = 0
        self.MyMIDI = MIDIFile(len(self.tracks))

        for i in range(len(self.tracks)):
            idx_channel = (self.song.channels).index(self.channels[i])
            print(idx_channel)
            if idx_channel in self.song.idx_melody:
                volume = constants.VOLUME_MELODY
            elif idx_channel in self.song.idx_drums:
                volume = constants.VOLUME_DRUMS
            else:
                volume = constants.VOLUME_ACCOMPANIMENT
            track_voices = self.tracks[i]
            channel_voices = self.channels[i]
            self.MyMIDI.addTempo(track_voices, 0, self.BPM)
            notes = self.track_notes[i]
            self.MyMIDI.addProgramChange(track_voices, channel_voices, 0, self.track_instruments[i])
            for (i, info) in enumerate(notes):
                if info[1] != 0:
                    self.MyMIDI.addNote(track_voices, channel_voices, info[0], utils.convertTicks2Beat(info[3], self.ticks_per_beat), utils.convertTicks2Beat(info[1], self.ticks_per_beat), volume)

        with open(constants.PATH_GENERATED_SONG, "wb") as output_file:
            self.MyMIDI.writeFile(output_file)
        print("generateSong done")
        return constants.PATH_GENERATED_SONG

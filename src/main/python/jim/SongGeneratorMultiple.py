from midiutil.MidiFile import MIDIFile
from music21 import volume
from jim import constants
from jim import musicfunctions
from jim import utils


class SongGeneratorMultiple(object):
    def __init__(self, array_songs):
        self.array_songs = array_songs
        self.BPM = array_songs[0].BPM
        self.ticks_per_beat = array_songs[0].ticks_per_beat
        # self.song = song
        self.tracks = []
        self.channels = []
        self.next_channel = 0
        self.track_notes = []
        self.track_instruments = []

    def addDrums(self):
        if len(self.array_songs[0].idx_drums) > 0:
            drums = self.array_songs[0].notes[self.array_songs[0].idx_drums[0]]
            self.addInstrument(drums, 100, 0, 9)

    def addMelody(self):
        melody_listA = [self.array_songs[0].notes[self.array_songs[0].idx_melody[i]] for i in
                        range(len(self.array_songs[0].idx_melody))]
        melody_listB = [self.array_songs[1].notes[self.array_songs[1].idx_melody[i]] for i in
                        range(len(self.array_songs[1].idx_melody))]
        print("LENGTH MELODY LIST A : " + str(len(melody_listA)))
        print("LENGTH MELODY LIST B : " + str(len(melody_listB)))
        channels_melodyB = [self.array_songs[1].channels[self.array_songs[1].idx_melody[i]] for i in
                            range(len(self.array_songs[1].idx_melody))]
        channels_melodyA = [self.array_songs[0].channels[self.array_songs[0].idx_melody[i]] for i in
                            range(len(self.array_songs[0].idx_melody))]

        instrument_numbers_melodyA = \
            [self.array_songs[0].instruments_list[self.array_songs[0].idx_melody[i]] for i in
             range(len(self.array_songs[0].idx_melody))]
        instrument_numbers_melodyB = \
            [self.array_songs[1].instruments_list[self.array_songs[1].idx_melody[i]] for i in
             range(len(self.array_songs[1].idx_melody))]

        converted_melodies = []
        for (i, melody) in enumerate(melody_listB):
            if i != 0:
                continue
            # PRENDRE MELODIE DE B ET APPLIQUER LE RYTHLE D'UNE MELODIE DE A

            # random_idx = random.randint(0,len(melody_listA)-1)

            for random_idx in range(len(melody_listA)):
                melodyA = melody_listA[random_idx]
                voices_add = musicfunctions.createChorus(melody, octaves_augmentation=1)
                voices_add_regularized = utils.deleteDuration0(
                    musicfunctions.rythmRegularization(melody, voices_add, 100, 1))
                converted_melody = musicfunctions.rythmicConversionMelodies(melodyA, voices_add_regularized)
                # channel = channels_melodyB[i]
                channel = channels_melodyA[random_idx]
                instrument_number = instrument_numbers_melodyA[random_idx]
                self.addInstrument(converted_melody, volume, instrument_number, channel)

            # random_idx = random.randint(0,len(melody_listA)-1)
            # rythmic_melody = melody_listA[random_idx]
            # voices_add = createChorus(rythmic_melody,octaves_augmentation=1)
            # voices_add_regularized = deleteDuration0(rythmRegularization(rythmic_melody,voices_add,100,1))
            # converted_melody = rythmicConversionMelodies(melody,voices_add_regularized)
            # converted_melodies += converted_melody
            # channel = channels_melodyB[i]
            # instrument_number = instrument_numbers_melodyB[i]
            # instrument_number = instrument_numbers_melodyA[random_idx]
            # # LA BEAUTE DES ERREURS
            # # self.addInstrument(converted_melodies,volume,instrument_number,channel)
            # self.addInstrument(converted_melody,volume,instrument_number,channel)

    # SON A ET SON B:
    # DRUMS A, TEMPO A MAIS MELODIE B (avec le rythme de A)
    def addInstrument(self, notes, volume, instrument_number, channel):
        self.tracks += [len(self.tracks)]
        self.track_instruments += [instrument_number]
        track = self.tracks[len(self.tracks) - 1]
        self.channels += [channel]
        self.track_notes += [notes]

    def generateSong(self):
        print("Generating music ...")
        print("Final Tracks : " + str(self.tracks))
        print("Final Channels : " + str(self.channels))
        print("Final midi Instruments : " + str(self.track_instruments))
        time = 0
        self.MyMIDI = MIDIFile(len(self.tracks))

        for i in range(len(self.tracks)):
            track_voices = self.tracks[i]
            channel_voices = self.channels[i]
            volume = 100  # 0-127, as per the MIDI standard
            self.MyMIDI.addTempo(track_voices, 0, self.BPM)
            notes = self.track_notes[i]
            self.MyMIDI.addProgramChange(track_voices, channel_voices, 0, self.track_instruments[i])
            for (i, info) in enumerate(notes):
                if info[1] != 0:
                    self.MyMIDI.addNote(track_voices, channel_voices, info[0],
                                        utils.convertTicks2Beat(info[3], self.ticks_per_beat),
                                        utils.convertTicks2Beat(info[1], self.ticks_per_beat), volume)
        with open(constants.PATH_GENERATED_SONG, "wb") as output_file:
            self.MyMIDI.writeFile(output_file)
        print("generateSong done")
        return constants.PATH_GENERATED_SONG

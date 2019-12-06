from mido import MidiFile, tempo2bpm
import constants


def printAll(path):
    mid = MidiFile(path)
    for i, track in enumerate(mid.tracks):
        for msg in track:
            print(msg)


def convertTempoToBPM(tempo):
    return tempo2bpm(tempo)


def convertTicks2Beat(ticks, ticks_per_beat):
    return ticks / ticks_per_beat


def findTempo(mid):
    tempo = 0
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
    return tempo


def convertInstrumentNumbers2InstrumentName(array_instrument_numbers):
    return [constants.instruments_dictionary[array_instrument_numbers[i]] for i in range(len(array_instrument_numbers))]


def convertTicks2Seconds(ticks,tricks_per_beat,tempo):
    return ticks*60/(tricks_per_beat*tempo)


def isGuitarOrBass(instrument_num):
    return (instrument_num >= 25) & (instrument_num <= 40)


def isChoir(instrument_num):
    return (instrument_num==53) | (instrument_num==54)


def isNoteInArray(info_note,array_notes):
    pitch = info_note[0]
    time = info_note[3]
    for info in array_notes:
        if (pitch == info[0]) & (time == info[3]):
            return True


def deleteDuration0(array_notes):
    return [notes for notes in array_notes if notes[1] != 0]


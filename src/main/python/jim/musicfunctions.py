from random import choices
import numpy
from jim import Song
from jim import SongGenerator
from jim import SongGeneratorMultiple
from jim import constants
from jim import utils


def createChorus(voices, octaves_augmentation):
    for i in range(1, octaves_augmentation+1):
        aux_voices = numpy.array(voices.copy())
        aux_voices[:, 0] += i*12
        voices = list(voices)+list(aux_voices)
    chorus = []
    index_list = list(range(len(voices)))
    len_chorus = int(len(voices)/9)
    index_voices = choices(index_list, k=3)
    count_len = 0
    if int(len(voices)/16) != 0:
        while count_len <= len_chorus:
            # print("Ici")
            for idx in index_voices:
                for i in range(int(len(voices)/16)):
                    chorus += [voices[(idx+i)%len(voices)]]
                    count_len += 1
    return chorus


def parseTrackLenOne(mid):
    event_list_per_track = []
    channels_list = []
    current_time = 0
    for msg in mid.tracks[0]:
        current_time += msg.time
        if msg.type == "note_on":
            note_channel = msg.channel
            if note_channel not in channels_list:
                channels_list += [note_channel]
                event_list_per_track += [[]]
            idx_channel = channels_list.index(msg.channel)
            msg.time = current_time
            event_list_per_track[idx_channel] += [msg]
        if msg.type == "note_off":
            if msg.channel not in channels_list:
                print(constants.RED_COLOR +"NOTE OFF WRONG CHANNEL ..." + constants.ENDC)
            else:
                msg.time = current_time
                event_list_per_track[channels_list.index(msg.channel)] += [msg]
    # LES VALEURS SONT ABOLUES ON CALCULE LES DELTA T:
    for event in event_list_per_track:
        delta_t = event[0].time
        for msg in event[1:len(event)]:
            msg.time -= delta_t
            delta_t += msg.time

    instruments_list = [1 for i in range(len(channels_list))]
    for msg in mid.tracks[0]:
        if msg.type == "program_change":
            if msg.channel in channels_list:
                idx_channel = channels_list.index(msg.channel)
                instruments_list[idx_channel] = msg.program+1
    return channels_list,instruments_list,event_list_per_track


def rythmRegularization(initial_notes,extracted_notes,repetition, overflow_max):
    count = 0
    notes_regularized = []
    count_total = 0
    for j in range(repetition):
        for (i,info) in enumerate(extracted_notes):
            # while(initial_notes[count][1] == 0):
            #     count += 1
            #     count %= len(initial_notes)
            new_note = [info[0],initial_notes[count][1],initial_notes[count][2],initial_notes[count][3]]
            if not utils.isNoteInArray(new_note,notes_regularized):
                notes_regularized += [new_note]
                count += 1
                count_total += 1
                if count_total == overflow_max*len(initial_notes):
                    break
                count %= len(initial_notes)
        if count_total == overflow_max*len(initial_notes):
            break
    return notes_regularized


def findInstrumentsAndDrumsAndChannels(mid):
    instruments_list = []
    channels_list = []
    track_count = -1
    idx_drums = []
    for i, track in enumerate(mid.tracks):
        note_activation = 0
        tmp_program_change = 1
        tmp_channel = 0
        for msg in track:
            if (msg.type == "note_on") & (not note_activation):
                note_activation = 1
                track_count += 1
                instruments_list += [1]
                instruments_list[track_count] = tmp_program_change
                channels_list  += [1]
                channels_list[track_count] = tmp_channel
                if msg.channel == 9:
                    idx_drums += [track_count]
            if msg.type == "program_change":
                tmp_program_change = int(msg.program) + 1
                tmp_channel = int(msg.channel)
    return instruments_list,idx_drums,channels_list


def findNotes(mid,channels,instruments_list):
    print("ICIIIII")
    track_count = -1
    channels = []
    if len(mid.tracks) > 1:
        event_list_per_track = mid.tracks
    else:
        channels,instruments_list, event_list_per_track = parseTrackLenOne(mid)
    notes = [[] for i in range(len(instruments_list))]

    for i, track in enumerate(event_list_per_track):
        dictionary_pressed_notes = {}
        note_activation, tmp_program_change,current_time,sum_velocity,nb_notes = 0,1,0,0,0
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on":
                if msg.velocity != 0:
                    if not note_activation:
                        note_activation = 1
                        track_count += 1
                    dictionary_pressed_notes[msg.note] = [current_time,msg.velocity]
                    sum_velocity += msg.velocity
                    nb_notes += 1
            if (msg.type == "note_off") | (msg.type == "note_on"):
                if ((msg.velocity == 0) & (msg.type=="note_on")) | (msg.type == "note_off"):
                    if msg.note in dictionary_pressed_notes.keys():
                        note = msg.note
                        duration = (current_time - dictionary_pressed_notes[msg.note][0])
                        velocity = dictionary_pressed_notes[msg.note][1]
                        time_played = dictionary_pressed_notes[msg.note][0]
                        notes[track_count] += [[note,duration,velocity,time_played]]
                        del dictionary_pressed_notes[msg.note]
    return notes,channels, instruments_list


def findMelody(notes,instruments_list,chosen_idx_melody):
    duration_list = []
    idx_potential_melody = []
    idx_melody = []

    for (i,notes_per_track) in enumerate(notes):
        dictionary_times_to_notes_and_chords, duration_total_track, count_total = {},0, 0
        for info in notes_per_track:
            duration_total_track += info[1]
            count_total += 1
            if info[3] not in dictionary_times_to_notes_and_chords.keys():
                dictionary_times_to_notes_and_chords[info[3]] = [info[0]]
            else:
                dictionary_times_to_notes_and_chords[info[3]] += [info[0]]
        list_chords_and_notes = []

        for key in dictionary_times_to_notes_and_chords:
            if dictionary_times_to_notes_and_chords[key] not in list_chords_and_notes:
                list_chords_and_notes += [dictionary_times_to_notes_and_chords[key]]
        list_notes_unique = [list_chords_and_notes[i] for i in range(len(list_chords_and_notes)) if len(list_chords_and_notes[i]) == 1]
        list_chords_unique = [list_chords_and_notes[i] for i in range(len(list_chords_and_notes)) if len(list_chords_and_notes[i]) > 1]

        all_notes = [key for key in dictionary_times_to_notes_and_chords.keys() if len(dictionary_times_to_notes_and_chords[key]) == 1]
        all_chords = [key for key in dictionary_times_to_notes_and_chords.keys() if len(dictionary_times_to_notes_and_chords[key]) > 1]

        if ((len(list_notes_unique) >= 7) & ((len(list_chords_unique)/(len(list_notes_unique)+len(list_chords_unique))) <= 0.2)) | (utils.isChoir(instruments_list[i])):
            if not utils.isGuitarOrBass(instruments_list[i]):
                color_write = constants.GREEN_COLOR
                idx_melody += [i]
            else:
                color_write = constants.YELLOW_COLOR
                idx_potential_melody += [i]
        else:
            color_write = constants.YELLOW_COLOR

        if i in chosen_idx_melody:
            color_write = constants.GREEN_COLOR
        print(color_write  + str(i) + ") " + constants.instruments_dictionary[instruments_list[i]] + " Numéro : " + str(instruments_list[i]) + constants.ENDC)
        print("Nb appuis différents : " + str(len(list_chords_and_notes)) + " on " + str(count_total))
        print(str(len(list_notes_unique)) + " notes on " + str(len(all_notes)))
        print(str(len(list_chords_unique)) + " chords on " + str(len(all_chords)))
        print("Different temporal variation (duration) : " +  str(len(numpy.unique(numpy.array(notes_per_track)[:,1]))) )
        if count_total!=0:
            duration_track_avg = duration_total_track/count_total
            print("Moyenne Durée : " + str(duration_track_avg))
            duration_list += [duration_track_avg]
        print("\n")

    if len(idx_melody) == 0:
        idx_melody = idx_potential_melody
    duration_avg = numpy.sum(duration_list)/len(notes)
    print("AVERAGE TOTAL DURATION : " + str(duration_avg) + " \n\n")

    return idx_melody


def rythmicConversionMelodies(melodyA, melodyB):
    melody_converted = []
    current_time = 0
    array_notes = []
    for i in range(len(melodyA)):
        idxb = i%len(melodyB)
        pitch = melodyB[idxb][0]
        duration = melodyA[i][1]
        volume = melodyA[i][2]
        time = melodyA[i][3]
        note = [pitch,duration,volume,time]
        if [pitch,time] not in array_notes:
            # note = [melodyA[i][0],melodyA[i][1],melodyA[i][2],melodyA[i][3]]
            melody_converted += [note]
            array_notes += [[pitch,time]]
    return melody_converted

    # melody_converted = []
    # current_time = 0
    # for i in range(len(melody_listB)):
    #     idx = i%len(melody_listA)
    #     time_played = melody_listB[i][3]
    #     if current_time <= time_played:
    #         melody_converted += [[melody_listA[idx][0],melody_listB[i][1],melody_listB[i][2],melody_listB[i][3]]]
    #         current_time = time_played
    # return melody_converted


def generateSongFromOneSong(path):
    song = Song.Song(path)
    songGenerator = SongGenerator.SongGenerator(song)
    songGenerator.addDrums()
    songGenerator.addMelody()
    songGenerator.addAccompaniment()  
    songGenerator.generateSong()


def generateSongFromTwoSongs(array_path_songs):
    if len(array_path_songs) >= 2:
        songA = Song.Song(array_path_songs[0])
        songB = Song.Song(array_path_songs[1])
        songGeneratorMultiple = SongGeneratorMultiple.SongGeneratorMultiple([songA,songB])
        songGeneratorMultiple.addDrums()
        songGeneratorMultiple.addMelody()
        songGeneratorMultiple.generateSong()





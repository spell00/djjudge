from jim import Song
from jim import SongGenerator
from jim import musicfunctions
from jim import utils

# PATH_FILE = 'midis/WhereIsMyMind.mid'
# PATH_FILE = 'midis/Holiday.mid'
# PATH_FILE = 'midis/snow.mid'
PATH_FILE = 'midis/Hit_the_Road_Jack.mid'
# PATH_FILE = 'midis/test2.mid'
# PATH_FILE = 'midis/Bohemian-Rhapsody-1.mid'
# PATH_FILE = 'midis/interstellar.mid'
# PATH_FILE = 'midis/hit_test2.mid'
# PATH_FILE = 'midis/cjamblues.mid'
# PATH_FILE = 'midis/cjamblueslogic.mid'
# PATH_FILE = 'midis/hit_test.mid'
# PATH_FILE = 'midis/broken_dreams.mid'
# PATH_FILE = 'midis/Canon_rock.mid'
# PATH_FILE = 'midis/moonlight1st.mid'
# PATH_FILEB = 'midis/georgia_in_my_mind.mid'
# PATH_FILE = 'midis/what_id_say.mid'
# PATH_FILE = 'midis/stairway_to_heaven.mid'
# PATH_FILE = 'midis/Purple_Haze.mid'
# PATH_FILE = 'midis/impromptu.mid'
# PATH_FILE = 'midis/alla_turca.mid'
# PATH_FILE = 'midis/cancan.mid'
# PATH_FILE = 'midis/imyours.mid'
# PATH_FILE = 'midis/test.mid'
# PATH_FILE = 'mid_test_tempo_120.mid'
# PATH_FILEB = 'midis/Kiki_s_Delivery_Service_Umi_no_Mieru_Gai.mid'
PATH_FILEB = 'midis/impromptu.mid'


utils.printAll(PATH_FILE)

print("generating song from one song")
musicfunctions.generateSongFromOneSong(PATH_FILE)
print("generating song from two songs")
musicfunctions.generateSongFromTwoSongs([PATH_FILE, PATH_FILEB])





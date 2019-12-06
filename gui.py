#!/usr/bin/python
import vlc
import platform
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QSlider, QLabel, \
    QTreeWidget, QTreeWidgetItem, QAction, QFileDialog
import Rating
import constants
import details
import musicfunctions


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.playlist = []
        # Create a basic vlc instance
        self.instance = vlc.Instance(["--audio-visual=visual", "--effect-list=spectrum"])
        self.mediaplayer = self.instance.media_player_new()
        self.title = 'Open Deep Jockey'
        self. width = 1024
        self. height = 512
        self. left = 256
        self.top = 256
        self.tune = []
        self.init_ui()

    def init_ui(self):
        self.initAct()
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        details.setColors(self)
        self.addControls()
        self.show()

    def addControls(self):
        # Create global container
        layout = QWidget(self)
        self.setCentralWidget(layout)
        layoutArea = QHBoxLayout()

        # create global layouts
        listArea = QVBoxLayout()  # left part of global
        playerArea = QHBoxLayout()  # central part of global
        controlArea = QVBoxLayout()  # right part of global

        # Add  to dedicated zone
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['File', 'Rating', 'State'])
        # self.tree.setSelectionMode(QAbstractItemView.MultiSelection)
        QTreeWidgetItem(self.tree, ['midis/Hit_the_Road_Jack.mid', '2', 'Example'])
        QTreeWidgetItem(self.tree, ['midis/Holiday.mid', '4', 'Example'])
        # create Tune buttun
        self.tuneBtn = QPushButton('Tune')  # Tune button
        # Add elements to list zone
        listArea.addWidget(self.tree)
        listArea.addWidget(self.tuneBtn)
        # Connect each signal to their appropriate function
        self.tuneBtn.clicked.connect(self.TuneHandler)

        # Create player img widget
        imgPlayer = QLabel(self)
        pixmap = QPixmap('base.png')
        imgPlayer.setPixmap(pixmap)
        # Create player slider
        # TODO
        # Add img to player zone
        player = QHBoxLayout()
        player.addWidget(imgPlayer)
        # Add to dedicated zone
        playerArea.addLayout(player)

        if platform.system() == "Linux":  # for Linux using the X Server
            self.mediaplayer.set_xwindow(int(imgPlayer.winId()))
        elif platform.system() == "Windows":  # for Windows
            self.mediaplayer.set_hwnd(int(imgPlayer.winId()))
        elif platform.system() == "Darwin":  # for MacOS
            self.mediaplayer.set_nsobject(int(imgPlayer.winId()))

        # create rating widget
        ratingWidget = Rating.RatingWidget()
        ratingWidget.value_updated.connect(
            lambda value: self.rateFile(value)
        )
        # create song controls
        volumeSlider = QSlider(Qt.Horizontal, self)
        volumeSlider.setFocusPolicy(Qt.NoFocus)
        volumeSlider.setValue(100)
        self.actionBtn = QPushButton('Play')  # action button : pause play
        self.clearBtn = QPushButton('Clear')  # stop button
        # Add buttons to song controls zone
        controls = QHBoxLayout()
        controls.addWidget(self.actionBtn)
        controls.addWidget(self.clearBtn)
        # Add to control zone
        controlArea.addWidget(ratingWidget)
        controlArea.addWidget(volumeSlider)  # add to controlArea directly to position it above
        controlArea.addLayout(controls)
        # Connect each signal to their appropriate function
        self.actionBtn.clicked.connect(self.actionHandler)
        self.clearBtn.clicked.connect(self.clearHandler)
        volumeSlider.valueChanged[int].connect(self.changeVolume)

        # Add to global layout
        layoutArea.addLayout(listArea)
        layoutArea.addLayout(playerArea)
        layoutArea.addLayout(controlArea)

        layout.setLayout(layoutArea)

        self.statusBar()
        # self.playlist.currentMediaChanged.connect(self.songChanged)

    def TuneHandler(self):
        # Tune buttun pushed
        print("TuneHandler")
        if len(self.tune) == 0:
            if self.tree.selectedItems():
                # select first file
                self.tune.append(self.tree.currentItem().data(0, 0))
                self.tree.currentItem().setData(2, 0, "To Tune")
                self.tree.clearSelection()
            else:
                self.statusBar().showMessage("No item to tune")
        elif len(self.tune) == 1:
            if self.tree.selectedItems():
                # select second file
                self.tune.append(self.tree.currentItem().data(0, 0))
                self.tree.currentItem().setData(2, 0, "To Tune")
                self.tree.clearSelection()
            else:
                # "Tuning 1 file"
                # TODO modulation
                self.tune.append(self.tree.currentItem().data(0, 0))
                musicfunctions.generateSongFromOneSong(self.tune[0])
                QTreeWidgetItem(self.tree, [constants.PATH_GENERATED_SONG, '', "Generated"])
                self.tree.clearSelection()
                self.tune = []
        elif len(self.tune) == 2:
            if self.tree.selectedItems():
                # Remove first, append new one
                self.tune[0] = self.tune[1]
                self.tune[1] = self.tree.selectedItems()[0].data(0, 0)
                self.tree.currentItem().setData(2, 0, "To Tune")
                self.tree.clearSelection()
            else:
                # "Tuning 2 files"
                # TODO modulation
                print("Tune selection" + str(self.tune))
                musicfunctions.generateSongFromTwoSongs([self.tune[0], self.tune[1]])
                QTreeWidgetItem(self.tree, [constants.PATH_GENERATED_SONG, '', "Generated"])
                self.tree.clearSelection()
                self.tune = []
        else:
            print("TuneHandler error")

    def actionHandler(self):
        if not self.tree.selectedItems():
            self.statusBar().showMessage("no media loaded")
            # shortcuts.openFile(self)
        else:
            if self.mediaplayer.is_playing():
                self.mediaplayer.pause()
                self.actionBtn.setText("Play")
            else:
                # getOpenFileName returns a tuple, so use only the actual file name
                media = self.instance.media_new(self.tree.currentItem().data(0, 0))
                # Put the media in the media player
                self.mediaplayer.set_media(media)
                self.mediaplayer.play()
                self.actionBtn.setText("Pause")

    def clearHandler(self):  # TODO
        self.mediaplayer.stop()
        self.tree.clearSelection()
        self.actionBtn.setText("Play")
        self.resize(1025, 512)  # shitty workaround
        self.resize(1024, 512)  # discard mediaplayer instance
        self.statusBar().showMessage("Stopped and cleared playlist")
        print("Stopped and cleared playlist")

    def changeVolume(self, value):
        self.mediaplayer.audio_set_volume(value)

    def songChanged(self, media):  # TODO
        if not media.isNull():
            url = media.canonicalUrl()
            self.statusBar().showMessage(self.playlist[0])

    def initAct(self):
        # Add to menu
        loadAction = QAction("Load file", self)
        loadAction.setShortcut("Ctrl+L")
        loadAction.setStatusTip('Load file')
        # connect to function
        loadAction.triggered.connect(self.openFile)

        # Add to menu
        exitAction = QAction("Exit App", self)
        exitAction.setShortcut("Ctrl+W")
        exitAction.setStatusTip('Leave The App')
        # connect to function
        exitAction.triggered.connect(details.on_exit)

        # Add file menu
        menubar = self.menuBar()
        appMenu = menubar.addMenu('App')
        appMenu.addAction(loadAction)
        appMenu.addAction(exitAction)

    def openFile(self):
        song = QFileDialog.getOpenFileName(None, 'Open Music Folder', '~', 'Sound Files(*.wav *.mid)')

        if song[0] != '':
            url = QUrl.fromLocalFile(song[0])
            if len(self.playlist) > 0:
                # self.playlist = []
                print("playlist not empty")
            # self.playlist.append(Media.Media(song[0]))
            QTreeWidgetItem(self.tree, [song[0], '', "Pending"])
            self.statusBar().showMessage(str(song[0])+" loaded")
            self.actionBtn.setText("Play")
            self.clearBtn.setText("Clear")
        else:
            self.statusBar().showMessage("nothing loaded")

    def rateFile(self, value):
        if self.tree.selectedItems():
            self.tree.currentItem().setData(1, 0, value + 1)  # TODO manage when no item
        else:
            self.statusBar().showMessage("no media selected")
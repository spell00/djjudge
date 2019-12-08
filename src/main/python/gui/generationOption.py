from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QHBoxLayout, QSlider, QVBoxLayout, QAction, QLabel
from PyQt5.QtCore import Qt

from gui import details
from jim import constants


class GenerationOption(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        print("GenerationOption")
        self.title = 'GenerationOption'
        self.width = 1024
        self.height = 512
        self.left = 256
        self.top = 256
        self.setWindowTitle(self.title)
        details.setColors(self)

        # Create global container
        layout = QWidget(self)
        self.setCentralWidget(layout)
        self.layoutArea = QVBoxLayout()

        self.set_layout()
        self.menu()

        layout.setLayout(self.layoutArea)

    def close_option(self):
        self.close()

    def menu(self):
        # create second layouts
        self.menuArea = QVBoxLayout()

        # create menu
        menubar = self.menuBar()
        # create Action
        self.closeBtn = QPushButton('close option windows')
        self.menuArea.addWidget(self.closeBtn)
        self.closeBtn.clicked.connect(self.close_option)
        exitAction = QAction("Exit Window", self)
        exitAction.setShortcut("Ctrl+W")
        exitAction.setStatusTip('Close the window')
        # connect to function
        exitAction.triggered.connect(self.close_option)
        appMenu = menubar.addMenu('Menu')
        appMenu.addAction(exitAction)

        # Add to global layout
        self.layoutArea.addLayout(self.menuArea)

    def set_layout(self):
        # create second layouts
        self.slidersArea = QHBoxLayout()

        # create third layouts
        self.melodyArea = QVBoxLayout()  # MELODY
        self.drumsArea = QVBoxLayout()  # DRUMS
        self.accompanimentArea = QVBoxLayout()  # ACCOMPANIMENT
        self.difficultyArea = QVBoxLayout()  # DIFFICULTY

        self.sliderMELODY()
        self.sliderDRUMS()
        self.sliderACCOMPANIMENT()
        self.sliderDIFFICULTY()

        # Add to secondary layout
        self.slidersArea.addLayout(self.melodyArea)
        self.slidersArea.addLayout(self.drumsArea)
        self.slidersArea.addLayout(self.accompanimentArea)
        self.slidersArea.addLayout(self.difficultyArea)

        # Add to global layout
        self.layoutArea.addLayout(self.slidersArea)

    def sliderMELODY(self):
        self.lb_MELODY = QLabel("VOLUME_MELODY")
        self.melodyArea.addWidget(self.lb_MELODY)

        self.lb_max_MELODY = QLabel("Max : "+str(constants.VOLUME_MELODY_MAX))
        self.melodyArea.addWidget(self.lb_max_MELODY)

        self.sl_MELODY = QSlider()
        self.sl_MELODY.setMinimum(constants.VOLUME_MELODY_MIN)
        self.sl_MELODY.setMaximum(constants.VOLUME_MELODY_MAX)
        self.sl_MELODY.setValue(constants.VOLUME_MELODY)
        self.sl_MELODY.setTickPosition(QSlider.TicksBelow)
        self.sl_MELODY.setTickInterval(constants.VOLUME_MELODY_MAX/4)
        self.melodyArea.addWidget(self.sl_MELODY)
        self.sl_MELODY.valueChanged.connect(self.valueMELODYchange)

        self.lb_min_MELODY = QLabel("min : "+str(constants.VOLUME_MELODY_MIN))
        self.melodyArea.addWidget(self.lb_min_MELODY)

        print("sliderMELODY")

    def sliderDRUMS(self):
        self.lb_DRUMS = QLabel("VOLUME_DRUMS")
        self.drumsArea.addWidget(self.lb_DRUMS)

        self.lb_max_DRUMS = QLabel("Max : "+str(constants.VOLUME_DRUMS_MAX))
        self.drumsArea.addWidget(self.lb_max_DRUMS)

        self.sl_DRUMS = QSlider()
        self.sl_DRUMS.setMinimum(constants.VOLUME_DRUMS_MIN)
        self.sl_DRUMS.setMaximum(constants.VOLUME_DRUMS_MAX)
        self.sl_DRUMS.setValue(constants.VOLUME_DRUMS)
        self.sl_DRUMS.setTickPosition(QSlider.TicksBelow)
        self.sl_DRUMS.setTickInterval(constants.VOLUME_DRUMS_MAX/4)
        self.drumsArea.addWidget(self.sl_DRUMS)
        self.sl_DRUMS.valueChanged.connect(self.valueDRUMSchange)

        self.lb_min_DRUMS = QLabel("min : "+str(constants.VOLUME_DRUMS_MIN))
        self.drumsArea.addWidget(self.lb_min_DRUMS)

        print("sliderDRUMS")

    def sliderACCOMPANIMENT(self):
        self.lb_ACCOMPANIMENT = QLabel("VOLUME_ACCOMPANIMENT")
        self.accompanimentArea.addWidget(self.lb_ACCOMPANIMENT)

        self.lb_max_ACCOMPANIMENT = QLabel("Max : "+str(constants.VOLUME_ACCOMPANIMENT_MAX))
        self.accompanimentArea.addWidget(self.lb_max_ACCOMPANIMENT)

        self.sl_ACCOMPANIMENT = QSlider()
        self.sl_ACCOMPANIMENT.setMinimum(constants.VOLUME_ACCOMPANIMENT_MIN)
        self.sl_ACCOMPANIMENT.setMaximum(constants.VOLUME_ACCOMPANIMENT_MAX)
        self.sl_ACCOMPANIMENT.setValue(constants.VOLUME_ACCOMPANIMENT)
        self.sl_ACCOMPANIMENT.setTickPosition(QSlider.TicksBelow)
        self.sl_ACCOMPANIMENT.setTickInterval(constants.VOLUME_ACCOMPANIMENT_MAX/4)
        self.accompanimentArea.addWidget(self.sl_ACCOMPANIMENT)
        self.sl_ACCOMPANIMENT.valueChanged.connect(self.valueACCOMPANIMENTchange)

        self.lb_min_ACCOMPANIMENT = QLabel("min : "+str(constants.VOLUME_ACCOMPANIMENT_MIN))
        self.accompanimentArea.addWidget(self.lb_min_ACCOMPANIMENT)

        print("sliderACCOMPANIMENT")

    def sliderDIFFICULTY(self):
        self.lb_DIFFICULTY = QLabel("DIFFICULTY_LVL")
        self.difficultyArea.addWidget(self.lb_DIFFICULTY)

        self.lb_max_DIFFICULTY = QLabel("Max : "+str(constants.DIFFICULTY_LVL_MAX))
        self.difficultyArea.addWidget(self.lb_max_DIFFICULTY)

        self.sl_DIFFICULTY = QSlider()
        self.sl_DIFFICULTY.setMinimum(constants.DIFFICULTY_LVL_MIN)
        self.sl_DIFFICULTY.setMaximum(constants.DIFFICULTY_LVL_MAX)
        self.sl_DIFFICULTY.setValue(constants.DIFFICULTY_LVL)
        self.sl_DIFFICULTY.setTickPosition(QSlider.TicksBelow)
        self.sl_DIFFICULTY.setTickInterval(constants.DIFFICULTY_LVL_MAX/4)
        self.difficultyArea.addWidget(self.sl_DIFFICULTY)
        self.sl_DIFFICULTY.valueChanged.connect(self.valueDIFFICULTYchange)

        self.lb_min_DIFFICULTY = QLabel("min : "+str(constants.DIFFICULTY_LVL_MIN))
        self.difficultyArea.addWidget(self.lb_min_DIFFICULTY)

        print("sliderDIFFICULTY")

    def valueMELODYchange(self):
        constants.VOLUME_MELODY = self.sl_MELODY.value()

    def valueDRUMSchange(self):
        constants.VOLUME_DRUMS = self.sl_DRUMS.value()

    def valueACCOMPANIMENTchange(self):
        constants.VOLUME_ACCOMPANIMENT = self.sl_ACCOMPANIMENT.value()

    def valueDIFFICULTYchange(self):
        constants.DIFFICULTY_LVL = self.sl_DIFFICULTY.value()

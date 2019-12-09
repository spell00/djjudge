from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QMainWindow, QProgressBar, QPushButton, QWidget, QVBoxLayout, QAction, QHBoxLayout, QDialog
import time
from gui import details
from gui.WaitingSpinnerWidget import QtWaitingSpinner
from sim.djjudge import cnn_log_likelihood

TIME_LIMIT = 5


class Djjudge(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        print("Djjudge")
        self.initUI()

    def initUI(self):
        self.title = 'Djjudge'
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
        self.loadingArea = QHBoxLayout()

        '''
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        self.loadingArea.addWidget(self.progress)
        '''

        self.w = QtWaitingSpinner()

        self.button = QPushButton('Start', self)
        self.button.move(0, 30)
        self.button.clicked.connect(self.onButtonClick)
        self.loadingArea.addWidget(self.button)

        # Add to global layout
        self.layoutArea.addLayout(self.loadingArea)

    def onButtonClick(self):
        self.w.start()
        self.calc = External(self.ctx)
        self.calc.countChanged.connect(self.onCountChanged)
        self.calc.start()

    def onCountChanged(self):
        self.w.stop

    def close_option(self):
        self.close()


class External(QThread):
    countChanged = pyqtSignal(int)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def run(self):
        value = int(cnn_log_likelihood.main(self.ctx))
        self.countChanged.emit(value)

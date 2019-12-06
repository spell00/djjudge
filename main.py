#!/usr/bin/python
import sys
import gui
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)

    gui = gui.App()

    sys.exit(app.exec_())


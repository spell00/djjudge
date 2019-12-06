import os

from PyQt5.QtCore import pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QHBoxLayout


class Media:
    def __init__(self, path):
        self._path = path
        self._rating = None
        self._state = None

    # Properties
    def get_path(self):
        return self._path

    def set_path(self, new_path):
        self._path = new_path

    def get_rating(self):
        return self._rating

    def set_rating(self, new_rating):
        self._rating = new_rating

    def get_state(self):
        return self._state

    def set_state(self, new_state):
        self._state = new_state

    @property
    def path(self):
        return self._path

    @property
    def max_rating(self):
        return 5

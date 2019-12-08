#!/usr/bin/python
import sys
from PyQt5.QtCore import QDirIterator, QUrl, Qt
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtMultimedia import QMediaContent
from PyQt5.QtWidgets import QFileDialog
from gui import generationOption
from fbs_runtime.application_context.PyQt5 import ApplicationContext


def setColors(self):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.black)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.black)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(235, 101, 54))
    palette.setColor(QPalette.Highlight, QColor(235, 101, 54))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    self.setPalette(palette)


def on_exit():
    print("on_exit")
    sys.exit()


def on_load():
    print("on_load")


def on_option():
    ctx = ApplicationContext()       # 1. Instantiate ApplicationContext
    option = generationOption.GenerationOption(ctx)
    option.show()
    print("on_option")


def on_djjudge():
    print("on_djjudge")


def folderIterator(self):
    folderChosen = QFileDialog.getOpenFileName(None, 'Open Music Folder', '~', 'All Files(*.*)')
    if folderChosen != None:
        it = QDirIterator(folderChosen)
        it.next()
        while it.hasNext():
            if it.fileInfo().isDir() == False and it.filePath() != '.':
                fInfo = it.fileInfo()
                if fInfo.suffix() in ('mp3', 'mid', 'wav', 'flac'):
                    self.playlist.append(QMediaContent(QUrl.fromLocalFile(it.filePath())))
            it.next()
        if it.fileInfo().isDir() == False and it.filePath() != '.':
            fInfo = it.fileInfo()
            if fInfo.suffix() in ('mp3', 'mid', 'wav', 'flac'):
                self.playlist.append(QMediaContent(QUrl.fromLocalFile(it.filePath())))
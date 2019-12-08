from PyQt5.QtWidgets import QMainWindow


class Djjudge(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def main(self):
        print("djjudge")

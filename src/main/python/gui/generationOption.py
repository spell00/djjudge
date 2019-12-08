from jim import constants



class generationOption(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def main(self):
        constants.VOLUME_MELODY = 127
        constants.VOLUME_DRUMS = 100
        constants.VOLUME_ACCOMPANIMENT = 60
        constants.DIFFICULTY_LVL = 20

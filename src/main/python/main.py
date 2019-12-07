#!/usr/bin/python
import sys
from gui import gui
from PyQt5.QtWidgets import QApplication
from fbs_runtime.application_context.PyQt5 import ApplicationContext

if __name__ == '__main__':
    ctx = ApplicationContext()       # 1. Instantiate ApplicationContext
    gui = gui.App(ctx)
    exit_code = ctx.app.exec_()      # 2. Invoke appctxt.app.exec_()

    sys.exit(exit_code)


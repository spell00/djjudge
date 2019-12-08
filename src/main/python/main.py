#!/usr/bin/python
import sys
from gui import gui
from sim import *
from fbs_runtime.application_context.PyQt5 import ApplicationContext

if __name__ == '__main__':
    ctx = ApplicationContext()       # 1. Instantiate ApplicationContext
    # gui = gui.App(ctx)
    exec(open("sim/djjudge/example_cnn.py").read())
    exit_code = ctx.app.exec_()      # 2. Invoke ctx.app.exec_()

    sys.exit(exit_code)

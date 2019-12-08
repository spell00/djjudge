#!/usr/bin/python
import sys
from gui import gui
from sim import example_cnn
from fbs_runtime.application_context.PyQt5 import ApplicationContext

if __name__ == '__main__':
    ctx = ApplicationContext()       # 1. Instantiate ApplicationContext
    # gui = gui.App(ctx)
    example_cnn.main(ctx)
    exit_code = ctx.app.exec_()      # 2. Invoke ctx.app.exec_()

    sys.exit(exit_code)

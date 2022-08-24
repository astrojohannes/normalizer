# -*- coding: utf-8 -*-

from PySide2 import QtCore, QtWidgets
import sys
import normalizer as n


if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Breeze')  # ['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion', 'Cleanlooks']
    start = n.start("dev/gui.ui")
    start.connect_buttons()
    sys.exit(app.exec_())


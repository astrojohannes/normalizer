import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QAction
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class CustomToolbar(NavigationToolbar):
    slicePressed = pyqtSignal()
    homePressed = pyqtSignal()

    def __init__(self, canvas, parent, coordinates=True):
        super().__init__(canvas, parent, coordinates)

        # define new button/action to slice spectrum based on current view
        self.slice_action = QAction('Slice', self)  # Create the action
        self.slice_action.triggered.connect(self.slice_spectrum)  # Connect to its slot

        # insert new slice button at a specific position, i.e. after pan
        actions = self.actions()  # Get all current actions
        pan_action_index = None  # search for the index of the 'Pan' action

        for i, action in enumerate(actions):
            if action.text() == 'Pan':
                pan_action_index = i
                break

        # insert the new action at given position
        if pan_action_index is not None:
            self.insertAction(actions[pan_action_index + 1], self.slice_action)
        else:
            self.addAction(self.slice_action)

    def slice_spectrum(self):
        self.slicePressed.emit()

    def home(self, *args, **kwargs):
        # Override the home function with custom behavior
        super().home(*args, **kwargs)
        self.homePressed.emit()

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectrum Normalizer Plot Window")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.figure = Figure(figsize=(7.5,12))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # Custom toolbar with a new button
        self.custom_toolbar = CustomToolbar(self.canvas, self)
        self.addToolBar(self.custom_toolbar)

        # Give access to Figure and Axes
        self.ax = self.init_plot()

    def init_plot(self):
        ax = [None,None]
        ax[0] = self.canvas.figure.add_subplot(211)
        ax[1] = self.canvas.figure.add_subplot(212, sharex = ax[0])
        self.canvas.draw()
        return ax

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QAction
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

class CustomToolbar(NavigationToolbar):
    slicePressed = pyqtSignal()
    resetPressed = pyqtSignal()
    homePressed = pyqtSignal()

    def __init__(self, canvas, parent, coordinates_callback):
        super().__init__(canvas, parent, coordinates_callback)

        self.coordinates_callback = coordinates_callback

        ##### Slice action
        # define new button/action to slice spectrum based on current view
        self.slice_action = QAction('Slice', self)  # Create the action
        self.slice_action.triggered.connect(self.slice_spectrum)  # Connect to its slot

        ##### Reset action
        # define new button/action to slice spectrum based on current view
        self.reset_action = QAction('Reset', self)
        self.reset_action.triggered.connect(self.reset_spectrum)

        # insert new slice toolbar action at a specific position, i.e. after pan
        actions = self.actions()  # Get all current actions
        pan_action_index = None  # search for the index of the 'Pan' action

        for i, action in enumerate(actions):
            if action.text() == 'Pan':
                pan_action_index = i
                break

        # insert the sclice action/button at given position
        if pan_action_index is not None:
            self.insertAction(actions[pan_action_index + 1], self.slice_action)
        else:
            self.addAction(self.slice_action)

        # Insert new reset button at end of toolbar
        self.insertAction(actions[-1], self.reset_action)

        #########################
        # Insert UNFLAG button
        self.unflag_button = QPushButton('UNFLAG')
        self.unflag_button.clicked.connect(self.activate_rectangle_selection_unflag)
        self.unflag_button.setStyleSheet("background-color: #a0a0a0;")
        self.addWidget(self.unflag_button)

        #########################
        # Insert BAD flag button
        self.badflag_button = QPushButton('Flag BAD')
        self.badflag_button.clicked.connect(self.activate_rectangle_selection_bad)
        self.badflag_button.setStyleSheet("background-color: #a0a0a0;")
        self.addWidget(self.badflag_button)

        #########################
        # Insert LINE flag button
        self.lineflag_button = QPushButton('Flag LINE')
        self.lineflag_button.clicked.connect(self.activate_rectangle_selection_line)
        self.lineflag_button.setStyleSheet("background-color: #a0a0a0;")
        self.addWidget(self.lineflag_button)

        self.x0, self.y0, self.x1, self.y1, self.flagtype = None, None, None, None, None

    def home(self, *args, **kwargs):
        # Override the home function with custom behavior
        super().home(*args, **kwargs)
        self.homePressed.emit()

    def slice_spectrum(self):
        self.slicePressed.emit()

    def reset_spectrum(self):
        self.resetPressed.emit()

    def deactivate_zoom_pan(self):
        # Deactivate zoom and pan if they are active
        if self._actions['zoom'].isChecked():
            self.zoom()
        if self._actions['pan'].isChecked():
            self.pan()

        self.badflag_button.setStyleSheet("background-color: #a0a0a0;")  # Change color to indicate active selection mode
        self.lineflag_button.setStyleSheet("background-color: #a0a0a0;")  # Change color to indicate active selection mode
        self.unflag_button.setStyleSheet("background-color: #a0a0a0;")  # Change color to indicate active selection mode


    def activate_rectangle_selection_unflag(self):
        self.deactivate_zoom_pan()

        self.flagtype = 'UNFLAG'

        # Connect the mouse press and release events to custom handlers
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # Before activating the rectangle selection, you might want to change the button style to indicate the mode is active
        self.unflag_button.setStyleSheet("background-color: blue;")  # Change color to indicate active selection mode
 
        # Change cursor to crosshair
        self.canvas.setCursor(Qt.CrossCursor)

    def activate_rectangle_selection_bad(self):
        self.deactivate_zoom_pan()

        self.flagtype = 'BAD'

        # Connect the mouse press and release events to custom handlers
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # Before activating the rectangle selection, you might want to change the button style to indicate the mode is active
        self.badflag_button.setStyleSheet("background-color: red;")  # Change color to indicate active selection mode
 
        # Change cursor to crosshair
        self.canvas.setCursor(Qt.CrossCursor)

    def activate_rectangle_selection_line(self):
        self.deactivate_zoom_pan()

        self.flagtype = 'LINE'

        # Connect the mouse press and release events to custom handlers
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)

        # Before activating the rectangle selection, you might want to change the button style to indicate the mode is active
        self.lineflag_button.setStyleSheet("background-color: #ccc;")  # Change color to indicate active selection mode
 
        # Change cursor to crosshair
        self.canvas.setCursor(Qt.CrossCursor)

    def on_press(self, event):
        # Record the start point (x0, y0)
        self.x0, self.y0 = event.xdata, event.ydata
        

    def on_release(self, event):
        # Record the end point (x1, y1) and trigger the user-defined action
        self.x1, self.y1 = event.xdata, event.ydata
        # Ensure the starting and ending points are defined
        if None not in (self.x0, self.y0, self.x1, self.y1):
            self.coordinates_callback(self.x0, self.y0, self.x1, self.y1, self.flagtype)
        # Disconnect the events after selection to prevent multiple connections
        self.canvas.mpl_disconnect(self.canvas.callbacks.connect('button_press_event', self.on_press))
        self.canvas.mpl_disconnect(self.canvas.callbacks.connect('button_release_event', self.on_release))

        # Reset cursor back to default
        self.canvas.unsetCursor()

        self.badflag_button.setStyleSheet("background-color: #a0a0a0;")  # Change color to indicate active selection mode
        self.lineflag_button.setStyleSheet("background-color: #a0a0a0;")  # Change color to indicate active selection mode
        self.unflag_button.setStyleSheet("background-color: #a0a0a0;")  # Change color to indicate active selection mode


class PlotWindow(QMainWindow):
    coordinatesSelected = pyqtSignal(float, float, float, float, str) 

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
        self.custom_toolbar = CustomToolbar(self.canvas, self, self.user_flagging)
        self.addToolBar(self.custom_toolbar)

        # Give access to Figure and Axes
        self.ax = self.init_plot()

    def init_plot(self):
        ax = [None,None]
        ax[0] = self.canvas.figure.add_subplot(211)
        ax[1] = self.canvas.figure.add_subplot(212, sharex = ax[0])
        self.canvas.draw()
        return ax

    def user_flagging(self, x0, y0, x1, y1, flagtype):
        self.coordinatesSelected.emit(x0, y0, x1, y1, flagtype)

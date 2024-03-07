import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QAction
from PyQt5.QtCore import pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt
import numpy as np

class CustomToolbar(NavigationToolbar):
    slicePressed = pyqtSignal()
    resetPressed = pyqtSignal()
    homePressed = pyqtSignal()
    rectangleSelected = pyqtSignal(float, float, float, float, str)

    def __init__(self, canvas, parent, ax):
        super().__init__(canvas, parent)

        self.ax = ax 

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

    #########################################################################################

    def deactivate_rectangle_selector(self):
        # General method to deactivate any existing rectangle selectors.
        if hasattr(self, 'rectangle_selector_unflag'):
            self.rectangle_selector_unflag.set_active(False)
        if hasattr(self, 'rectangle_selector_bad'):
            self.rectangle_selector_bad.set_active(False)
        if hasattr(self, 'rectangle_selector_line'):
            self.rectangle_selector_line.set_active(False)
        # Remove any existing rectangle by redrawing without it.
        self.canvas.draw_idle()


    def activate_rectangle_selection_unflag(self):
        self.deactivate_zoom_pan()
        self.flagtype = 'UNFLAG'
        current_ax = self.ax[0]

        # Disconnect existing events to avoid conflicts
        self.deactivate_rectangle_selector()

        # Create a new RectangleSelector and attach it to the current axes
        self.rectangle_selector_unflag = RectangleSelector(
            current_ax, self.on_select_rectangle,
            useblit=False, button=[1],  # Only left mouse button
            minspanx=5, minspany=5, spancoords='pixels',
            props={'alpha':0.2, 'facecolor':'blue'},
            interactive=True)

        # Update button styles and cursor.
        self.unflag_button.setStyleSheet("background-color: blue;")  # Change color to indicate active selection mode
        self.canvas.setCursor(Qt.CrossCursor)

    #########################################################################################

    def activate_rectangle_selection_bad(self):
        self.deactivate_zoom_pan()

        self.flagtype = 'BAD'
        current_ax = self.ax[0]

        # Disconnect existing events to avoid conflicts
        self.deactivate_rectangle_selector()

        # Create a new RectangleSelector and attach it to the current axes
        self.rectangle_selector_bad = RectangleSelector(
            current_ax, self.on_select_rectangle,
            useblit=False, button=[1],  # Only left mouse button
            minspanx=5, minspany=5, spancoords='pixels',
            props={'alpha':0.2, 'facecolor':'red'},
            interactive=True)

        # Update button styles and cursor.
        self.badflag_button.setStyleSheet("background-color: red;")  # Change color to indicate active selection mode
        self.canvas.setCursor(Qt.CrossCursor)

    def activate_rectangle_selection_line(self):
        self.deactivate_zoom_pan()

        self.flagtype = 'LINE'
        current_ax = self.ax[0]

        # Disconnect existing events to avoid conflicts
        self.deactivate_rectangle_selector()

        # Create a new RectangleSelector and attach it to the current axes
        self.rectangle_selector_line = RectangleSelector(
            current_ax, self.on_select_rectangle,
            useblit=False, button=[1],  # Only left mouse button
            minspanx=5, minspany=5, spancoords='pixels',
            props={'alpha':0.2, 'facecolor':'#a0a0a0'},
            interactive=True)

        # Update button styles and cursor.
        self.lineflag_button.setStyleSheet("background-color: #eee;")  # Change color to indicate active selection mode
        self.canvas.setCursor(Qt.CrossCursor)

    #########################################################################################

    # Add this new method to handle rectangle selection
    def on_select_rectangle(self, eclick, erelease):
        # Coordinates of the rectangle's corners
        self.x0, self.y0 = eclick.xdata, eclick.ydata
        self.x1, self.y1 = erelease.xdata, erelease.ydata

        # Emit the signal and transport coordinates and flagtype
        self.rectangleSelected.emit(self.x0, self.y0, self.x1, self.y1, self.flagtype)

        # Reset button styles and disconnect selector after drawing the rectangle
        self.reset_button_styles()
        self.deactivate_rectangle_selector()

    def reset_button_styles(self):
        # Reset the styles of all buttons to indicate that selection mode is off
        self.unflag_button.setStyleSheet("background-color: #a0a0a0;")
        self.badflag_button.setStyleSheet("background-color: #a0a0a0;")
        self.lineflag_button.setStyleSheet("background-color: #a0a0a0;")
        self.canvas.unsetCursor()
        self.canvas.draw_idle()


    #########################################################################################


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

        # Give access to Figure and Axes
        self.ax = self.init_plot()

        # Custom toolbar with a new button
        self.custom_toolbar = CustomToolbar(self.canvas, self, self.ax)
        self.addToolBar(self.custom_toolbar)

    def init_plot(self):
        ax = [None,None]
        ax[0] = self.canvas.figure.add_subplot(211)
        ax[1] = self.canvas.figure.add_subplot(212, sharex = ax[0])
        self.canvas.draw()
        return ax


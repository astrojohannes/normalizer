#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QAction
from PyQt5.QtCore import QFile, QIODevice, QObject, Qt, QSortFilterProxyModel, QDir, QCoreApplication, QEvent
from PyQt5.uic import loadUi
from PyQt5.QtGui import QFont, QClipboard, QKeySequence

import numpy as np
from numpy import inf, nan
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os,sys
import warnings
import ast

mpl.rcParams['text.usetex'] = False

import astropy
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning

from PyAstronomy import pyasl

from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline, interp1d
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.stats import norm

from mask_peaks import PeakMask
from exp_mask import exp_mask
from plotwindow import PlotWindow
from about import AboutWin


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# TableWidget event filter function
def table_key_press_event_filter(obj, event):
    if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
        # Get all selected items
        selectedItems = obj.selectedItems()

        # Clear the content of each selected item
        for item in selectedItems:
            item.setText('')  # Set the text of the item to an empty string

        return True  # Indicate that the event has been handled


    elif event.type() == QEvent.KeyPress and event.matches(QKeySequence.Copy):

        selected_ranges = obj.selectedRanges()
        if not selected_ranges:
            return

        table_data = []
        for selected_range in selected_ranges:
            for row in range(selected_range.topRow(), selected_range.bottomRow() + 1):
                row_data = []
                for col in range(selected_range.leftColumn(), selected_range.rightColumn() + 1):
                    item = obj.item(row, col)
                    row_data.append(item.text() if item else '')
                table_data.append('\t'.join(row_data))

        # Convert the selected table data to a string
        table_string = '\n'.join(table_data)

        # Copy the table data to the clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(table_string)

        return True

    elif event.type() == QEvent.KeyPress and event.matches(QKeySequence.Paste):
        # Handle pasting clipboard data into the table
        clipboard = QApplication.clipboard()
        clipboard_text = clipboard.text()

        if clipboard_text:
            selected_ranges = obj.selectedRanges()

            if selected_ranges:
                # Start pasting at the top-left corner of the selected range
                top_left_row = selected_ranges[0].topRow()
                top_left_col = selected_ranges[0].leftColumn()

                # Split clipboard data into rows and columns
                rows = clipboard_text.split('\n')
                for i, row_data in enumerate(rows):
                    columns = row_data.split('\t')
                    for j, text in enumerate(columns):
                        target_row = top_left_row + i
                        target_col = top_left_col + j

                        if target_row < obj.rowCount() and target_col < obj.columnCount():
                            item = obj.item(target_row, target_col)
                            if not item:
                                item = QTableWidgetItem()
                                obj.setItem(target_row, target_col, item)
                            item.setText(text)

        return True  # Indicate that the event has been handled

    return False  # Pass other events to the base class


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


class start(QMainWindow):

    def __init__(self, ui_file, parent=None):

        """ Initialize main window for user interactions
        """

        super(start, self).__init__(parent)
       
        self.gui = self     # this has historic reasons....
        loadUi(ui_file, self.gui)

        if not self.gui:
            print(loader.errorString())
            sys.exit(-1)

        # Create the menu bar
        menubar = self.gui.menuBar()
        menubar.setNativeMenuBar(True)

        # Create 'File' menu and add actions
        fileMenu = menubar.addMenu('&File')
        saveAction = QAction('&Save as...', self)
        exitAction = QAction('&Exit', self)
        saveAction.triggered.connect(lambda: self.saveFile(showfiledialogue=True))
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)

        # Create 'Help' menu and add actions
        helpMenu = menubar.addMenu('&Help')
        aboutAction = QAction('&About', self)
        aboutAction.triggered.connect(self.showAboutDialog)
        helpMenu.addAction(aboutAction)

        self.gui.label_8.setHidden(True)
        self.gui.lineEdit_interior_knots.setHidden(True)

        self.gui.label_4.setHidden(True)
        self.gui.lineEdit_fixed_width.setHidden(True)
        
        self.gui.show()

        # figure with 2 subplots
        self.plotwindow = PlotWindow()
        self.plotwindow.custom_toolbar.slicePressed.connect(self.on_slice_pressed)
        self.plotwindow.custom_toolbar.resetPressed.connect(self.on_reset_pressed)
        self.plotwindow.custom_toolbar.homePressed.connect(self.on_home_pressed)
        self.plotwindow.custom_toolbar.rectangleSelected.connect(self.on_coordinates_selected)

        self.gui.fig = self.plotwindow.figure  # Use the Figure from PlotWindow 
        self.gui.ax = self.plotwindow.ax  # Use the Axes from PlotWindow

        # set standard values
        self.gui.method='Polynomial'
        
        self.gui.lineEdit_degree.setText('0')
        self.gui.lineEdit_smooth.setText('200')
        self.gui.label_2.setVisible(False) 
        self.gui.lineEdit_smooth.setVisible(False)
        self.gui.label_9.setVisible(False) 
        self.gui.lineEdit_fixpoints.setVisible(False)
        self.gui.lineEdit_sigma_high.setText('5.0')
        self.gui.lineEdit_sigma_low.setText('2.5')
        self.gui.lineEdit_fixed_width.setText('10')
        self.gui.lineEdit_interior_knots.setText('200')
        self.gui.lineEdit_offset.setText('1.0')
        self.gui.lineEdit_auto_velocity_shift.setText('0.0')
        self.gui.lineEdit_auto_velocity_shift_lim1.setText('-400')
        self.gui.lineEdit_auto_velocity_shift_lim2.setText('400')
        self.gui.lineEdit_auto_velocity_shift_lim1.setReadOnly(True)
        self.gui.lineEdit_auto_velocity_shift_lim2.setReadOnly(True)
        self.gui.lineEdit_telluric_vrad.setReadOnly(True)

        self.vradshift_aa = []
        self.vradshift_kms = 0.0
        self.vradshift_applied = False
        self.snr = None
        self.rms = None
        self.renorm_factor_autovalue = 1.0
 
        self.gui.x=np.array([])     # origianl wavelength range
        self.gui.y=np.array([])     # original spectrum

        self.gui.xzoom=np.array([])     # zoomed-in wavelength range
        self.gui.yzoom=np.array([])     # zoomed-in original spectrum
        
        self.gui.yi=np.array([])    # smoothed, masked and interpolated spectrum used for normalisation (continuum fitting)
        self.gui.ynorm=np.array([]) # normalized array
        
        self.gui.xcurrent=np.array([])
        self.gui.ycurrent=np.array([])    # figure 0
        self.gui.ynormcurrent=np.array([])    # figure 1
        self.gui.ymaskedcurrent=np.array([])
        
        self.gui.mask=np.array([])
        self.gui.telluricmask=np.array([])

        self.mask_history = []  # keep mask history for undo function
        self.userpath = os.getcwd()
        
        self.gui.xlim_l_last=0
        self.gui.xlim_h_last=0

        # Identify the layout that contains the main parts, including the tableWidget
        layout = self.gui.horizontalLayout_main

        # Hide button to apply velo shift
        self.gui.pushButton_shift_spectrum.setVisible(True)
        self.gui.lineEdit_auto_velocity_shift.setVisible(True)

        # adjust table row height
        table = self.gui.tableWidget
        for i in range(table.rowCount()):
            table.setRowHeight(i, 20)  # Set each row's height to 40 pixels

        font = QFont()
        font.setPointSize(10)  # Set the font size to 10 points for table

        # Apply the font to the table
        table.setFont(font)

        # Install the event filter for the table
        self.gui.tableWidget.installEventFilter(self)

        self.c = 299792.458  # speed of light in km/s

    def closeEvent(self, event):
        # Close the plotwindow when the main window is about to close
        self.plotwindow.close()

    def showAboutDialog(self):
        # Create and show the About dialog
        dialog = AboutWin(self)
        dialog.exec_()  # Show the dialog modally

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def eventFilter(self, obj, event):
        if obj == self.tableWidget:
            return table_key_press_event_filter(obj, event)
        return super().eventFilter(obj, event)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def on_coordinates_selected(self, x0, y0, x1, y1, flagtype):

        #print(f"Coordinates selected: ({x0}, {y0}) to ({x1}, {y1})")
        tellurics = self.gui.lineEdit_telluric.text()
        x0_user = round(x0,3)
        x1_user = round(x1,3)

        if x0_user > x1_user:
            x0 = x1_user
            x1 = x0_user
        else:
            x0 = x0_user
            x1 = x1_user

        if flagtype=='BAD':
            self.mask_history.append(np.copy(self.gui.mask))
            self.gui.lineEdit_telluric.setText(f"{tellurics}, ({x0},{x1})")

        elif flagtype=='LINE':
            self.mask_history.append(np.copy(self.gui.mask))
            linewidth = round(0.5*(x1-x0),3)
            linecenter = round(x0 + linewidth,3)
            self.add_values_to_first_empty_row(self.gui.tableWidget, [linecenter, linewidth])


        elif flagtype == 'UNFLAG':
            self.mask_history.append(np.copy(self.gui.mask))

            # Unflag line flags:
            table = self.gui.tableWidget
            rowCount = table.rowCount()
        
            # List to store rows to remove and new rows to add
            rows_to_remove = []
            rows_to_add = []
        
            # Iterate through the rows in reverse order
            for row in range(rowCount -1, -1, -1):  # Start from the last row
       
                # Retrieve the value from the first and second columns of the current row
                item_center = table.item(row, 0)  # Center value in column 0
                item_width = table.item(row, 1)  # Width value in column 1
       
                # Skip if either item is empty or invalid
                if not item_center or not item_width:
                    continue
        
                try:
                    center_value = float(item_center.text())
                    width_value = float(item_width.text())
                except ValueError:
                    # Skip rows with non-numeric data
                    continue
 
                center_value = float(item_center.text())
                width_value = float(item_width.text())
        
                # Calculate the range
                range_start = center_value - width_value
                range_end = center_value + width_value
        
                # Check for overlap with the user's selection [x0, x1]
                if x0 <= range_start and x1 >= range_end:
                    # Case 0: Remove the entire range
                    rows_to_remove.append(row)
        
                elif x0 > range_start and x1 < range_end:
                    # Case 1: Exclude from the middle part of the existing range
                    rows_to_remove.append(row)
        
                    # Add the left part
                    new_width_left = (x0 - range_start) / 2.0
                    new_center_left = range_start + new_width_left
                    rows_to_add.append((row, new_center_left, new_width_left))
        
                    # Add the right part
                    new_width_right = (range_end - x1) / 2.0
                    new_center_right = x1 + new_width_right
                    rows_to_add.append((row + 1, new_center_right, new_width_right))
        
                elif x1 > range_end and x0 > range_start and x0 < range_end:
                    # Case 2: Remove the right part of the range
                    rows_to_remove.append(row)
                    new_width = (x0 - range_start) / 2.0
                    new_center = range_start + new_width
                    rows_to_add.append((row, new_center, new_width))

                elif x0 < range_start and x1 < range_end and x1 > range_start:
                    # Case 3: Remove the left part of the range
                    rows_to_remove.append(row)
                    new_width = (range_end - x1) / 2.0
                    new_center = x1 + new_width
                    rows_to_add.append((row, new_center, new_width))

            # Remove the rows after processing
            for row in rows_to_remove:
                table.removeRow(row)
 
            # Now add the new rows
            for row, center, width in rows_to_add:
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(str(round(center, 3))))
                table.setItem(row, 1, QTableWidgetItem(str(round(width, 3))))


            # unflag tellurics
            # Parse the intervals from the telluric line edit
            # and shift to the observed scale
            if self.gui.lineEdit_telluric.text().strip().strip(',') != '' and len(self.gui.lineEdit_telluric.text().strip().strip(','))>0:
                telluric_intervals = ast.literal_eval(self.gui.lineEdit_telluric.text().strip().strip(','))

                tellurics_lambda_corrfactor = self.calc_tellurics_lambda_corrfactor()
                if self.is_iterable(telluric_intervals[0]):
                    telluric_intervals = [(a * tellurics_lambda_corrfactor, b * tellurics_lambda_corrfactor) for a, b in telluric_intervals]
                else:
                    telluric_intervals = self.fix_telluric_intervals_notalist(telluric_intervals)
                

                # Filter and adjust the intervals based on the user selection
                updated_intervals = []
                for a, b in telluric_intervals:
                    if b <= x0 or a >= x1:
                        # Interval is completely outside user selection, keep it as is
                        updated_intervals.append((a, b))
                    elif a < x0 and b > x1:
                        # User selection is completely within the interval, split it into two
                        updated_intervals.append((a, x0))
                        updated_intervals.append((x1, b))
                    elif a < x0 <= b:
                        # Only the upper part of the interval overlaps with user selection
                        updated_intervals.append((a, x0))
                    elif a >= x0 and b > x1:
                        # Only the lower part of the interval overlaps with user selection
                        updated_intervals.append((x1, b))
                    # If the interval is entirely within the user selection, it gets removed (no action required)

                # Convert the updated intervals back to string format for the line edit
                updated_intervals_str = str(updated_intervals).replace(' ', '').replace('[','').replace(']','')  # Format it to match the original input format

                # Set the updated string back to the QLineEdit
                self.gui.lineEdit_telluric.setText(updated_intervals_str)
        else:
            print("Flag type unknown.")

        self.linetable_mask()
        self.fit_spline()




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def on_slice_pressed(self):

        xlim_l=float(self.gui.ax[0].get_xlim()[0])
        xlim_h=float(self.gui.ax[0].get_xlim()[1])

        # Conditions to get the indices within the desired limits
        indices = (self.gui.xcurrent >= xlim_l) & (self.gui.xcurrent <= xlim_h)

        self.gui.xcurrent=self.gui.xcurrent[indices]
        self.gui.ycurrent=self.gui.ycurrent[indices]

        self.gui.xlim_l_last=xlim_l
        self.gui.xlim_h_last=xlim_h
            
        # remove fit
        self.gui.yi=np.array([])
        self.gui.ynorm = np.array([])
        self.gui.ynormcurrent=np.array([])
        self.gui.knots_x = np.array([])
        self.gui.knots_y = np.array([])

        # preserve masks

        # handle strange case where length of self.gui.mask is greater by 1 than length of original self.gui.xcurrent
        if len(self.gui.mask)-len(indices):
            self.gui.mask = self.gui.mask[:-1]

        if len(self.gui.telluricmask)>0: self.gui.telluricmask=self.gui.telluricmask[indices]
        if len(self.gui.mask)>0: self.gui.mask=self.gui.mask[indices]
        if len(self.gui.ymaskedcurrent)>0: self.gui.ymaskedcurrent[indices]

        self.fit_spline(showfit=True)

 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
        
    def on_reset_pressed(self):
        
        self.gui.xcurrent=self.gui.x
        self.gui.ycurrent=self.gui.y
        self.gui.ymaskedcurrent=self.gui.y
        self.gui.ynorm = np.array([])
        self.gui.ynormcurrent=self.gui.ynorm
        self.gui.yi=np.array([])
        #self.gui.mask=np.array([])
        self.gui.telluricmask=np.array([])
        
        if len(self.gui.x)>0:
            self.gui.ax[0].set_xlim([min(self.gui.x),max(self.gui.x)])        
            self.gui.ax[1].set_xlim([min(self.gui.x),max(self.gui.x)])        

            self.gui.xlim_h_last=max(self.gui.x)
            self.gui.xlim_l_last=min(self.gui.x)
        else:
            self.gui.xlim_h_last=0
            self.gui.xlim_l_last=0

        self.gui.knots_x = np.array([])
        self.gui.knots_y = np.array([])

        self.gui.ax[0].cla()
        self.gui.ax[1].cla()

        # Standard telluric absorption bands
        self.create_telluric_mask()

        self.linetable_mask()

        self.make_fig(0)

        #self.gui.tableWidget.clearContents()

        #self.fit_spline(showfit=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def on_home_pressed(self, *args, **kwargs):
        
        """ add some functionality to matplotlib's home button
        
        """

        self.gui.ax[0].set_xlim([min(self.gui.xcurrent),max(self.gui.xcurrent)])
        self.gui.ax[1].set_xlim([min(self.gui.xcurrent),max(self.gui.xcurrent)])
        
        self.fit_spline(showfit=True)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def connect_buttons(self):

        """ Connect the GUI buttons with slots
        """
        self.gui.pushButton_openfits.clicked.connect(self.selectFile)
        self.gui.comboBox_method.currentIndexChanged.connect(self.method_changed) 
        self.gui.pushButton_normalize.clicked.connect(lambda _: self.fit_spline(showfit=True))
        self.gui.pushButton_identify_mask_lines.clicked.connect(self.identify_mask)
        self.gui.pushButton_linetable_mask.clicked.connect(self.linetable_mask)
        self.gui.pushButton_savefits.clicked.connect(self.saveFile)
        self.gui.pushButton_determine_rad_velocity.clicked.connect(self.determine_rad_velocity)
        self.gui.pushButton_undo.clicked.connect(self.undo_mask_change)
        self.gui.pushButton_shift_spectrum.clicked.connect(self.apply_velocity_shift)
        self.gui.pushButton_renorm_factor_auto.clicked.connect(self.renorm_auto)

#                                 IO PART
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def selectFile(self):

        """ Opens a window for the user to select a FITS files
        """
 
        if self.gui.lbl_fname.text() is not None and os.path.isfile(self.gui.lbl_fname.text()):
            mydir = os.path.dirname(self.gui.lbl_fname.text())
        else:
            mydir = QDir.currentPath()

        filename,_ = QFileDialog.getOpenFileName(None,'Open FITS spectrum', self.userpath, self.tr("*.fits"))
 
        if filename == '':
            # cancel was clicked
            return

        if mydir == QDir.currentPath():
            self.gui.lbl_fname.setText(os.path.basename(filename))
        else:
            self.gui.lbl_fname.setText(filename)

        self.readfits(filename)
        self.make_fig(0)
        self.on_reset_pressed()
 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def saveFile(self, showfiledialogue=False):
        """
        if self.gui.lbl_fname.text() is not None and os.path.isfile(self.gui.lbl_fname.text()):
            mydir = os.path.dirname(self.gui.lbl_fname.text())
        else:
            mydir = QtCore.QDir.currentPath()

        filename,_ = QFileDialog.getSaveFileName(None,'Save to FITS', self.tr("(*.fits)"))
        """
        if len(self.gui.x) > 0:
            if showfiledialogue:
                # Set the options for the dialog
                options = QFileDialog.Options()
                filename, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", self.userpath,
                                                  "All Files (*);;FITS Files (*.fits)", 
                                                  options=options)

            else:
                file_basename = os.path.basename(self.gui.lbl_fname.text())
                # Split the filename and extension
                file_name_without_extension, file_extension = os.path.splitext(file_basename)

                filename = f"{file_name_without_extension}_{str(int(self.gui.xlim_l_last))}-{str(int(self.gui.xlim_h_last))}.fits"

            # Check if the filename is valid, e.g. when user pressed "Cancel" the filename is empty
            if filename != '':
                self.gui.lbl_fname2.setText(filename)
                self.writefits(filename)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def zoom_fig(self,wave_min,wave_max):
        if wave_min < min(self.gui.x): wave_min = min(self.gui.x)
        if wave_max > max(self.gui.x): wave_max = max(self.gui.x)
        self.gui.ax[0].set_xlim([wave_min,wave_max])
        #self.gui.ax[1].set_xlim()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def tr(self, text):
        return QObject.tr(self, text)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def remove_spikes(self, indata, n_std=10):
        """
        Iteratively remove spikes from the edges of a 1D spectrum if they exceed a certain threshold.
    
        Parameters:
            data: numpy array of 1D spectrum.
            n_std: number of standard deviations above/below the mean to use as a threshold for spikes.
        
        Returns:
            The cleaned spectrum.
        """

        data = self.smooth(indata,100)

        # Calculate mean and standard deviation of the data
        mean = np.mean(data)
        std = np.std(data)
    
        # Define threshold
        threshold_upper = mean + n_std * std
        threshold_lower = mean - n_std * std
    
        # Define indices for iterative edge spike check
        start_index = 0
        end_index = len(data) - 1
    
        # Check for spikes from edges towards the center
        while start_index < end_index:
            # Check start
            if data[start_index] > threshold_upper or data[start_index] < threshold_lower:
                start_index += 1
            else:
                # If no spike is found, stop the iteration
                break
            
        # Start from the end and move towards the start
        while end_index >= start_index:
            # Check end
            if data[end_index] > threshold_upper or data[end_index] < threshold_lower:
                end_index -= 1
            else:
                # If no spike is found, stop the iteration
                break
            
        # Return the cleaned spectrum
        return start_index, end_index+1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def find_telluric_intervals(self, filename):
        # Read the data from the file
        data = pd.read_csv(filename, sep='\t', comment='#', header=None, 
                       names=["Wavelength", "Molecular Absorption", "Ozone", 
                              "Rayleigh Scattering", "Aerosol Extinction"])

        # Multiply the Wavelength column by 10
        data["Wavelength"] = data["Wavelength"] * 10

        # Filter the data where Molecular Absorption < 0.99
        filtered_data = data[data["Molecular Absorption"] < 0.99]

        # Find continuous regions/intervals
        intervals = []
        start = end = np.round(filtered_data['Wavelength'].iloc[0],3)
    
        for i in range(1, len(filtered_data)):
            if filtered_data['Wavelength'].iloc[i] - end > 1:  # Change this as per the gap in your data
                intervals.append((start, end))
                start = np.round(filtered_data['Wavelength'].iloc[i],3)
            end = np.round(filtered_data['Wavelength'].iloc[i],3)
    
        intervals.append((start, end))

        return intervals

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def is_iterable(self,obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def fix_telluric_intervals_notalist(self, telluric_intervals):
        telluric_intervals = [list(telluric_intervals)]
        telluric_intervals.append([0.0001,0.0002] )

        return telluric_intervals

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def create_telluric_mask(self):

        telluric_intervals = self.find_telluric_intervals(os.environ['NORMALIZER_DIR']+"/skycalc_molec_abs.txt")
        self.gui.lineEdit_telluric.setText(', '.join('({}, {})'.format(*t) for t in telluric_intervals))

        if self.gui.lineEdit_telluric.text().strip().strip(',') != '' and len(self.gui.lineEdit_telluric.text().strip().strip(','))>0:

            # Initialize new array for telluric mask
            self.gui.telluricmask = np.full_like(self.gui.xcurrent, 1)
            
            # Parse the intervals from the telluric line edit
            # and shift to the observed scale
            telluric_intervals = ast.literal_eval(self.gui.lineEdit_telluric.text().strip().strip(','))
            tellurics_lambda_corrfactor = self.calc_tellurics_lambda_corrfactor()
            if self.is_iterable(telluric_intervals[0]):
                telluric_intervals = [(a * tellurics_lambda_corrfactor, b * tellurics_lambda_corrfactor) for a, b in telluric_intervals]
            else:
                telluric_intervals = self.fix_telluric_intervals_notalist(telluric_intervals)

            for a, b in telluric_intervals:
                self.gui.telluricmask[(self.gui.xcurrent >= a) & (self.gui.xcurrent <= b)] = 0

            return self.gui.telluricmask, telluric_intervals
        else:
            self.gui.telluricmask = np.array([])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def calc_tellurics_lambda_corrfactor(self):
        try:
            user_vrad = float(self.gui.lineEdit_telluric_vrad.text())
        except:
            user_vrad = 0.0
        tellurics_lambda_corrfactor = 1.0 / self.doppler_shift(user_vrad)

        return tellurics_lambda_corrfactor

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def doppler_shift(self, v_rad):
        """
        Calculate the wavelength shift from v_rad

        Parameters
        ----------
        v_rad : float
            The radial velocity (in same units as speed of light).

        Returns
        -------
        float
            The shifted wavelength.
        """

        # Scale factor: sf = lambda_observed/lambda_emitted

        sf = np.sqrt((self.c + v_rad)/(self.c - v_rad))

        return sf


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def readfits(self, fitsfile, hduid=0):

        # store the path of the input filename for later use when saving
        self.userpath = os.path.dirname(fitsfile)

        # show filename in Plot Window header
        self.plotwindow.setWindowTitle(os.path.basename(fitsfile))

        self.gui.ax[0].cla()
        self.gui.ax[1].cla()
   
        mask = np.array([])
 
        # Read the input file
        if fitsfile.lower().endswith('.fits') or fitsfile.lower().endswith('.fit') or fitsfile.lower().endswith('.tfit') or fitsfile.lower().endswith('.tfits'):
            # If the input file is a FITS file, read it using Astropy's fits module
            hdus = fits.open(fitsfile)
            # Check if a BinTableHDU exists in the list of HDUs
            binary_table_hdu = None
            for hdu in hdus:
                if isinstance(hdu, fits.BinTableHDU):
                    binary_table_hdu = hdu
                    break
            # Load binary table data if it exists
            if binary_table_hdu is not None:
                # Read some header info
                current_header = hdus[1].header
                if all(key in current_header for key in ['SN_RVAPL', 'SN_RVVAL']):
                    if bool(current_header['SN_RVAPL']):
                        self.gui.lineEdit_telluric_vrad.setReadOnly(False)
                        self.gui.lineEdit_telluric_vrad.setText(str(current_header['SN_RVVAL']))
                        self.gui.lineEdit_telluric_vrad.setReadOnly(True)
 
                    else:
                        self.gui.lineEdit_telluric_vrad.setReadOnly(False) 
                        self.gui.lineEdit_telluric_vrad.setText('0.0')
                        self.gui.lineEdit_telluric_vrad.setReadOnly(True)
 
                self.create_telluric_mask()

                available_cols = binary_table_hdu.data.columns.names  # Get the list of available columns
                
                # Check if 'Wavelength', 'Normalized_Flux' or 'Flux' columns are available
                wavecolumnnotfound = False
                if 'wavelength' in available_cols: tablename_wave = 'wavelength'
                elif 'Wavelength' in available_cols: tablename_wave = 'Wavelength'
                elif 'wave' in available_cols: tablename_wave = 'wave'
                elif 'Wave' in available_cols: tablename_wave = 'Wave'
                elif 'WAVE' in available_cols: tablename_wave = 'WAVE'
                elif 'WAVELENGTH' in available_cols: tablename_wave = 'WAVELENGTH'
                else:
                    wavecolumnnotfound = True

                fluxcolumnnotfound = False
                if 'normalized_flux' in available_cols: tablename_flux = 'normalized_flux'
                elif 'Normalized_Flux' in available_cols: tablename_flux = 'Normalized_Flux'
                elif 'flux' in available_cols: tablename_flux = 'flux'
                elif 'Flux' in available_cols: tablename_flux = 'Flux'
                elif 'FLUX' in available_cols: tablename_flux = 'FLUX'
                elif 'NORMALIZED_FLUX' in available_cols: tablename_flux = 'NORMALIZED_FLUX'
                else:
                    fluxcolumnnotfound = True

                maskcolumnnotfound = False
                if 'Mask' in available_cols: tablename_mask = 'Mask'
                elif 'mask' in available_cols: tablename_mask = 'mask'
                elif 'MASK' in available_cols: tablename_mask = 'MASK'
                else:
                    maskcolumnnotfound = True

                if not wavecolumnnotfound and not fluxcolumnnotfound:
                    x = binary_table_hdu.data[tablename_wave].ravel()
                    y = binary_table_hdu.data[tablename_flux].ravel()

                    if not maskcolumnnotfound:
                        mask = np.array(binary_table_hdu.data[tablename_mask].ravel(),dtype=int)

                    hdr = binary_table_hdu.header
                else:
                    # check primary header for WSTART, WEND and DELTA_W
                    current_header = hdus[0].header
                    if all(key in current_header for key in ['WSTART', 'WEND', 'DELTA_W', 'N_PIXELS']) and fluxcolumnnotfound == False and wavecolumnnotfound == True:
                        wstart = float(current_header['WSTART'])
                        delta_w = float(current_header['DELTA_W'])
                        n_pix = int(current_header['N_PIXELS'])

                        # Calculate the wavelengths
                        new_wave = [wstart + int(i) * delta_w for i in range(n_pix)]

                        x = np.array(new_wave,dtype=np.float64)
                        y = np.array(binary_table_hdu.data[tablename_flux].ravel(),dtype=np.float64)

                        if not maskcolumnnotfound:
                            mask = np.array(binary_table_hdu.data[tablename_mask].ravel(),dtype=int)

                        hdr = binary_table_hdu.header
                    else:
                        # Let user select columns
                        print("At least one of the expected columns 'Wave', 'Wavelength', 'Flux' or 'Normalized_Flux' are not available.")
                        print("Available columns are: ", available_cols)
                        x_col = input("Please enter the column name to use as Wavelength: ")
                        while x_col not in available_cols:
                            print(f"{x_col} is not a valid column name. Please enter a valid column name for Wavelength: ")
                            x_col = input()
                    
                        y_col = input("Please enter the column name to use as Normalized_Flux: ")
                        while y_col not in available_cols or y_col == x_col:
                            if y_col == x_col:
                                print(f"{y_col} is already used as Wavelength. Please enter a different column name for Normalized_Flux: ")
                            else:
                                print(f"{y_col} is not a valid column name. Please enter a valid column name for Normalized_Flux: ")
                            y_col = input()
                    
                        x = binary_table_hdu.data[x_col].ravel()
                        y = binary_table_hdu.data[y_col].ravel()
                        hdr = binary_table_hdu.header

            else:
                # Otherwise, assume a regular FITS file and load image data
                hdr = hdus[hduid].header
                img = hdus[hduid].data

                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter('always', AstropyWarning)  # Change 'always' to 'error' to turn warnings into exceptions

                    wcs = WCS(hdr)

                    # Check if any relevant warnings were caught
                    for warning in caught_warnings:
                        if isinstance(warning.message, astropy.wcs.FITSFixedWarning):
                            print("Caught an astropy WCS warning. Trying to fix header now.")
                            hdr = self.clean_wcs_for_1d_fits_header(fitsfile)

                # Recreate the WCS object with the potentially updated header
                wcs = WCS(hdr)

                if int(hdr['NAXIS']) == 2:
                    y = img.sum(axis=0)   # summing up along spatial direction
                    x = wcs.all_pix2world([(x, 0) for x in range(len(y))], 0)
                    x = np.delete(x, 1, axis=1)
                    x = x.flatten()
                    y = y.flatten()
                elif int(hdr['NAXIS']) == 1:
                    y = img
                    crpix1 = hdr['CRPIX1']  # Pixel coordinate of reference point
                    crval1 = hdr['CRVAL1']  # Coordinate value at reference point
                    cdelt1 = hdr['CDELT1']  # Coordinate increment at reference point
                    x = crval1 + cdelt1 * (np.arange(len(y)) - (crpix1 - 1))

        else:
            # Otherwise, assume the input file is an ASCII file and read it using numpy
            if fitsfile.lower().endswith('.csv'):
                delimiter = ','
            else:
                delimiter = '\t'
            data = np.loadtxt(fitsfile, delimiter=delimiter, comments='#')
            x = data[:, 0]
            y = data[:, 1]
            hdr = fits.Header()

        # detect spikes at edges
        start, end = self.remove_spikes(y)

        # check for constant flux at edges and truncate
        _, start_idx, end_idx = self.truncate_constant_edges(y, 5)
        x = x[start_idx:end_idx]
        y = y[start_idx:end_idx]

        # remove nans
        nan_indices_x = np.isnan(x)
        x = x[~nan_indices_x]
        y = y[~nan_indices_x]

        nan_indices_y = np.isnan(y)
        x = x[~nan_indices_y]
        y = y[~nan_indices_y]

        # Save re-usable quantities in global variables
        self.gui.x = x
        self.gui.y = y
        self.gui.xcurrent = x
        self.gui.ycurrent = y
        self.gui.ymaskedcurrent = y
        self.gui.hdr = hdr
        self.gui.ynorm = np.array([])
        self.gui.ynormcurrent = np.array([])
        self.gui.yi = np.array([])
        self.gui.knots_x = np.array([])
        self.gui.knots_y = np.array([])

        # recover mask
        # mask value 0...BAD/Telluric
        # mask value 1...Line
        # mask value 2...Continuum
        normalizer_mask = np.ones_like(mask)
        normalizer_mask[mask==2] = 0
        self.gui.mask = np.array(normalizer_mask,dtype=bool)
 
        self.gui.xlim_h_last=0
        self.gui.xlim_l_last=0

        #self.apply_mask()

        #self.fit_spline()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    # Define function to clean WCS keywords for 1D data and return the updated header
    def clean_wcs_for_1d_fits_header(self,fits_file):
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            header = hdul[0].header  # Assuming the primary header contains the WCS info
        
            # Check if the data is 1-dimensional
            if hdul[0].data.ndim == 1:
                print(f"Data in {fits_file} is 1-dimensional, but header suggests otherwise!")
            
                # List of WCS keywords that are irrelevant for 1D data
                wcs_keywords_2d_3d = ['CRPIX2', 'CRPIX3', 'CDELT2', 'CDELT3',
                                  'CRVAL2', 'CRVAL3', 'CTYPE2', 'CTYPE3',
                                  'CUNIT2', 'CUNIT3', 'CROTA2', 'CROTA3',
                                  'PC2_', 'PC3_', 'CD2_', 'CD3_', 'PV2_', 'PV3_']
            
                # Create a copy of the header before modification
                updated_header = header.copy()
            
                # Remove the irrelevant WCS keywords from the copy
                for key in wcs_keywords_2d_3d:
                    # Check each pattern and remove if present
                    for k in list(updated_header.keys()):
                        if k.startswith(key):
                            del updated_header[k]
                        
            return updated_header

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def writefits(self, fitsfile, hduid=0):
        """ Save normalized spectrum and mask in
            fits file
        """

     
        export_mask = np.array(np.copy(self.gui.mask),dtype=int)
        export_mask[export_mask==True] = 1
        export_mask[export_mask==False] = 2

        if export_mask.shape[0] == 0:
            print("No user line mask available for export.")
            export_mask = self.gui.telluricmask
        else:
            export_mask[self.gui.telluricmask==0] = 0

        # Create columns for wavelength (x-axis) and normalized flux (y-axis)
        col1 = fits.Column(name='Wavelength', format='E', array=self.gui.xcurrent)
        col2 = fits.Column(name='Normalized_Flux', format='E', array=self.gui.ynormcurrent)
        col3 = fits.Column(name='Mask', format='E', array=export_mask)
    
        # Create a ColDefs object from the columns
        cols = fits.ColDefs([col1, col2, col3])
    
        # Create a BinTableHDU object from the ColDefs object
        tbhdu = fits.BinTableHDU.from_columns(cols)
    
        # Create a new header for the output file
        """
        new_hdr = fits.Header()
        new_hdr['SIMPLE'] = True
        new_hdr['BITPIX'] = -32
        new_hdr['NAXIS'] = 2
        new_hdr['EXTEND'] = True
        """   

        # Modify the header
        new_hdr = tbhdu.header  # Use the existing header of tbhdu
        new_hdr.add_comment("Spectrum processed with Spectrum Normalizer")
        new_hdr.add_comment("https://github.com/astrojohannes/normalizer")
    
        vrad = self.gui.lineEdit_telluric_vrad.text()
        if self.vradshift_applied:
            try:
                vrad = str(round(float(vrad),2))
                new_hdr['SN_RVVAL'] = (vrad, 'Rad. vel. in km/s from Spec. Normalizer')
                new_hdr['SN_RVAPL'] = ('True', 'If True, spectrum was shifted by SN_RVVAL')
            except:
                pass 
        else:
            try:
                vrad = str(round(float(vrad),2))
                new_hdr['SN_RVVAL'] = (vrad, 'Rad. vel. in km/s from Spec. Normalizer')
                new_hdr['SN_RVAPL'] = ('False', 'If True, spectrum was shifted by SN_RVVAL')
            except:
                pass

        if self.snr != None:
            new_hdr['SN_SNR'] = (self.snr, 'signal-to-noise ratio measured by Normalizer')
        if self.rms != None:
            new_hdr['SN_RMS'] = (self.rms, 'continuum r.m.s. measured by Normalizer')
 
        # Write the data to the output FITS file
        tbhdu.writeto(self.gui.lbl_fname2.text(), overwrite=True)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
#                                  PLOTTING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def make_fig(self, figid, showfit=True):
      """ make the 2-panel matplotlib figure """
      if len(self.gui.xcurrent)>0:
        fig = self.gui.fig
        ax = self.gui.ax

        fig.subplots_adjust(left=0.11, bottom=0.1, right=0.98, top=0.98, wspace=0, hspace=0)

        ax[figid].cla()

        if self.gui.xlim_l_last > 0:
            ax[0].set_xlim([self.gui.xlim_l_last, self.gui.xlim_h_last])
            ax[1].set_xlim([self.gui.xlim_l_last, self.gui.xlim_h_last])
        else:
            ax[0].set_xlim([min(self.gui.xcurrent), max(self.gui.xcurrent)])
            ax[1].set_xlim([min(self.gui.xcurrent), max(self.gui.xcurrent)])

        if figid == 0:
            x, y = self.gui.xcurrent, self.gui.ycurrent
        else:
            x, y = self.gui.xcurrent, self.gui.ynormcurrent

        col = ['b', 'g']

        ax[figid].step(x, y, color=col[figid], lw=1.0)

        if figid == 0:
            ax[0].text(0.03, 0.1, 'Original', fontsize=20, transform=ax[0].transAxes)
        elif figid == 1:
            ax[1].text(0.03, 0.1, 'Normalized', fontsize=20, transform=ax[1].transAxes)


        # plot telluric regions
        if figid == 0 and showfit:

            if self.gui.lineEdit_telluric.text().strip().strip(',') != '' and len(self.gui.lineEdit_telluric.text().strip().strip(','))>0:

                _, telluric_intervals = self.create_telluric_mask()

                if not self.is_iterable(telluric_intervals[0]):
                    telluric_intervals = self.fix_telluric_intervals_notalist(telluric_intervals)

                for start, end in telluric_intervals:
                    ax[0].fill_betweenx(np.linspace(min(y), max(y), 10), start, end, color='red', alpha=0.2, label='telluric')

        ax[1].set_xlabel('wavelength [AA]')

        # update the line mask table and get mask edges
        mask_edges = self.update_line_mask_table()
        
        # plot mask in top panel
        if len(self.gui.mask)>0 and len(mask_edges) > 0:
            ii=0
            while ii < len(mask_edges)-1:
                xx1 = float(x[mask_edges[ii]])
                if mask_edges[ii+1] < len(x):
                    xx2 = float(x[mask_edges[ii+1]])
                else:
                    xx2 = xx1
                ii+=2
                ylim_l=np.nanmin(self.gui.ycurrent)
                ylim_h=np.nanmax(self.gui.ycurrent)
                yy=np.linspace(ylim_l,ylim_h,10)
                ax[0].fill_betweenx(yy,xx1,xx2, color='lightgray', alpha=0.2, label='line')


        # Plot continuum fit in figure 0
        if len(self.gui.yi) > 0 and figid == 0 and showfit:
            # plot unmasked ranges (=continuum)
            ax[0].plot(self.gui.xcurrent,self.gui.ymaskedcurrent,'orange', lw=1.5)
            ax[0].plot(x, self.gui.yi, color='r', lw=3.0, label='continuum')




        # plot a horizontal line at 1
        ax[1].axhline(y=1.0, linestyle='--', color='k', lw=2)

        # plot the LSQ method spline knots in upper panel
        if self.gui.comboBox_method.currentText()=='LSQUnivariateSpline':
            # Convert list of knots to a numpy array for efficient operations
            if len(self.gui.knots_x)>0 and len(self.gui.yi)>0:
                knots_x = np.array(self.gui.knots_x)
                knots_y = np.array(self.gui.knots_y)

                # Plot all points at once
                ax[0].scatter(knots_x, knots_y, c='r')

        self.plotwindow.canvas.draw()
        self.plotwindow.show()
        self.plotwindow.activateWindow()
        self.gui.fig.tight_layout()
 

#                                  FITTING 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def find_nearest_idx(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def fit_spline(self, showfit=True):
        """ Fit a spline using different methods and the user input parameters """
        if len(self.gui.xcurrent) > 1:
            # update telluric mask
            self.create_telluric_mask()
    
            # apply both, telluric and line mask --> updates ymaskedcurrent
            self.apply_mask()
    
            # FITTING part
            x, y = self.gui.xcurrent, self.gui.ycurrent
    
            xlim_l = float(self.gui.ax[0].get_xlim()[0])
            xlim_h = float(self.gui.ax[0].get_xlim()[1])
    
            self.gui.xlim_l_last = xlim_l
            self.gui.xlim_h_last = xlim_h
 
            if self.gui.method == 'Polynomial':

                degree = int(self.gui.lineEdit_degree.text())
                if degree <=0: degree=0
                elif degree>9: degree=9

                w = np.isnan(self.gui.ymaskedcurrent)
                weights = np.ones_like(y)
                weights[w] = 0.0

                # Fit a weighted polynomial
                coefficients = np.polyfit(x, y, degree, w=weights)
                yi = np.polyval(coefficients, x)
 
            else:
                k = int(self.gui.lineEdit_degree.text())
                if k <= 1: k = 1
                elif k > 5: k = 5
    
                # read user input: smoothing parameter
                s = int(self.gui.lineEdit_smooth.text())
    
                w = np.isnan(self.gui.ymaskedcurrent)
                weights = np.ones_like(y)
                weights[w] = 0.0
    
                if self.gui.comboBox_method.currentText() == 'LSQUnivariateSpline':
                    every = int(self.gui.lineEdit_interior_knots.text())
    
                    t_indices = np.arange(1, len(x), every)
                    final_indices = t_indices[~np.isnan(self.gui.ymaskedcurrent[t_indices])]
                    self.gui.knots_x = x[final_indices]
                    t = self.gui.knots_x
    
                    spl = self.gui.method(x, y, t, k=k, w=weights, check_finite=False, ext=3)
                    yi = np.copy(spl(x)).flatten()
    
                    if len(self.gui.ymaskedcurrent) > 0 and len(yi) > 0:
                        self.gui.knots_y = np.copy(spl(self.gui.knots_x)).flatten()
    
                else:
                    scalingfactor = np.nanmean(y)
                    if scalingfactor < 1000:
                        scalingfactor = 1.0
                    spl = self.gui.method(x, y/scalingfactor, k=k, w=weights/scalingfactor, s=s, check_finite=True, ext=3)
                    spl.set_smoothing_factor(s)
                    yi = scalingfactor * np.copy(spl(x)).flatten()
    
            if self.gui.lineEdit_offset.text().strip() == '':
                offs = 1.0
            else:
                try:
                    offs = float(self.gui.lineEdit_offset.text())
                except:
                    offs = 1.0
    
            ynorm = np.divide(y, np.array(yi), where=np.array(yi) != 0)
            ynorm *= 1.0 / offs
    
            self.gui.yi = np.array(yi)
    
            if not len(self.gui.ynorm) > 0:
                self.gui.ynorm = ynorm
            self.gui.ynormcurrent = ynorm
    
            if len(self.gui.telluricmask) > 0:
                self.gui.yi[self.gui.telluricmask == 0] = np.nan
                self.gui.ynormcurrent[self.gui.telluricmask == 0] = 1.0
    
            self.make_fig(0, showfit=showfit)
            self.make_fig(1, showfit=showfit)

        # calculate RMS of current plot zoom
        self.calc_rms()




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def truncate_constant_edges(self, arr, min_repeat=10):
        # Convert to a numpy array for easier handling
        arr = np.array(arr)
    
        # Initialize variables to store the length of constant sequences at start and end
        start_constant_length = 0
        end_constant_length = 0
    
        # Function to compare considering NaN values
        def equals_handling_nan(a, b):
            return (a == b) | (np.isnan(a) & np.isnan(b))

        # Check for constant values at the start of the array
        for i in range(1, len(arr)):
            if equals_handling_nan(arr[i], arr[0]):
                start_constant_length = i
            else:
                break  # Stop at the first non-constant value

        # Update start index if there are more than min_repeat constant values at the start
        start_index = start_constant_length + 1 if start_constant_length >= min_repeat else 0
    
        # Check for constant values at the end of the array
        for i in range(len(arr) - 2, -1, -1):
            if equals_handling_nan(arr[i], arr[-1]):
                end_constant_length = (len(arr) - i - 1)
            else:
                break  # Stop at the first non-constant value

        # Update end index if there are more than min_repeat constant values at the end
        end_index = len(arr) - (end_constant_length + 1) if end_constant_length >= min_repeat else len(arr)
    
        # Return the truncated array based on the found indices
        return arr[start_index:end_index], start_index, end_index

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def nan_helper(self,y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def method_changed(self):

        """ Update plot using fitting method selected by user
        """
        current_method=self.gui.comboBox_method.currentText()
        if current_method=='UnivariateSpline':
            self.gui.label_9.setHidden(False)
            self.gui.lineEdit_fixpoints.setHidden(False)
            self.gui.label_8.setHidden(True)
            self.gui.lineEdit_interior_knots.setHidden(True)
            
            self.gui.label_2.setHidden(False)
            self.gui.lineEdit_smooth.setHidden(False)
            
            self.gui.method=UnivariateSpline
        elif current_method=='LSQUnivariateSpline':
            self.gui.label_9.setHidden(False)
            self.gui.lineEdit_fixpoints.setHidden(False) 
            self.gui.label_2.setHidden(True)
            self.gui.lineEdit_smooth.setHidden(True)

            self.gui.label_8.setHidden(False)
            self.gui.lineEdit_interior_knots.setHidden(False)
            
            self.gui.method=LSQUnivariateSpline
 
        elif current_method == 'Polynomial':
            self.gui.label_9.setHidden(True)
            self.gui.lineEdit_fixpoints.setHidden(True)
            self.gui.label_8.setHidden(True)
            self.gui.lineEdit_interior_knots.setHidden(True)
            self.gui.label_2.setHidden(True)
            self.gui.lineEdit_smooth.setHidden(True)
        
            self.gui.method = 'Polynomial'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def undo_mask_change(self):
        if self.mask_history:
            self.gui.mask = self.mask_history.pop()
            self.fit_spline(showfit=True)
            self.make_fig(0)
            self.make_fig(1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def identify_mask(self):
        """ Identify/mask lines in normed spectrum using rms measured over smoothed spectrum iteratively until change in rms is less than 1 percent """
   
        self.mask_history.append(np.copy(self.gui.mask))
 
        if len(self.gui.ynormcurrent) > 1:
            # Check if the radio_maskmode_add is checked
            if self.gui.radio_maskmode_add.isChecked():
                # If so, initialize the new mask with the current mask, if it exists
                if len(self.gui.mask) > 0:
                    new_mask = np.copy(self.gui.mask)
                else:
                    new_mask = np.array([False for _ in self.gui.ynormcurrent], dtype=bool)
            else:
                # Otherwise, start with a fresh mask
                new_mask = np.array([False for _ in self.gui.ynormcurrent], dtype=bool)
    
            x = np.copy(self.gui.xcurrent)
            y = np.copy(self.gui.ynormcurrent)
    
            if len(self.gui.telluricmask) > 0:
                xfit = x[self.gui.telluricmask != 0]
                yfit = y[self.gui.telluricmask != 0]
                telluric_mask = self.gui.telluricmask[self.gui.telluricmask != 0].astype(bool)
            else:
                xfit = x
                yfit = y
                telluric_mask = np.ones_like(xfit, dtype=bool)  # No telluric mask, so all ones
    
            sigma_high = float(self.gui.lineEdit_sigma_high.text())
            sigma_low = float(self.gui.lineEdit_sigma_low.text())
    
            # Define chunk size in Ångströms
            chunk_size = 20.0
   
            # Process the spectrum in chunks
            start_idx = 0
            while start_idx < len(xfit):
                # Find the end index where the x-axis value is 200 Ångströms away from the start
                end_idx = start_idx
                while end_idx < len(xfit) and (xfit[end_idx] - xfit[start_idx]) <= chunk_size:
                    end_idx += 1
    
                x_chunk = xfit[start_idx:end_idx]
                y_chunk = yfit[start_idx:end_idx]
                mask_chunk = new_mask[start_idx:end_idx]
                telluric_chunk = telluric_mask[start_idx:end_idx]
    
                if len(x_chunk) < 2:
                    start_idx = end_idx
                    continue
    
                # Fit a polynomial to the chunk for baseline correction
                coefficients = np.polyfit(x_chunk, y_chunk, 1)
                polynomial = np.poly1d(coefficients)
                fitted_y = polynomial(x_chunk)
                normed_y = y_chunk / fitted_y
    
                try:
                    # Peak finding using the sigma thresholds
                    masker_high = PeakMask(normed_y, sigma_smooth=1, sigma_threshold=sigma_high, rms_tolerance=0.1, maxnumber_iterations=2)
                    masker_low = PeakMask(normed_y, sigma_smooth=1, sigma_threshold=sigma_low, rms_tolerance=0.1, maxnumber_iterations=2)
    
                    mask_high, _, rms_high = masker_high.create_mask()
                    mask_low, _, rms_low = masker_low.create_mask()

                    # Apply telluric mask
                    mask_high &= telluric_chunk
                    mask_low &= telluric_chunk
    
                    # Ensure the masks are not empty before proceeding
                    if np.any(mask_high) or np.any(mask_low):
                        iter_mask = np.array(exp_mask(mask_high, constraint=mask_low, keep_mask=True, quiet=True), dtype=bool)
                        mask_chunk = np.logical_or(mask_chunk, iter_mask)
                    else:
                        print(f"No valid peaks found in chunk {x_chunk[0]}-{x_chunk[-1]}. Skipping.")
    
                    # Update the chunk in the overall mask
                    new_mask[start_idx:end_idx] = mask_chunk
    
                except Exception as e:
                    print(f"Masking failure in line identification for chunk {x_chunk[0]}-{x_chunk[-1]}:\n{e}")
                    # Move to the next chunk
                    start_idx = end_idx
                    continue
    
                # Move to the next chunk
                start_idx = end_idx
    
            # Update the global mask with the new mask
            self.gui.mask = new_mask
   
            # Perform the final fit and update the plot
            self.fit_spline(showfit=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def renorm_auto(self):
        self.calc_rms(autorenorm=True)
        self.fit_spline(showfit=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
   
    def calc_rms(self,autorenorm=False):

        if len(self.gui.mask)>1:
            # Calculate RMS of ycurrent where the mask is not NaN
            valid_ycurrent = self.gui.ycurrent[~self.gui.mask & ~np.isnan(self.gui.ycurrent)]
            number_contpoints = len(valid_ycurrent)
            if len(valid_ycurrent) > 0:
                rms = round(np.sqrt(np.mean((valid_ycurrent - np.mean(valid_ycurrent))**2)),4)
                snr = round(1.0/rms,2)
                self.snr = snr
                self.rms = rms
                mean = round((1.0/np.mean(valid_ycurrent)),4)
                if autorenorm:
                    self.gui.lineEdit_offset.setText(str(mean))
                print(f"Normalized continuum (# points={number_contpoints}): MEAN={mean} RMS={rms}, SNR={snr}")
            else:
                print("No valid data to calculate RMS.")

    

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def update_line_mask_table(self):
        if len(self.gui.mask)>0:
            mask_edges = self.find_mask_edges()
            return mask_edges

        return np.array([])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def find_mask_edges(self):
        last_val=False
        edges=[]
        for ii in range(len(self.gui.mask)):
            this_val=self.gui.mask[ii]
            if this_val!=last_val:
                edges.append(ii)
                last_val=this_val
                
        if self.gui.mask[-1]==True:
            edges.append(len(self.gui.mask))

        # Calculate the center and width of each region
        # and write these into the first and second column of the table, respectively
        self.gui.tableWidget.setRowCount(0)
        self.gui.tableWidget.setRowCount(10000)

        for i in range(0, len(edges), 2):
            try:
                start, end = edges[i], edges[i+1]
                center = round((self.gui.xcurrent[start] + self.gui.xcurrent[end]) / 2,3)
                width = round((self.gui.xcurrent[end] - self.gui.xcurrent[start])/2.0,3)

                # Write center and width into the table
                self.gui.tableWidget.setItem(i // 2, 0, QTableWidgetItem(str(center)))
                self.gui.tableWidget.setItem(i // 2, 1, QTableWidgetItem(str(width)))
            except IndexError:
                break
            
        return np.array(edges)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def add_values_to_first_empty_row(self, table_widget, values):
        # Step 1: Determine the first empty row
        row_count = table_widget.rowCount()
        first_empty_row = None
        for i in range(row_count):
            if table_widget.item(i, 0) is None:  # Assuming checking the first column for emptiness
                first_empty_row = i
                break
    
        # Step 2: Insert a new row if necessary
        if first_empty_row is None:  # All existing rows are filled
            first_empty_row = row_count
            table_widget.insertRow(first_empty_row)
    
        # Step 3: Set the item values for the cells in this row
        for column_index, value in enumerate(values):
            table_widget.setItem(first_empty_row, column_index, QTableWidgetItem(str(value)))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def apply_mask(self):

        """ combine the flags of line and telluric masks
            and produce an array self.gui.ymaskedcurrent
            where all flagged regions are set to NaN
        """

        # Backup original masks
        original_mask = np.copy(self.gui.mask) if len(self.gui.mask) > 0 else None
        original_telluric_mask = np.copy(self.gui.telluricmask) if len(self.gui.telluricmask) > 0 else None

        # Apply initial masks to the data
        if original_mask is not None or original_telluric_mask is not None:
            ymasked = np.copy(self.gui.ycurrent)
            if original_mask is not None:
                ymasked[original_mask] = np.nan
            if original_telluric_mask is not None:
                ymasked[original_telluric_mask == 0] = np.nan
            self.gui.ymaskedcurrent = np.array(ymasked)
        else:
            self.gui.ymaskedcurrent = np.copy(self.gui.ycurrent)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def linetable_mask(self, dofit=True):
        
        """ use the user lines from the table to mask spectral regions
            when doing the continuum normalization
        """
        
        self.gui.mask=np.array([])
        self.fit_spline()
        
        lines = []
        widths = []
        table=self.gui.tableWidget
        for row in range(table.rowCount()):
            l = table.item(row, 0)
            w = table.item(row, 1)
            try:
                c = float(l.text().strip()) if (l is not None) and (l.text().strip() != '') else np.nan
                w = float(w.text().strip()) if (w is not None) and (w.text().strip() != '') else np.nan
            except:
                c = np.nan
            if c != np.nan and c>0 and w != np.nan and w>0:
                lines.append(c)
                widths.append(w)

        idx=[]
        # loop and mask absorption lines
        for ii,c in enumerate(lines):
            ww=widths[ii]
            l = c-ww
            r = c+ww
            this_idx=np.where((self.gui.xcurrent >= l) & (self.gui.xcurrent <= r))
            idx.extend(this_idx)
        if len(idx)>1:
            idx=np.hstack(idx)

        idx=np.unique(np.array(idx,dtype=int))

        self.gui.mask = np.array([False for x in self.gui.xcurrent],dtype=bool)
        self.gui.mask[idx] = True

        self.fit_spline(showfit=True)
        self.make_fig(0)
        self.make_fig(1)


#                             RADIAL VELOCITY
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


    def gaussian(self, x, a, b, c):
        """ Gaussian function to fit the peak """
        return a * np.exp(-(x - b)**2 / (2 * c**2))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def read_template(self, file_path):
        #data = np.genfromtxt(file_path, dtype=[('waveobs', np.float64), ('flux', np.float64), ('err', np.float64)])
        data = np.genfromtxt(file_path, dtype=[('waveobs', np.float64), ('flux', np.float64)])
        return data.view(np.recarray)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def optimize_velocity_shift(self, dw, df, tw, tf):

        # Initial parameters for the coarse search
        steps = [
            {'drv': 1.0, 'vmin': -400.0, 'vmax': 400.0},
            {'drv': 0.5, 'vmin': -250.0, 'vmax': 250.0},
            {'drv': 0.1, 'vmin': -100.0, 'vmax': 100.0}
        ]
        

        rv_corrections_kms = []
        i=0
        for step in steps:
            i+=1
            drv = step['drv']
            vmin = step['vmin']
            vmax = step['vmax']
            

            if i==1: # First iteration on coarse grid
                # Carry out the cross-correlation with the current parameters
                rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, vmin, vmax, drv, skipedge=0)

                # Find the index of the maximum cross-correlation function
                maxind = np.argmax(cc)

                # Convert the radial velocity shift to a shift in wavelength
                mean_wavelength = np.mean(dw)
                rw = mean_wavelength - mean_wavelength * 1.0/self.doppler_shift(rv)

                # Print the current maximum shift
                print(f"Coarse iteration {i}: Cross-correlation function is maximized at dRV = {rv[maxind]} km/s")
                print(f"Coarse iteration {i}: Cross-correlation function is maximized at dRV = {rw[maxind]} AA")

                rvmax = rv[maxind]
                rv_corrections_kms.append(rvmax)
                dw = dw * 1.0/self.doppler_shift(rvmax)


            elif i==2: # Second iteration on coarse grid

                # Carry out the cross-correlation with the current parameters
                rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, vmin, vmax, drv, skipedge=0)

                # Find the index of the maximum cross-correlation function
                maxind = np.argmax(cc)

                # Convert the radial velocity shift to a shift in wavelength
                mean_wavelength = np.mean(dw)
                rw = mean_wavelength - mean_wavelength * 1.0/self.doppler_shift(rv)

                # Print the current maximum shift
                print(f"Coarse iteration {i}: Cross-correlation function is maximized at dRV = {rv[maxind]} km/s")
                print(f"Coarse iteration {i}: Cross-correlation function is maximized at dRV = {rw[maxind]} AA")

                rvmax = rv[maxind]
                rv_corrections_kms.append(rvmax)
                dw = dw * 1.0/self.doppler_shift(rvmax)


            else: # Refine
                # Update vmin,vmax
                deltashift=99999
                while deltashift > 0.05 and i < 20:
                    rvmax_prev = rvmax
                    # Carry out the cross-correlation with the current parameters
                    rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, vmin, vmax, drv, skipedge=0)
    
                    # Find the index of the maximum cross-correlation function
                    maxind = np.argmax(cc)
    
                    # Convert the radial velocity shift to a shift in wavelength
                    mean_wavelength = np.mean(dw)
                    rw = mean_wavelength - mean_wavelength * 1.0/self.doppler_shift(rv)
    
                    # Print the current maximum shift
                    print(f"Fine iteration {i}: Cross-correlation function is maximized at dRV = {rv[maxind]} km/s")
                    print(f"Fine iteration {i}: Cross-correlation function is maximized at dRV = {rw[maxind]} AA")
          
                    rvmax = rv[maxind]
                    rv_corrections_kms.append(rvmax)
                    dw = dw * 1.0/self.doppler_shift(rvmax)

                    deltashift = abs(rvmax)

                    i+=1

        rv_correction_kms = np.sum(rv_corrections_kms)

        return rv,rw,cc,maxind,rv_correction_kms

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def determine_rad_velocity(self):
        c = self.c

        if len(self.gui.ynormcurrent)>0:
            waveobs,flux=np.array(self.gui.xcurrent,dtype=np.float64),np.array(self.gui.ynormcurrent,dtype=np.float64)
        else:
            self.gui.lineEdit_auto_velocity_shift.setText('0.0')
            return

        err=np.array([0.0 for a in range(len(self.gui.xcurrent))],dtype=np.float64)
        this_arr=np.vstack((waveobs,flux,err))
        this_spec=np.core.records.fromrecords(this_arr.T, names='waveobs,flux,err')
        this_spec=this_spec.view(np.recarray)

        # Check and interpolate missing data for this_spec
        mask = np.isfinite(this_spec.flux)
        this_spec.flux = np.interp(this_spec.waveobs, this_spec.waveobs[mask], this_spec.flux[mask])

        #--- Radial Velocity determination with template -------------------------------
        #template = self.read_template(os.environ['NORMALIZER_DIR']+"/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
        template = self.read_template(os.environ['NORMALIZER_DIR']+"/templates/Kitt_Peak_Flux_Atlas_2005.csv")
        #template['waveobs']=template['waveobs']*10.0    # convert to Angstroem

        # Check and interpolate missing data for template
        mask = np.isfinite(template.flux)
        template.flux = np.interp(template.waveobs, template.waveobs[mask], template.flux[mask])

        # Template
        tw = template.waveobs 
        tf = template.flux

        # Data
        dw = this_spec.waveobs
        df = this_spec.flux

        # Plot template and data
        """
        fig = plt.figure()
        plt.title("Template (blue) and data (red)")
        plt.plot(tw, tf, 'b.-')
        plt.plot(dw, df, 'r.-')
        plt.show()
        """

        # Carry out iteratively the cross-correlation.
        rv,rw,cc,maxind,rv_correction_kms = self.optimize_velocity_shift(dw, df, tw, tf)

        self.vradshift_kms = rv_correction_kms

        if rv[maxind] > 0.0:
            print("  A red-shift with respect to the template")
        else:
            print("  A blue-shift with respect to the template")

        fig = plt.figure(figsize=(10,6))
        plt.plot(rw, cc/np.nanmax(cc), 'bp-')
        plt.plot(rw[maxind], cc[maxind]/np.nanmax(cc), 'ro')
        plt.text(rw[maxind]+0.1,cc[maxind]/np.nanmax(cc),"R$_V$="+str(round(self.vradshift_kms,3))+" km s$^{-1}$")
        plt.xlabel('R$_V$ [Å]')
        plt.tight_layout()
        plt.show(block=False)

        self.gui.lineEdit_auto_velocity_shift.setText(str(round(self.vradshift_kms,3)))
 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def apply_velocity_shift(self):

        self.vradshift_aa = 1.0/self.doppler_shift(self.vradshift_kms)
 
        self.gui.xcurrent = self.gui.xcurrent * self.vradshift_aa
        self.gui.x = self.gui.x * self.vradshift_aa
        self.gui.xzoom = self.gui.xzoom * self.vradshift_aa

        self.vradshift_applied = True

        # Shift the line mask
        # Unflag line flags:
        table = self.gui.tableWidget
        rowCount = table.rowCount()
        
        # Iterate through the rows in reverse order
        for row in range(rowCount -1, -1, -1):  # Start from the last row
            # Retrieve the value from the first and second columns of the current row
            item_center = table.item(row, 0)  # Center value in column 0
            item_width = table.item(row, 1)  # Width value in column 1

            # Skip if either item is empty or invalid
            if not item_center or not item_width:
                continue

            try:
                center_value = float(item_center.text())
                width_value = float(item_width.text())
            except ValueError:
                # Skip rows with non-numeric data
                continue

            center_value = float(item_center.text()) * self.vradshift_aa
            width_value = float(item_width.text())

            table.setItem(row, 0, QTableWidgetItem(str(round(center_value, 3))))

        # shift tellurics
        try:
            vrad_tell = float(self.gui.lineEdit_telluric_vrad.text())
        except:
            vrad_tell = 0.0

        self.gui.lineEdit_telluric_vrad.setReadOnly(False)
        self.gui.lineEdit_telluric_vrad.setText(str(round(vrad_tell+float(self.vradshift_kms),3)))
        self.gui.lineEdit_telluric_vrad.setReadOnly(True)

        self.fit_spline()
        self.make_fig(0)
        self.make_fig(1)
        

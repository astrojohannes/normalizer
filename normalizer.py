#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QAction
from PyQt5.QtCore import QFile, QIODevice, QObject, Qt, QSortFilterProxyModel, QDir, QCoreApplication, QEvent
from PyQt5.uic import loadUi
from PyQt5.QtGui import QFont

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
        self.gui.method=UnivariateSpline
        
        self.gui.lineEdit_degree.setText('3')
        self.gui.lineEdit_smooth.setText('200')
        self.gui.lineEdit_sigma_high.setText('10.0')
        self.gui.lineEdit_sigma_low.setText('3.0')
        self.gui.lineEdit_fixed_width.setText('10')
        self.gui.lineEdit_interior_knots.setText('200')
        self.gui.lineEdit_offset.setText('1.0')
        self.gui.lineEdit_auto_velocity_shift.setText('0.0')
        self.gui.lineEdit_auto_velocity_shift_lim1.setText('-50')
        self.gui.lineEdit_auto_velocity_shift_lim2.setText('50')
        self.gui.rv=0.0
        self.gui.rv_err=0.0
        self.snr = None
 
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
        
        self.gui.xlim_l_last=0
        self.gui.xlim_h_last=0

        # Identify the layout that contains the main parts, including the tableWidget
        layout = self.gui.horizontalLayout_main

        # Standard telluric absorption bands
        telluric_intervals = self.find_telluric_intervals("skycalc_molec_abs.txt")
        self.gui.lineEdit_telluric.setText(', '.join('({}, {})'.format(*t) for t in telluric_intervals))
        self.gui.lineEdit_telluric_vrad.setText('0.0')

        # Hide button to apply velo shift
        self.gui.pushButton_shift_spectrum.setVisible(False)
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
        x0_user = round(x0,2)
        x1_user = round(x1,2)

        if x0_user > x1_user:
            x0 = x1_user
            x1 = x0_user
        else:
            x0 = x0_user
            x1 = x1_user

        if flagtype=='BAD':
            self.gui.lineEdit_telluric.setText(f"{tellurics}, ({x0},{x1})")
        elif flagtype=='LINE':
            linewidth = round(0.5*(x1-x0),2)
            linecenter = round(x0 + linewidth,2)
            self.add_values_to_first_empty_row(self.gui.tableWidget, [linecenter, linewidth])
        elif flagtype=='UNFLAG':

            # unflag line flags:
            table = self.gui.tableWidget
            rowCount = table.rowCount()
            # Iterate through the rows in reverse order
            for row in range(rowCount - 1, -1, -1):  # Start from the last row
                # Retrieve the value from the first column of the current row
                item = table.item(row, 0)  # 0 for the first column
                if item:  # Check if the item is not None
                    value = float(item.text())  # or int(item.text()), as appropriate
                    # Check if the value is between x0 and x1
                    if x0 <= value <= x1:
                        table.removeRow(row)  # Delete the row if the condition is met

            # unflag tellurics
            # Parse the intervals from the telluric line edit
            # and shift to the observed scale
            if self.gui.lineEdit_telluric.text().strip().strip(',') != '' and len(self.gui.lineEdit_telluric.text().strip().strip(','))>0:
                telluric_intervals = ast.literal_eval(self.gui.lineEdit_telluric.text().strip().strip(','))
                lambda_obs_factor=1/self.doppler_shift(float(self.gui.lineEdit_telluric_vrad.text()))

                if self.is_iterable(telluric_intervals[0]):
                    telluric_intervals = [(a * lambda_obs_factor, b * lambda_obs_factor) for a, b in telluric_intervals]
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
        self.gui.mask=np.array([])
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
        telluric_intervals = self.find_telluric_intervals("skycalc_molec_abs.txt")
        self.gui.lineEdit_telluric.setText(', '.join('({}, {})'.format(*t) for t in telluric_intervals))
        self.gui.lineEdit_telluric_vrad.setText('0.0')

        self.make_fig(0)

        self.gui.tableWidget.clearContents()

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
        #self.gui.pushButton_shift_spectrum.clicked.connect(self.apply_velocity_shift)

#                                 IO PART
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def selectFile(self):

        """ Opens a window for the user to select a FITS files
        """
 
        if self.gui.lbl_fname.text() is not None and os.path.isfile(self.gui.lbl_fname.text()):
            mydir = os.path.dirname(self.gui.lbl_fname.text())
        else:
            mydir = QDir.currentPath()

        filename,_ = QFileDialog.getOpenFileName(None,'Open FITS spectrum', self.tr("*.fits"))
 
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
                filename, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", 
                                                  "", "All Files (*);;Text Files (*.txt)", 
                                                  options=options)

            else:
                file_basename = os.path.basename(self.gui.lbl_fname.text())
                # Split the filename and extension
                file_name_without_extension, file_extension = os.path.splitext(file_basename)

                filename = f"{file_name_without_extension}_{str(int(self.gui.xlim_l_last))}-{str(int(self.gui.xlim_h_last))}.fits"

            self.gui.lbl_fname2.setText(filename)
            self.writefits(filename)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def zoom_fig(self,wave_min,wave_max):
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

        if self.gui.lineEdit_telluric.text().strip().strip(',') != '' and len(self.gui.lineEdit_telluric.text().strip().strip(','))>0:

            # Initialize new array for telluric mask
            self.gui.telluricmask = np.full_like(self.gui.xcurrent, 1)
            
            # Parse the intervals from the telluric line edit
            # and shift to the observed scale
            telluric_intervals = ast.literal_eval(self.gui.lineEdit_telluric.text().strip().strip(','))
            lambda_obs_factor=1/self.doppler_shift(float(self.gui.lineEdit_telluric_vrad.text()))
            if self.is_iterable(telluric_intervals[0]):
                telluric_intervals = [(a * lambda_obs_factor, b * lambda_obs_factor) for a, b in telluric_intervals]
            else:
                telluric_intervals = self.fix_telluric_intervals_notalist(telluric_intervals)

            for a, b in telluric_intervals:
                self.gui.telluricmask[(self.gui.xcurrent >= a) & (self.gui.xcurrent <= b)] = 0

            return self.gui.telluricmask, telluric_intervals
        else:
            self.gui.telluricmask = np.array([])

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
        c = 299792.458  # speed of light in km/s

        # Scale factor: sf = lambda_observed/lambda_emitted

        sf = np.sqrt((c + v_rad)/(c - v_rad))

        return sf


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def readfits(self, fitsfile, hduid=0):

        self.gui.ax[0].cla()
        self.gui.ax[1].cla()
    
        # Read the input file
        if fitsfile.lower().endswith('.fits') or fitsfile.lower().endswith('.fit'):
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
                available_cols = binary_table_hdu.data.columns.names  # Get the list of available columns
                
                # Check if 'Wavelength' and 'Normalized_Flux' columns are available
                if 'Wavelength' in available_cols and 'Normalized_Flux' in available_cols:
                    x = binary_table_hdu.data['Wavelength'].ravel()
                    y = binary_table_hdu.data['Normalized_Flux'].ravel()
                    hdr = binary_table_hdu.header
                else:
                    print("The expected columns 'Wavelength' and/or 'Normalized_Flux' are not available.")
                    print("Available columns are: ", available_cols)
                    
                    # Let user select columns
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

   
        self.gui.xlim_h_last=0
        self.gui.xlim_l_last=0

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
        vrad = self.gui.lineEdit_auto_velocity_shift.text()
        if vrad != '':
            try:
                vrad = str(round(float(vrad),2))
                new_hdr['SPEN_RV'] = (vrad, 'Rad. vel. in km/s from Spec. Normalizer')
            except:
                pass 
 
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

        ax[figid].step(x, y, color=col[figid], lw=0.8)

        if figid == 0:
            ax[0].text(0.03, 0.9, 'Original', fontsize=20, transform=ax[0].transAxes)
        elif figid == 1:
            ax[1].text(0.03, 0.9, 'Normalized', fontsize=20, transform=ax[1].transAxes)

        # Plot continuum fit in figure 0
        if len(self.gui.yi) > 0 and figid == 0 and showfit:
            ax[0].plot(x, self.gui.yi, color='r', lw=1.5)

        # plot telluric regions
        if figid == 0 and showfit:

            if self.gui.lineEdit_telluric.text().strip().strip(',') != '' and len(self.gui.lineEdit_telluric.text().strip().strip(','))>0:

                _, telluric_intervals = self.create_telluric_mask()

                if not self.is_iterable(telluric_intervals[0]):
                    telluric_intervals = self.fix_telluric_intervals_notalist(telluric_intervals)

                for start, end in telluric_intervals:
                    ax[0].fill_betweenx(np.linspace(min(y), max(y), 10), start, end, color='red', alpha=0.3, label='telluric')

        ax[1].set_xlabel('wavelength [AA]')
        
        # plot mask in top panel
        if len(self.gui.mask)>0:
            mask_edges=self.find_mask_edges()
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
                ax[0].fill_betweenx(yy,xx1,xx2, color='lightgray', alpha=0.3, label='line')

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

        """ Fit a spline using different methods
            and the user input parameters
        """

        if len(self.gui.xcurrent)>1:

            # update telluric mask
            self.create_telluric_mask()

            # apply both, telluric and line mask --> updates ymaskedcurrent
            self.apply_mask()

            ##############
            # FITTING part
            x, y = self.gui.xcurrent, self.gui.ycurrent

            xlim_l=float(self.gui.ax[0].get_xlim()[0])
            xlim_h=float(self.gui.ax[0].get_xlim()[1])

            self.gui.xlim_l_last=xlim_l
            self.gui.xlim_h_last=xlim_h
                
            k = int(self.gui.lineEdit_degree.text())
            if k <=1: k=1
            elif k>5: k=5
    
            # read user input: smoothing parameter
            s = int(self.gui.lineEdit_smooth.text())
    
            # Note: the masked input array contains nan values, which InterpolatedUnivariateSpline cannot handle
            # A workaround is to use zero weights for not-a-number data points:
            w = np.isnan(self.gui.ymaskedcurrent)
            weights = np.ones_like(y)
            weights[w] = 0.0
            
            # if user provided fixpoints, raise their weights to assure that fit will intersect with these points
            wu=self.gui.lineEdit_fixpoints.text()+','
            wu=wu.split(',')
            if len(wu)>0 and wu[0].strip() != '':
                wu=np.array([float(a) for a in wu[:-1]],dtype=float)
                for fp in wu:
                    weights[self.find_nearest_idx(x,fp)]=1e6

            if self.gui.comboBox_method.currentText()=='LSQUnivariateSpline':
                
                every=int(self.gui.lineEdit_interior_knots.text())

                # Obtain the t array with step 'every'
                t_indices = np.arange(1, len(x), every)  # Get the indices with the step 'every'.

                if len(self.gui.ymaskedcurrent)>0:
                    # filter these indices based on the non-NaN status in 'ymaskedcurrent'.
                    final_indices = t_indices[~np.isnan(self.gui.ymaskedcurrent[t_indices])]
                else:
                    final_indices = t_indices

                # Use these final indices to access the corresponding items in 'xcurrent'.
                self.gui.knots_x = x[final_indices]
                t = self.gui.knots_x

                # do the fit
                spl = self.gui.method(x, y, t, k=k, w=weights, check_finite=False, ext=3)
                yi=np.copy(spl(x)).flatten()

                if len(self.gui.ymaskedcurrent)>0 and len(yi)>0:
                    self.gui.knots_y = np.copy(spl(self.gui.knots_x)).flatten()
    
            else:
                # UnivariateSpline
                # do the fit
                # UnivariateSpline was found to have problems with large numeric y-values, so we down-scale to the mean
                scalingfactor = np.nanmean(y)

                if scalingfactor < 1000:
                    scalingfactor = 1.0
                spl = self.gui.method(x, y/scalingfactor, k=k, w=weights/scalingfactor, s=s, check_finite=True, ext=3)
                spl.set_smoothing_factor(s)
                yi=scalingfactor*np.copy(spl(x)).flatten()
   
            if self.gui.lineEdit_offset.text().strip()=='':
                offs = 1.0
            else:
                try:
                    offs = float(self.gui.lineEdit_offset.text())
                except:
                    offs = 1.0

            ynorm = np.divide(y, np.array(yi), where=np.array(yi) != 0)
            ynorm *= 1.0/offs

            self.gui.yi = np.array(yi)
 
            if not len(self.gui.ynorm)>0:
                self.gui.ynorm = ynorm
            self.gui.ynormcurrent = ynorm

            # now we set to NaN the telluric-masked regions
            if len(self.gui.telluricmask)>0:
                self.gui.yi[self.gui.telluricmask==0] = np.nan
                self.gui.ynormcurrent[self.gui.telluricmask==0] = 1.0

            self.make_fig(0,showfit=showfit)
            self.make_fig(1,showfit=showfit)


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
            self.gui.label_8.setHidden(True)
            self.gui.lineEdit_interior_knots.setHidden(True)
            
            self.gui.label_2.setHidden(False)
            self.gui.lineEdit_smooth.setHidden(False)
            
            self.gui.method=UnivariateSpline
        else:
            self.gui.label_2.setHidden(True)
            self.gui.lineEdit_smooth.setHidden(True)

            self.gui.label_8.setHidden(False)
            self.gui.lineEdit_interior_knots.setHidden(False)
            
            self.gui.method=LSQUnivariateSpline
 

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def identify_mask(self):
        """ Identify/mask lines in normed spectrum
            using rms measured over smoothed spectrum
            iteratively until change in rms is less than 1 percent
        """
 
        if len(self.gui.ynormcurrent)>1:
 
            del self.gui.mask
            self.gui.mask=np.array([])

            self.apply_mask()

            x = np.copy(self.gui.xcurrent)
            y = np.copy(self.gui.ynormcurrent)


            if len(self.gui.telluricmask)>0:
                xfit = x[self.gui.telluricmask != 0]
                yfit = y[self.gui.telluricmask != 0]
            else:
                xfit = x
                yfit = y

            sigma_high=float(self.gui.lineEdit_sigma_high.text())
            sigma_low=float(self.gui.lineEdit_sigma_low.text())        

            iters=1     # we do several iterations in PeakMask, so this is left from previous versions
            new_mask=np.array([True for x in yfit])
            for i in range(iters):

                if np.nansum(new_mask)==0:
                    print(f"Stopping line identification early.")
                    if 'yfit_pref' in locals():
                        yfit = yfit_prev
                    continue

                yfit_prev = yfit

                # Fit a 3rd degree polynomial to the data during first 2 iterations
                # else do first degree
                if i<2:
                    coefficients = np.polyfit(xfit, yfit, 1)    # just skipt to 1st degree for the moment
                else:
                    coefficients = np.polyfit(xfit, yfit, 1) 

                # Create a polynomial function from the coefficients
                polynomial = np.poly1d(coefficients)

                # Evaluate the polynomial at each x-value
                fitted_y = polynomial(x)

                # Divide the original y-values by the fitted ones
                normed_y = self.gui.ynormcurrent / fitted_y

                try:
                    masker_high = PeakMask(normed_y, sigma_smooth=1, sigma_threshold=sigma_high, rms_tolerance=0.1, maxnumber_iterations=1)
                    masker_low = PeakMask(normed_y, sigma_smooth=1, sigma_threshold=sigma_low, rms_tolerance=0.1, maxnumber_iterations=1)

                    mask_high, iter_high, rms_high = masker_high.create_mask()
                    mask_low, iter_low, rms_low = masker_low.create_mask()

                    self.snr = round(1/(0.5*((rms_high+rms_low))),1)

                    print(f"Peak-finder performed {iter_high}/{iter_low} iterations for high/low mask. SNR: {self.snr}")
 
                except Exception as e:
                    print(f"PeakMask failure in line identification:\n{e}")
                    self.snr = None
                    continue

                mask_high[self.gui.telluricmask==0] = False
                mask_low[self.gui.telluricmask==0] = False

                try:
                    new_mask = np.array(exp_mask(mask_high,constraint=mask_low, keep_mask=True, quiet=True),dtype=bool)
                    # grow mask by user input
                    #new_mask = np.array(exp_mask(new_mask, radius=1, keep_mask=True, quiet=True),dtype=bool)
                except Exception as e:
                    print(f"Exp_mask failure in line identification:\n{e}")
                    self.snr = None
                    continue

                new_mask[self.gui.telluricmask==0] = False

                # Masked values are replaced with np.nan
                y[new_mask==True] = np.nan

                # Prepare indices for interpolation
                x_values = np.arange(len(y))

                # Find the indices where new_mask is False
                indices_false = np.where(~new_mask)[0]

                # Interpolate y values at positions where new_mask is True
                try:
                    y[new_mask] = np.interp(x_values[new_mask], x_values[~new_mask], y[~new_mask])
                    if len(new_mask)>0:
                        self.gui.mask=new_mask
                except Exception as e:
                    print(f"{e}")

                # xfit = x[new_mask != 0] # not needed anymore, since we perform no looping
                # yfit = y[new_mask != 0]
                
            self.fit_spline(showfit=True)


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
        self.gui.tableWidget.setRowCount(1000)

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

    def read_template(self, file_path):
        data = np.genfromtxt(file_path, dtype=[('waveobs', np.float64), ('flux', np.float64), ('err', np.float64)])
        return data.view(np.recarray)

    def determine_rad_velocity(self):
        c = 300000.0  # speed of light in km/s

        if len(self.gui.ynormcurrent)>0:
            waveobs,flux=np.array(self.gui.xcurrent-self.gui.rv,dtype=np.float64),np.array(self.gui.ynormcurrent,dtype=np.float64)
        else:
            self.gui.rv=0.0
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
        template = self.read_template("./templates/NARVAL.Sun.370_1048nm/template.txt.gz")
        template['waveobs']=template['waveobs']*10.0    # convert to Angstroem

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

        # Carry out the cross-correlation.
        # The RV-range is -300 - +300 km/s in steps of 0.1 km/s.
        rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -300., 300., 0.1, skipedge=0)

        # Find the index of maximum cross-correlation function
        maxind = np.argmax(cc)

        # Convert the radial velocity shift to a shift in wavelength.
        # Assume a mean wavelength for the conversion. 
        mean_wavelength = np.mean(dw)  # or any specific wavelength you are interested in
        rw = mean_wavelength * (rv / c)

        print("Cross-correlation function is maximized at dRV = ", rv[maxind], " km/s")
        print("Cross-correlation function is maximized at dRV = ", rw[maxind], " AA")
 
        if rv[maxind] > 0.0:
            print("  A red-shift with respect to the template")
        else:
            print("  A blue-shift with respect to the template")

        fig = plt.figure(figsize=(10,6))
        plt.plot(rw, cc/np.nanmax(cc), 'bp-')
        plt.plot(rw[maxind], cc[maxind]/np.nanmax(cc), 'ro')
        plt.text(rw[maxind]+0.1,cc[maxind]/np.nanmax(cc),"R$_V$="+str(round(rv[maxind],3))+" km s$^{-1}$")
        plt.xlabel('R$_V$ [Å]')
        plt.tight_layout()
        plt.show()

        self.gui.lineEdit_auto_velocity_shift.setText(str(round(rv[maxind],3)))

    def apply_velocity_shift(self,rv_new=0.0):
        
        if rv_new==0:
            rv_new=float(self.gui.lineEdit_auto_velocity_shift.text())
        new_xcurrent=np.copy(self.gui.xcurrent)+rv_new-self.gui.rv
        self.gui.xcurrent=new_xcurrent
        self.gui.rv=rv_new
        self.fit_spline()
        self.make_fig(0)
        self.make_fig(1)
        

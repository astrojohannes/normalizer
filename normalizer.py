# -*- coding: utf-8 -*-

from PySide2 import QtCore, QtGui, QtWidgets
from PySide2.QtWidgets import *
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QApplication, QTableWidgetItem
from PySide2.QtCore import QFile, QIODevice, QObject, Qt, QSortFilterProxyModel
from PySide2.QtGui import QIcon, QPixmap, QWindow

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backend_bases import NavigationToolbar2
import os,sys

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.modeling import models

"""
import specutils
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import estimate_line_parameters, find_lines_derivative, fit_lines
from specutils.manipulation import extract_region
"""

from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline

from exp_mask import exp_mask
from ispec_helper import *

class start(QObject):

    def __init__(self, ui_file, parent=None):

        """ Initialize main window for user interactions
        """

        super(start, self).__init__(parent)
        ui_file = QFile(ui_file)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file}: {ui_file.errorString()}")
            sys.exit(-1)
        loader = QUiLoader()
        self.gui = loader.load(ui_file)
        ui_file.close()

        if not self.gui:
            print(loader.errorString())
            sys.exit(-1)
        
        self.gui.label_8.setHidden(True)
        self.gui.lineEdit_interior_knots.setHidden(True)
        
        self.gui.setGeometry(0, 0, 700, 680)        
        self.gui.show()

        # figure with 2 subplots
        fig, ax = plt.subplots(2, 1, sharex = True)
        mngr = plt.get_current_fig_manager()
        geom = mngr.window.geometry()
        x,y,dx,dy = geom.getRect()
        # put window into the upper left corner for example:
        mngr.window.setGeometry(701, 0, dx, dy)
        fig.set_size_inches(8, 8)
        
        self.gui.fig = fig
        self.gui.ax = ax
        plt.ion()
        
        home = NavigationToolbar2.home
        NavigationToolbar2.home = self.new_home

        # set standard values
        self.gui.method=UnivariateSpline
        
        self.gui.lineEdit_degree.setText('3')
        self.gui.lineEdit_smooth.setText('20')
        self.gui.lineEdit_sigma_high.setText('1.8')
        self.gui.lineEdit_sigma_low.setText('1.0')
        self.gui.lineEdit_fixed_width.setText('10')
        self.gui.lineEdit_interior_knots.setText('200')
        
        self.gui.x=np.array([])     # origianl wavelength range
        self.gui.y=np.array([])     # original spectrum

        self.gui.xzoom=np.array([])     # zoomed-in wavelength range
        self.gui.yzoom=np.array([])     # zoomed-in original spectrum
        
        self.gui.yi=np.array([])    # smoothed, masked and interpolated spectrum used for normalisation (continuum fitting)
        self.gui.ynorm=np.array([]) # normalized array
        
        
        self.gui.xcurrent=np.array([])    # x,y currently used for manipulation
        self.gui.ycurrent=np.array([])    # figure 0
        self.gui.ynormcurrent=np.array([])    # figure 1
        self.gui.ymaskedcurrent=np.array([])
        
        self.gui.mask=np.array([])
        
        self.gui.xlim_l_last=0
        self.gui.xlim_h_last=0

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
        
    def new_home(self, *args, **kwargs):
        
        """ add some functionality to matplotlib's home button
        
        """
        
        self.gui.xcurrent=self.gui.x
        self.gui.ycurrent=self.gui.y
        self.gui.ymaskedcurrent=self.gui.y
        self.gui.ynormcurrent=self.gui.ynorm
        self.gui.yi=np.array([])
        self.gui.mask=np.array([])

        self.gui.ax[0].set_xlim([min(self.gui.x),max(self.gui.x)])        
        self.gui.ax[1].set_xlim([min(self.gui.x),max(self.gui.x)])        

        self.gui.xlim_h_last=max(self.gui.x)
        self.gui.xlim_l_last=min(self.gui.x)

        self.fit_spline(showfit=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def connect_buttons(self):

        """ Connect the GUI buttons with slots
        """
        self.gui.pushButton_openfits.clicked.connect(self.selectFile)
        self.gui.comboBox_method.currentIndexChanged.connect(self.method_changed) 
        self.gui.pushButton_normalize.clicked.connect(lambda: self.fit_spline(showfit=True))
        self.gui.pushButton_identify_mask_lines.clicked.connect(self.identify_mask)
        self.gui.pushButton_linetable_mask.clicked.connect(self.linetable_mask)
        self.gui.pushButton_savefits.clicked.connect(self.saveFile)

#                                 IO PART
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def selectFile(self):

        """ Opens a window for the user to select a FITS files
        """
 
        if self.gui.lbl_fname.text() is not None and os.path.isfile(self.gui.lbl_fname.text()):
            mydir = os.path.dirname(self.gui.lbl_fname.text())
        else:
            mydir = QtCore.QDir.currentPath()

        filename,_ = QFileDialog.getOpenFileName(None,'Open FITS spectrum', self.tr("Spectrum (*.fits)"))
 
        self.gui.lbl_fname.setText(filename)

        self.readfits(filename)
        self.make_fig(0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def saveFile(self):
        """
        if self.gui.lbl_fname.text() is not None and os.path.isfile(self.gui.lbl_fname.text()):
            mydir = os.path.dirname(self.gui.lbl_fname.text())
        else:
            mydir = QtCore.QDir.currentPath()

        filename,_ = QFileDialog.getSaveFileName(None,'Save to FITS', self.tr("(*.fits)"))
        """
        filename=str(self.gui.lbl_fname.text().split('.')[0:-1]).replace('[','').replace(']','').replace('\'','')+'_'+str(int(self.gui.xlim_l_last))+'_'+str(int(self.gui.xlim_h_last))+'.fits'
        self.gui.lbl_fname2.setText(filename)
        self.writefits(filename)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def tr(self, text):
        return QObject.tr(self, text)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def readfits(self,fitsfile,hduid=0):

        self.gui.ax[0].cla()
        self.gui.ax[1].cla()

        """ Read the fits file selected by user
        """

        hdus = fits.open(fitsfile)
        hdr = hdus[hduid].header
       
        if 'TELESCOP' in hdr and 'INSTRUME' in hdr:
            if hdr['TELESCOP'].strip()=='NOT' and hdr['INSTRUME'].strip()=='FIES':
                for kk in ['CTYPE2', 'CTPYE2', 'CUNIT2','CTYPE1','CUNIT1']:
                    if kk in hdr:
                        del hdr[kk]
        
        img = hdus[hduid].data
        wcs = WCS(hdr)
       
        # make spectral axis
        if int(hdr['NAXIS'])==2:
            y = img.sum(axis=0)   # summing up along spatial direction
            x = wcs.all_pix2world([(x,0) for x in range(len(y))], 0)
            x = np.delete(x,1,axis=1)
            x = x.flatten()
            y = y.flatten()
        elif int(hdr['NAXIS'])==1:
            y = img
            x = wcs.all_pix2world([x for x in range(len(y))], 0)[0]

        # save re-usable quantities in global variables
        self.gui.x = x
        self.gui.y = y
        self.gui.xcurrent = x
        self.gui.ycurrent = y
        self.gui.ymaskedcurrent = y
        self.gui.hdr = hdr

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def writefits(self,fitsfile,hduid=0):

        """ Save normalized spectrum in
            user fits file
        """
        
        # cleanup header
        hdr=self.gui.hdr
        
        for kk in ['NAXIS2','CRPIX2','CDELT2','CTYPE2','CRVAL2']:
            if kk in hdr: del hdr[kk]
        hdr['NAXIS']=1
        
        fits.writeto(self.gui.lbl_fname2.text(),data=self.gui.ynormcurrent,header=hdr,overwrite=True)

#                                  PLOTTING
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def make_fig(self,figid,showfit=True):
        
        """ make the 2-panel matplotlib figure
        """
        
        plt.ion()
        fig = self.gui.fig
        plt.subplots_adjust(left=0.11, bottom=0.1, right=0.98, top=0.98, wspace=0, hspace=0)

        self.gui.ax[figid].cla()

        if self.gui.xlim_l_last>0:
            self.gui.ax[0].set_xlim([self.gui.xlim_l_last,self.gui.xlim_h_last])
            self.gui.ax[1].set_xlim([self.gui.xlim_l_last,self.gui.xlim_h_last])
                
        if figid==0:
            x, y = self.gui.xcurrent, self.gui.ycurrent
        else:
            x,y = self.gui.xcurrent,self.gui.ynormcurrent

        col=['b','g']

        self.gui.ax[figid].step(x, y,color=col[figid],lw=0.8)

        # Plot continnum fit in figure 0
        yi = self.gui.yi
        if len(yi)>0 and figid==0 and showfit==True:
            self.gui.ax[0].plot(x, yi, color='r', lw=1.5)

        #self.gui.ax[1].set_xlabel('$\lambda\ [\mathrm{\AA}]$')
        self.gui.ax[1].set_xlabel('wavelength')

        # plot mask in bottom
        if len(self.gui.mask)>0:
            for xx in self.gui.xcurrent[self.gui.mask>0]:
                self.gui.ax[0].axvline(x=xx, color='lightgray', alpha=0.1)

        # draw and show
        plt.draw()
        plt.show(block=False)
#                                  FITTING 
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def fit_spline(self, showfit=True):

        """ Fit a spline using different methods
            and the user input parameters
        """

        xlim_l=float(plt.gca().get_xlim()[0])
        xlim_h=float(plt.gca().get_xlim()[1])

        # user has zoomed in
        if abs(xlim_l-self.gui.xlim_l_last)>1 and abs(xlim_h-self.gui.xlim_h_last)>1:
            self.gui.xcurrent=self.gui.x[(self.gui.x>=xlim_l) & (self.gui.x<=xlim_h)]
            self.gui.ycurrent=self.gui.y[(self.gui.x>=xlim_l) & (self.gui.x<=xlim_h)]

            self.gui.xlim_l_last=xlim_l
            self.gui.xlim_h_last=xlim_h
            
            self.gui.mask=np.array([])
            self.gui.yi=np.array([])
            
        self.apply_mask()
        x, y = self.gui.xcurrent, self.gui.ymaskedcurrent

        k = int(self.gui.lineEdit_degree.text())
        if k <=1: k=1
        elif k>5: k=5

        s = int(self.gui.lineEdit_smooth.text())

        # Note: the masked input array contains nan values, which InterpolatedUnivariateSpline cannit handle
        # A workaround is to use zero weights for not-a-number data points:
        w = np.isnan(y)
        y[w] = 0.
        if self.gui.comboBox_method.currentText()=='LSQUnivariateSpline':
            tu=self.gui.lineEdit_interior_knots.text()
            tu=tu.split(',')
            if len(tu)>1:
                t=np.array([float(x) for x in tu],dtype=float)
            else:
                every=int(self.gui.lineEdit_interior_knots.text())
                t=self.gui.xcurrent[1:-1:every]
                t=t[~np.isnan(t)]
            spl = self.gui.method(x, y, t, k=k, w=~w)
        else:
            spl = self.gui.method(x, y, k=k, w=~w)
            spl.set_smoothing_factor(s)

        # normalize: divide spline
        yi=np.copy(spl(x)).flatten()
                    
        ynorm = np.array(self.gui.ycurrent) / np.array(yi)
        
        self.gui.yi = np.array(yi)
        if not len(self.gui.ynorm)>0:
            self.gui.ynorm = ynorm
        self.gui.ynormcurrent = ynorm

        self.make_fig(0,showfit=showfit)
        self.make_fig(1,showfit=showfit)

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

        """ Identify and mask lines in normed spectrum
            using rms measured over smoothed spectrum
            and application of an expanding mask
        """

        self.gui.mask=np.array([])
        self.fit_spline()
                
        snr_high=float(self.gui.lineEdit_sigma_high.text())
        snr_low=float(self.gui.lineEdit_sigma_low.text())
        
        y = np.copy(self.gui.ymaskedcurrent)

        ysmooth = self.smooth(y,2)
        rms=np.std(ysmooth)
                
        mask_high = (abs(ysmooth-np.mean(ysmooth)) > snr_high*rms)
        mask_low  = (abs(ysmooth-np.mean(ysmooth)) > snr_low*rms)
        
        new_mask = np.array(exp_mask(mask_high,constraint=mask_low, quiet=True),dtype=bool)
        
        if len(new_mask)>0:
            self.gui.mask=new_mask
        self.fit_spline()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def apply_mask(self):
        if len(self.gui.mask)>0:
            ymasked = np.copy(self.gui.ycurrent)
            ymasked[self.gui.mask] = np.nan
            self.gui.ymaskedcurrent = np.array(ymasked)
        else:
            self.gui.ymaskedcurrent = np.copy(self.gui.ycurrent)
            
        nans, x= self.nan_helper(self.gui.ymaskedcurrent)
        self.gui.ymaskedcurrent[nans]= np.interp(x(nans), x(~nans), self.gui.ymaskedcurrent[~nans])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def linetable_mask(self):
        
        """ use the user lines from the table to mask spectral regions
            when doing the continuum normalization
        """
        
        self.gui.mask=np.array([])
        self.fit_spline()
        #self.gui.tableWidget.clear()
        
        """
        specutils.conf.do_continuum_function_check = False
        this_unit='erg cm^-2 s^-1'

        # convert to specutils spectrum
        self.gui.lamb = self.gui.xcurrent * u.AA
        self.gui.flux = (self.gui.ynormcurrent-1) * u.Unit(this_unit)
        spec = Spectrum1D(spectral_axis=self.gui.lamb, flux=self.gui.flux)

        # find lines
        #lines_table = find_lines_derivative(spec)

        for ii,line in enumerate(lines):
            line_center=str(round(line['line_center'].value,3))
            line_type=str(line['line_type'])
            if ii==0: table.setRowCount(0)
            rowPosition = table.rowCount()
            table.insertRow(rowPosition)
            table.setItem(rowPosition , 0, QTableWidgetItem(line_center))
            table.setItem(rowPosition , 1, QTableWidgetItem(line_type))

        """

        # win width
        ww=int(self.gui.lineEdit_fixed_width.text())

        lines = []
        widths = []
        table=self.gui.tableWidget
        for row in range(table.rowCount()):
            l = table.item(row, 0)
            w = table.item(row, 1)
            c = float(l.text()) if l is not None else np.nan
            w = float(w.text()) if w is not None else np.nan
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
        self.gui.mask = np.array([False for x in self.gui.xcurrent])
        self.gui.mask[idx] = True

        self.fit_spline()

        """
            sub_region = SpectralRegion(c-w/2, c+w/2)
            sub_spectrum = extract_region(spec, sub_region)

            # Fit the line and calculate the fitted flux values (``y_fit``)
            g_init = models.Gaussian1D(amplitude=-0.5 * u.Unit(this_unit), mean=c, stddev=0.1*u.AA)
            g_fit = fit_lines(sub_spectrum, g_init)
            y_fit = g_fit(sub_spectrum.spectral_axis)

            # Plot the original spectrum and the fitted.
            plt.plot(sub_spectrum.spectral_axis, sub_spectrum.flux, label="Original spectrum")
            plt.plot(sub_spectrum.spectral_axis, y_fit, label="Fit result")
            plt.title('Single fit peak')
            plt.grid(True)
            plt.legend()
            plt.draw()
        """

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem
from PyQt5.QtCore import QFile, QIODevice, QObject, Qt, QSortFilterProxyModel, QDir, QCoreApplication
from PyQt5.QtGui import QIcon, QPixmap, QWindow
from PyQt5.uic import loadUi

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backend_bases import NavigationToolbar2
import os,sys

mpl.rcParams['text.usetex'] = False

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
import ispec_helper as ispec

class start(QObject):

    def __init__(self, ui_file, parent=None):

        """ Initialize main window for user interactions
        """

        super(start, self).__init__(parent)
        
        self.gui = QMainWindow()
        loadUi(ui_file, self.gui)
        
        if not self.gui:
            print(loader.errorString())
            sys.exit(-1)
        
        self.gui.label_8.setHidden(True)
        self.gui.lineEdit_interior_knots.setHidden(True)

        self.gui.label_4.setHidden(True)
        self.gui.lineEdit_fixed_width.setHidden(True)
        
        self.gui.setGeometry(0, 0, 700, 815)        
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
        self.gui.lineEdit_auto_velocity_shift.setText('0')
        self.gui.lineEdit_auto_velocity_shift_lim1.setText('-50')
        self.gui.lineEdit_auto_velocity_shift_lim2.setText('50')
        self.gui.veloshift_current=0.0
        self.gui.rv=0.0
        self.gui.rv_err=0.0
        
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
        self.gui.veloshift_current=0.0

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
        self.gui.pushButton_normalize.clicked.connect(lambda _: self.fit_spline(showfit=True))
        self.gui.pushButton_identify_mask_lines.clicked.connect(self.identify_mask)
        self.gui.pushButton_linetable_mask.clicked.connect(self.linetable_mask)
        self.gui.pushButton_savefits.clicked.connect(self.saveFile)
        self.gui.pushButton_determine_rad_velocity.clicked.connect(self.determine_rad_velocity)
        self.gui.pushButton_shift_spectrum.clicked.connect(self.apply_velocity_shift)

#                                 IO PART
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def selectFile(self):

        """ Opens a window for the user to select a FITS files
        """
 
        if self.gui.lbl_fname.text() is not None and os.path.isfile(self.gui.lbl_fname.text()):
            mydir = os.path.dirname(self.gui.lbl_fname.text())
        else:
            mydir = QDir.currentPath()

        filename,_ = QFileDialog.getOpenFileName(None,'Open FITS spectrum', self.tr("Spectrum (*.fits)"))
 
        if mydir == QDir.currentPath():
            self.gui.lbl_fname.setText(os.path.basename(filename))
        else:
            self.gui.lbl_fname.setText(filename)

        self.readfits(filename)
        self.make_fig(0)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def zoom_fig(self,wave_min,wave_max):
        self.gui.ax[0].set_xlim([wave_min,wave_max])
        #self.gui.ax[1].set_xlim()
  
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
                x = binary_table_hdu.data['Wavelength']
                y = binary_table_hdu.data['Normalized_Flux']
                hdr = binary_table_hdu.header
            else:
                # Otherwise, assume a regular FITS file and load image data
                hdr = hdus[hduid].header
                img = hdus[hduid].data
                wcs = WCS(hdr)
                if int(hdr['NAXIS']) == 2:
                    y = img.sum(axis=0)   # summing up along spatial direction
                    x = wcs.all_pix2world([(x, 0) for x in range(len(y))], 0)
                    x = np.delete(x, 1, axis=1)
                    x = x.flatten()
                    y = y.flatten()
                elif int(hdr['NAXIS']) == 1:
                    y = img
                    x = wcs.all_pix2world([x for x in range(len(y))], 0)[0]
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
        self.gui.veloshift_current = 0.0
    
        self.fit_spline()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def readfits_old(self,fitsfile,hduid=0):

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
        self.gui.ynorm=np.array([])
        self.gui.ynormcurrent=np.array([])
        self.gui.yi=np.array([])
        self.gui.veloshift_current=0.0

        self.fit_spline()
        
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def writefits(self, fitsfile, hduid=0):
        """ Save normalized spectrum in
            user fits file
        """
    
        # Create columns for wavelength (x-axis) and normalized flux (y-axis)
        col1 = fits.Column(name='Wavelength', format='E', array=self.gui.xcurrent)
        col2 = fits.Column(name='Normalized_Flux', format='E', array=self.gui.ynormcurrent)
    
        # Create a ColDefs object from the columns
        cols = fits.ColDefs([col1, col2])
    
        # Create a BinTableHDU object from the ColDefs object
        tbhdu = fits.BinTableHDU.from_columns(cols)
    
        # Create a new header for the output file
        new_hdr = fits.Header()
        new_hdr['SIMPLE'] = True
        new_hdr['BITPIX'] = -32  # For single-precision floating-point values
        new_hdr['NAXIS'] = 2
        new_hdr['EXTEND'] = True
    
        # Write the data to the output FITS file
        tbhdu.writeto(self.gui.lbl_fname2.text(), overwrite=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def writefits_old(self,fitsfile,hduid=0):

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
        
        if figid==0:
            self.gui.ax[0].text(0.03,0.9,'Original',fontsize=20,transform=self.gui.ax[0].transAxes)
        elif figid==1:
            self.gui.ax[1].text(0.03,0.9,'Normalized',fontsize=20,transform=self.gui.ax[1].transAxes)

        # Plot continuum fit in figure 0
        yi = self.gui.yi
        if len(yi)>0 and figid==0 and showfit==True:
            self.gui.ax[0].plot(x, yi, color='r', lw=1.5, label='spline fit')

        #self.gui.ax[1].set_xlabel('$\lambda\ [\mathrm{\AA}]$')
        self.gui.ax[1].set_xlabel('wavelength [AA]')

        # plot mask in top panel
        if len(self.gui.mask)>0:
            mask_edges=self.find_mask_edges()
            ii=0
            while ii < len(mask_edges)-2:
                xx1 = float(x[mask_edges[ii]])
                xx2 = float(x[mask_edges[ii+1]])
                ii+=2
                ylim_l=np.nanmin(self.gui.ycurrent)
                ylim_h=np.nanmax(self.gui.ycurrent)
                yy=np.linspace(ylim_l,ylim_h,10)
                self.gui.ax[0].fill_betweenx(yy,xx1,xx2, color='lightgray', alpha=0.3, label='masked region')

        # draw and show
        plt.draw()
        plt.show(block=False)
        self.gui.activateWindow()
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

        # read user input: smoothing parameter
        s = int(self.gui.lineEdit_smooth.text())

        # Note: the masked input array contains nan values, which InterpolatedUnivariateSpline cannot handle
        # A workaround is to use zero weights for not-a-number data points:
        w = np.isnan(y)
        y[w] = 0.
        # the weights are found after inverting
        w=~w
        w=np.array(w,dtype=np.float64)
        
        # if user provided fixpoints, raise their weights to assure that fit will intersect with these points
        wu=self.gui.lineEdit_fixpoints.text()
        wu=wu.split(',')
        if len(wu)>0 and wu[0].strip() != '':
            wu=np.array([float(a) for a in wu],dtype=float)
            for fp in wu:
                w[self.find_nearest_idx(x,fp)]=1e10
        
        if self.gui.comboBox_method.currentText()=='LSQUnivariateSpline':
            
            # knot points where the polynomials connect
            tu=self.gui.lineEdit_interior_knots.text()
            tu=tu.split(',')
            
            # user input of knot points
            if len(tu)>1:
                t=np.array([float(x) for x in tu],dtype=float)
            # or use every ith value along x as knot point
            else:
                every=int(self.gui.lineEdit_interior_knots.text())
                t=self.gui.xcurrent[1:-1:every]
                t=t[~np.isnan(t)]
            
            # do the fit
            spl = self.gui.method(x, y, t, k=k, w=w)
        else:
            # do the fit
            spl = self.gui.method(x, y, k=k, w=w)
            spl.set_smoothing_factor(s)

        # normalize: divide spline
        yi=np.copy(spl(x)).flatten()
                    
        ynorm = np.divide(np.array(self.gui.ycurrent), np.array(yi), where=np.array(yi) != 0)

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

        ysmooth = self.smooth(y,4)
        rms=np.std(ysmooth)
                
        mask_high = (abs(ysmooth-np.mean(ysmooth)) > snr_high*rms)
        mask_low  = (abs(ysmooth-np.mean(ysmooth)) > snr_low*rms)
        
        new_mask = np.array(exp_mask(mask_high,constraint=mask_low, iters=100, keep_mask=True, quiet=True),dtype=bool)
        
        if len(new_mask)>0:
            self.gui.mask=new_mask
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
            
        return np.array(edges)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

    def apply_mask(self):
        if len(self.gui.mask)>0:
            ymasked = np.copy(self.gui.ycurrent)
            ymasked[self.gui.mask] = np.nan
            self.gui.ymaskedcurrent = np.array(ymasked)
        else:
            self.gui.ymaskedcurrent = np.copy(self.gui.ycurrent)
            
        nans, x= self.nan_helper(self.gui.ymaskedcurrent)
        if np.any(~nans):
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
        
        lines = []
        widths = []
        table=self.gui.tableWidget
        for row in range(table.rowCount()):
            l = table.item(row, 0)
            w = table.item(row, 1)
            c = float(l.text().strip()) if (l is not None) and (l.text().strip() != '') else np.nan
            w = float(w.text().strip()) if (w is not None) and (w.text().strip() != '') else np.nan
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
        
        self.fit_spline(showfit=True)

#                             RADIAL VELOCITY
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    def determine_rad_velocity(self):

        if len(self.gui.ynormcurrent)>0:
            waveobs,flux=np.array(self.gui.xcurrent-self.gui.rv,dtype=np.float64),np.array(self.gui.ynormcurrent,dtype=np.float64)
        elif len(self.gui.ycurrent)>0:
            waveobs,flux=np.array(self.gui.xcurrent-self.gui.rv,dtype=np.float64),np.array(self.gui.ycurrent,dtype=np.float64)
        else:
            self.gui.rv=0.0
            self.gui.lineEdit_auto_velocity_shift.setText('0')
        err=np.array([0.0 for a in range(len(self.gui.xcurrent))],dtype=np.float64)
        this_arr=np.vstack((waveobs,flux,err))
        this_spec=np.core.records.fromrecords(this_arr.T, names='waveobs,flux,err')
        this_spec=this_spec.view(np.recarray)

        #--- Radial Velocity determination with template -------------------------------
        # - Read synthetic template
        #template = ispec.read_spectrum("./templates/Atlas.Arcturus.372_926nm/template.txt.gz")
        #template = ispec.read_spectrum("./templates/Atlas.Sun.372_926nm/template.txt.gz")
        template = ispec.read_spectrum("./templates/NARVAL.Sun.370_1048nm/template.txt.gz")
        template['waveobs']=template['waveobs']*10.0
        
        #template = ispec.read_spectrum("./templates/Synth.Sun.300_1100nm/template.txt.gz")

        models, ccf = ispec.cross_correlate_with_template(this_spec, template, \
                                lower_velocity_limit=float(self.gui.lineEdit_auto_velocity_shift_lim1.text()), upper_velocity_limit=float(self.gui.lineEdit_auto_velocity_shift_lim2.text()), \
                                velocity_step=1.0, fourier=False)

        # Number of models represent the number of components
        components = len(models)
        if components>0:
            # First component:
            rv_new = models[0].mu() # km/s
            rv_err_new = models[0].emu() # km/s
            self.gui.lineEdit_auto_velocity_shift.setText(str(np.round(rv_new,2)))
        
        else:
            rv_new=0
            self.gui.lineEdit_auto_velocity_shift.setText('0')
               
        self.apply_velocity_shift(rv_new)

    def apply_velocity_shift(self,rv_new=0.0):
        
        if rv_new==0:
            rv_new=float(self.gui.lineEdit_auto_velocity_shift.text())
        new_xcurrent=np.copy(self.gui.xcurrent)+rv_new-self.gui.rv
        self.gui.xcurrent=new_xcurrent
        self.gui.rv=rv_new
        self.fit_spline()
        self.make_fig(0)
        self.make_fig(1)
        
        

# Normalizer

## Aim
The *normalizer* is an interactive 1D spectrum normalizer tool. It comes with two different spline fitting routines (*UnivariateSpline* and *LSQUnivariateSpline*) and offers the possibility to define masks (line windows), either user-defined ones or automatic masking via peak identification.

## Requirements
First of all, make sure to have **Python 3.X** and all other requirements installed. Currently, these are:

astropy (5.2.2), matplotlib (3.7.4), numpy (1.24.4), pandas (1.5.3), PyAstronomy (0.21.0), PyQt5 (5.15.10), PyQt5_sip (12.11.1), SciPy (1.10.1)

The numbers in brackets indicate module versions used during development. Earlier/later versions may work as well.

## Install
Download from GitHub or clone:
```
git clone https://github.com/astrojohannes/normalizer
```

## How to execute
Execute `python run_normalizer.py` or use additional options and arguments:
```
run_normalizer.py [--help] [--wave=min,max] [spectrum_file]
```

[*--help*] display help for basic usage

[*--wave=a,b*] use limits *a,b* for the wavelength range when calling together with [spectrum_file]

[spectrum_file] is a *.fits file containing a 1D spectrum, wavelength unit should be Angström

### Example
```
python run_normalizer.py --wave=6000,6800 5281_REDU_coadd_rebin.fits
```

## GUI Description of User Fields
The GUI contains two windows, one used to define the interactive user input and the second one to show the plots, i.e. the original spectrum and the normalized one. The fitting basically consists of four steps:

**i) Zoom-in to science part of spectrum**
The normalizer is designed to perform fits to the part of the spectrum that is currently shown (zoomed-in) in the plot window. Best practice is thus to first define the range of wavelengths used for science.

**ii) selection of fitting method and parameters**
Two methods are available for fitting (button **Fit continuum**) a spline (piecewise polynomials) to the input spectrum: *UnivariateSpline* and *LSQUnivariateSpline*. In both cases the user may adopt the **degree k** (1<=k<=5) of the spline polynomials and enter fixpoints. The **fixpoints** may be a single point at the wavelength axis or a comma-separated list of wavelengths. The weights for the fitting will be increased such that **the spline will intersect with the data at the fixpoints**. When using *UnivariateSpline*, one may adopt the **smooth parameter s**, which is a positive smoothing factor used to choose the number of knots. Knots are located where the polynomials of the spline connect. The number of knots will be increased until the smoothing condition is satisfied: sum((w[i] * (y[i]-spl(x[i])))**2) <= s. When using the method *LSQUnivariateSpline* the user may define the **number of knots t** directly. In this case every t-th point along the wavelength axis will be chosen as a knot point.

**iii) identification/masking of (strong) lines**
Masking of (strong) lines is necessary to avoid that the spline normalizes/removes spectral lines that are used for science. Two methods are available for the user. Either an automatic detection of peaks/ranges (button **Identify+mask lines using r.m.s.**) using the standard deviation (sigma) or a user-defined table of mask centers and widths (button **Manual mask using table**). In the former case, the user may adopt the **sigma high(h)/low(l)** values. The algorithm selects a line for masking when the following condition is satisfied: y[i]>h\*sigma and it will further expand the mask left- and rightwards along the wavelength axis until y[i]<l\*sigma.

**iv) determination of radial velocity**
To correct for (or determine) the radial velocity of the observed object and shift the spectrum to laboratory wavelengths, the normalizer may perform a cross-correlation (button **Determine vel. shift**) between the normalized spectrum and a solar spectrum and looks for peaks in the correlation function. Regions that are typically contaminated by telluric lines are avoided. The user may enter lower und upper **shift limits** given in units of km/s. The shift found from the cross-correlation will be shown in the input field next to the button. The user may accept and leave the number or overwrite it and use an own estimate before actually performing the correction (button **Shift spectrum***). Note that the shifts are always meant in relation to the original input spectrum, rather than any previously corrected one.

## Saving the normalized spectrum
Saving the result is done when pressing the **Save FITS** button. The basename of the output file will be the same as the input file, but with the wavelength-range of the current zoom indicated at the end of the filename.

#
#
# The code in this file was taken from iSpec, written by Sergi Blanco-Cuaresma (http://www.blancocuaresma.com/s/).
# Some parts were modified to be used in the "normalizer".
#
#
import numpy as np
from astropy.io import fits as pyfits
import os
import tempfile
import random
import gzip
import scipy
import copy
from mpfitmodels import GaussianModel, VoigtModel

def __improve_linemask_edges(xcoord, yvalues, base, top, peak):
    """
    Given a spectrum, the position of a peak and its limiting region where:
    - Typical shape: concave + convex + concave region.
    - Peak is located within the convex region, although it is just in the border with the concave region (first element).
    """
    # Try to use two additional position, since we are going to lose them
    # by doing the first and second derivative
    original_top = top
    top = np.min([top+2, len(xcoord)])
    y = yvalues[base:top+1]
    x = xcoord[base:top+1]
    # First derivative (positive => flux increases, negative => flux decreases)
    dy_dx = (y[:-1] - y[1:])/ (x[:-1] - x[1:])
    # Second derivative (positive => convex, negative => concave)
    d2y_dx2 = (dy_dx[:-1] - dy_dx[1:])/ (x[:-2] - x[2:])
    # Peak position inside the linemask region
    peak_relative_pos = peak - base
    # The peak should be in a convex region => the second derivative should be positive
    # - It may happen that the peak falls in the beginning/end of a concave region, accept also this cases
    if peak_relative_pos < len(d2y_dx2)-1 and peak_relative_pos > 0 and (d2y_dx2[peak_relative_pos-1] > 0 or d2y_dx2[peak_relative_pos] > 0 or d2y_dx2[peak_relative_pos+1] > 0):
        # Find the concave positions at both sides of the peak
        concave_pos = np.where(d2y_dx2<0)[0]
        if len(concave_pos) == 0:
            # This should not happen, but just in case...
            new_base = base
            new_top = original_top
        else:
            # Concave regions to the left of the peak
            left_concave_pos = concave_pos[concave_pos-peak_relative_pos < 0]
            if len(left_concave_pos) == 0:
                # This should not happen, but just in case...
                new_base = base
            else:
                # Find the edges of the concave regions to the left of the peak
                left_concave_pos_diff = left_concave_pos[:-1] - left_concave_pos[1:]
                left_concave_pos_diff = np.asarray([-1] + left_concave_pos_diff.tolist())
                left_concave_edge_pos = np.where(left_concave_pos_diff != -1)[0]
                if len(left_concave_edge_pos) == 0:
                    # There is only one concave region, we use its left limit
                    new_base = left_concave_pos[0] + base
                else:
                    # There is more than one concave region, select the nearest edge to the peak
                    new_base = np.max([left_concave_pos[np.max(left_concave_edge_pos)] + base, base])

            # Concave regions to the right of the peak
            right_concave_pos = concave_pos[concave_pos-peak_relative_pos > 0]
            if len(right_concave_pos) == 0:
                # This should not happen, but just in case...
                new_top = original_top
            else:
                # Find the edges of the concave regions to the right of the peak
                right_concave_pos_diff = right_concave_pos[1:] - right_concave_pos[:-1]
                right_concave_pos_diff = np.asarray(right_concave_pos_diff.tolist() + [1])
                right_concave_edge_pos = np.where(right_concave_pos_diff != 1)[0]
                if len(right_concave_edge_pos) == 0:
                    # There is only one concave region, we use its right limit
                    new_top = right_concave_pos[-1] + base
                else:
                    # There is more than one concave region, select the one with the nearest edge to the peak
                    new_top = np.min([right_concave_pos[np.min(right_concave_edge_pos)] + base, original_top])

    else:
        # This will happen very rarely (only in peaks detected at the extreme of a spectrum
        # and one of its basepoints has been "artificially" added and it happens to be
        # just next to the peak)
        new_base = base
        new_top = original_top
        #plt.plot(x, y)
        #l = plt.axvline(x = x[peak_relative_pos], linewidth=1, color='red')
        #print d2y_dx2, d2y_dx2[peak_relative_pos]
        #plt.show()

    return new_base, new_top

def __assert_structure(xcoord, yvalues, peaks, base_points):
    """
    Given a group of peaks and base_points with the following assumptions:
    - base_points[i] < base_point[i+1]
    - peaks[i] < peaks[i+1]
    - base_points[i] < peaks[i] < base_points[i+1]
    The function returns peaks and base_points where:
    - The first and last feature is a base point: base_points[0] < peaks[0] < base_points[1] < ... < base_points[n-1] < peaks[n-1] < base_points[n] where n = len(base_points)
    - len(base_points) = len(peaks) + 1
    """
    if len(peaks) == 0 or len(base_points) == 0:
        return [], []

    # Limit the base_points array to the ones that are useful, considering that
    # the first and last peak are always removed
    first_wave_peak = xcoord[peaks][0]
    first_wave_base = xcoord[base_points][0]
    if first_wave_peak > first_wave_base:
        if len(base_points) - len(peaks) == 1:
            ## First feature found in spectrum: base point
            ## Last feature found in spectrum: base point
            base_points = base_points
            peaks = peaks
        elif len(base_points) - len(peaks) == 0:
            ## First feature found in spectrum: base point
            ## Last feature found in spectrum: peak
            # - Remove last peak
            #base_points = base_points
            #peaks = peaks[:-1]
            # - Add a base point (last point in the spectrum)
            base_points = np.hstack((base_points, [len(xcoord)-1]))
            peaks = peaks
        else:
            raise Exception("This should not happen")
    else:
        if len(base_points) - len(peaks) == -1:
            ## First feature found in spectrum: peak
            ## Last feature found in spectrum: peak
            # - Remove first and last peaks
            #base_points = base_points
            #peaks = peaks[1:-1]
            # - Add two base points (first and last point in the spectrum)
            base_points = np.hstack(([0], base_points))
            base_points = np.hstack((base_points, [len(xcoord)-1]))
            peaks = peaks
        elif len(base_points) - len(peaks) == 0:
            ## First feature found in spectrum: peak
            ## Last feature found in spectrum: base point
            # - Remove first peak
            #base_points = base_points
            #peaks = peaks[1:]
            # - Add a base point (first point in the spectrum)
            base_points = np.hstack(([0], base_points))
            peaks = peaks
        else:
            raise Exception("This should not happen")

    return peaks, base_points

def __remove_consecutives_features(features):
    """
    Remove features (i.e. peaks or base points) that are consecutive, it makes
    no sense to have two peaks or two base points together.
    """
    if len(features) >= 2:
        duplicated_features = (np.abs(features[:-1] - features[1:]) == 1)
        duplicated_features = np.array([False] + duplicated_features.tolist())
        cleaned_features = features[~duplicated_features]
        return cleaned_features
    else:
        return features

def find_local_max_values(x):
    """
    For an array of values, find the position of local maximum values considering only
    the next and previous elements, except they have the same value.
    In that case, the next/previous different value is checked. Therefore,
    ::
        find_local_max([1,2,3,3,2,1,4,3])
    would return:
    ::
        [2, 3, 6]
    """
    ret = []
    n = len(x)
    m = 0;
    for i in np.arange(n):
        l_min = np.max([i-1, 0])
        #l_max = i-1
        #r_min = i+1
        #r_max = np.min([i+1, n-1])
        r_min = np.min([i+1, n-1])
        is_max = True

        # left side
        j = l_min
        # If value is equal, search for the last different value
        while j >= 0 and x[j] == x[i]:
            j -= 1

        if (j < 0 or x[j] > x[i]) and i > 0:
            is_max = False

        # right side
        if is_max:
            j = r_min
            # If value is equal, search for the next different value
            while j < n and x[j] == x[i]:
                j += 1
            if (j >= n or x[j] > x[i]) and i < n-1:
                is_max = False

        if is_max:
            ret.append(i)
    return np.asarray(ret)

def find_local_min_values(x):
    """
    For an array of values, find the position of local maximum values considering only
    the next and previous elements, except they have the same value.
    In that case, the next/previous different value is checked. Therefore,
    ::
        find_local_max([10,9,3,3,9,10,4,30])
    would return:
    ::
        [2, 3, 6]
    """
    ret = []
    n = len(x)
    m = 0;
    for i in np.arange(n):
        l_min = np.max([i-1, 0])
        #l_max = i-1
        #r_min = i+1
        #r_max = np.min([i+1, n-1])
        r_min = np.min([i+1, n-1])
        is_min = True
        # left side
        j = l_min
        # If value is equal, search for the last different value
        while j >= 0 and x[j] == x[i]:
            j -= 1

        if j < 0 or x[j] < x[i]:
            is_min = False

        # right side
        if is_min:
            j = r_min
            # If value is equal, search for the next different value
            while j < n and x[j] == x[i]:
                j += 1

            if j >= n or x[j] < x[i]:
                is_min = False

        if is_min:
            ret.append(i)
    return np.asarray(ret)

def __find_peaks_and_base_points(xcoord, yvalues):
    """
    Find peaks and base points. It works better with a smoothed spectrum (i.e. convolved using 2*resolution).
    """
    if len(yvalues[~np.isnan(yvalues)]) == 0 or len(yvalues[~np.isnan(xcoord)]) == 0:
        #raise Exception("Not enough data for finding peaks and base points")
        print("WARNING: Not enough data for finding peaks and base points")
        peaks = []
        base_points = []
    else:
        # Determine peaks and base points (also known as continuum points)
        peaks = find_local_min_values(yvalues)
        base_points = find_local_max_values(yvalues)

        # WARNING: Due to three or more consecutive values with exactly the same flux
        # find_local_max_values or find_local_min_values will identify all of them as peaks or bases,
        # where only one of the should be marked as peak or base.
        # These cases break the necessary condition of having the same number of
        # peaks and base_points +/-1
        # It is necessary to find those "duplicates" and remove them:
        peaks = __remove_consecutives_features(peaks)
        base_points = __remove_consecutives_features(base_points)

        if not (len(peaks) - len(base_points)) in [-1, 0, 1]:
            raise Exception("This should not happen")

        # Make sure that
        peaks, base_points = __assert_structure(xcoord, yvalues, peaks, base_points)

    return peaks, base_points

def find_duplicates(a, key):
    """
    Find duplicates in a column of a recarray. This is a simplified version of:
    ::
        import numpy.lib.recfunctions as rfn
        rfn.find_duplicates(...)
    """
    a = np.asanyarray(a).ravel()
    # Get the sorting data (by selecting the corresponding field)
    base = a[key]
    # Get the sorting indices and the sorted data
    sortidx = base.argsort()
    sorteddata = base[sortidx]
    # Compare the sorting data
    flag = (sorteddata[:-1] == sorteddata[1:])
    flag = np.concatenate(([False], flag))
    # We need to take the point on the left as well (else we're missing it)
    flag[:-1] = flag[:-1] + flag[1:]
    duplicates = a[sortidx][flag]
    duplicates_index = sortidx[flag]
    return (duplicates, duplicates_index)

def create_spectrum_structure(waveobs, flux=None, err=None):
    """
    Create spectrum structure
    """
    spectrum = np.recarray((len(waveobs), ), dtype=[('waveobs', float),('flux', float),('err', float)])
    spectrum['waveobs'] = waveobs

    if flux is not None:
        spectrum['flux'] = flux
    else:
        spectrum['flux'] = 0.0

    if err is not None:
        spectrum['err'] = err
    else:
        spectrum['err'] = 0.0

    return spectrum

def __read_fits_spectrum(spectrum_filename):
    """
    Reads the 'PRIMARY' HDU of the FITS file, considering that it contains the fluxes.
    The wavelength are derived from the headers, if not possible it checks if the
    data from the HDU contains 2 axes and takes the first as the wavelength
    and the second as the flux.
    It tries to find the errors in other HDU by checking the names and the length,
    if none are found then they are set to zero.
    Inspired by pyspeckit:
        https://bitbucket.org/pyspeckit/pyspeckit.bitbucket.org/src/ae1e0714410b58905466740b04b54318d5f318f8/pyspeckit/spectrum/readers/fits_reader.py?at=default
    Finally, if nothing has worked, it searches for a binary table with 3 columns
    'AWAV', 'FLUXES' and 'SIGMA'.
    """
    hdulist = pyfits.open(spectrum_filename)

    data = hdulist['PRIMARY'].data
    hdr = hdulist['PRIMARY'].header

    if data is not None and (type(hdulist['PRIMARY']) is pyfits.hdu.image.PrimaryHDU or \
                                type(hdulist['PRIMARY']) is pyfits.hdu.image.ImageHDU):
        axis = 1
        specaxis = str(axis)
        # Try to determine if wavelength is in Angstrom (by default) or nm
        ctype = hdr.get('CTYPE%i' % axis)
        cunit = hdr.get('CUNIT%i' % axis)
        if str(cunit).upper() in ['NM']:
            unit = "nm"
        else:
            #if str(ctype).upper() in ['AWAV', 'ANGSTROM', '0.1 NM'] or str(cunit).upper() in ['AWAV', 'ANGSTROM', '0.1 NM']:
            unit = "Angstrom"

        flux = data.flatten()
        waveobs = None

        # Try to read World Coordinate System (WCS) that defines the wavelength grid
        if hdr.get(str('CD%s_%s' % (specaxis,specaxis))) is not None:
            wave_step = hdr['CD%s_%s' % (specaxis,specaxis)]
            wave_base = hdr['CRVAL%s' % (specaxis)]
            reference_pixel = hdr['CRPIX%s' % (specaxis)]
        elif hdr.get(str('CDELT%s' % (specaxis))) is not None:
            wave_step = hdr['CDELT%s' % (specaxis)]
            wave_base = hdr['CRVAL%s' % (specaxis)]
            reference_pixel = hdr['CRPIX%s' % (specaxis)]
        elif len(data.shape) > 1:
            # No valid WCS, try assuming first axis is the wavelength axis
            if hdr.get('CUNIT%s' % (specaxis)) is not None:
                waveobs = data[0,:]
                flux = data[1,:]
                if data.shape[0] > 2:
                    errspec = data[2,:]
            else:
                raise Exception("Unknown FITS file format")
        else:
            raise Exception("Unknown FITS file format")


        # Deal with logarithmic wavelength binning if necessary
        if waveobs is None:
            if hdr.get('WFITTYPE') == 'LOG-LINEAR':
                xconv = lambda v: 10**((v-reference_pixel+1)*wave_step+wave_base)
                waveobs = xconv(np.arange(len(flux)))
                # Angstrom to nm
                if unit != "nm":
                    waveobs /= 10
                print("Log scale")
            else:
                # Angstrom to nm
                if unit != "nm":
                    wave_base /= 10
                    wave_step /= 10
                xconv = lambda v: ((v-reference_pixel+1)*wave_step+wave_base)
                waveobs = xconv(np.arange(len(flux)))

        num_measures = len(flux)
        spectrum = create_spectrum_structure(waveobs, flux)

        # Try to find the errors in the extensions (HDU different than the PRIMARY):
        spectrum['err'] = np.zeros(len(flux))
        for i in range(len(hdulist)):
            name = hdulist[i].name.upper()
            if name == str('PRIMARY') or len(hdulist[i].data.flatten()) != len(flux) or type(hdulist[i]) is pyfits.hdu.table.BinTableHDU:
                continue
            if 'IVAR' in name or 'IVARIANCE' in name:
                spectrum['err'] = np.sqrt(1. / hdulist[i].data.flatten()) # Not sure
                #spectrum['err'] = 1. / hdulist[i].data.flatten()
                break
            elif 'VAR' in name or 'VARIANCE' in name:
                spectrum['err'] = np.sqrt(hdulist[i].data.flatten()) # Not sure
                #spectrum['err'] = hdulist[i].data.flatten()
                break
            elif 'NOISE' in name or 'ERR' in name or 'SIGMA' in name:
                spectrum['err'] = hdulist[i].data.flatten()
                break

    elif data is None:
        # Try to find a binary table with an irregular spectra and 3 columns
        spectrum = None
        for i in range(len(hdulist)):
            if type(hdulist[i]) is pyfits.hdu.table.BinTableHDU:
                data = hdulist[i].data
                if len(data) == 1:
                    # Sometimes we have an array inside another as a single element
                    data = data[0]
                # iSpec binary table for irregular spectra
                waveobs = None
                for key in ('AWAV', 'WAVE'):
                    try:
                        waveobs = data[key]
                    except:
                        continue
                    else:
                        break
                flux = None
                for key in ('FLUX',):
                    try:
                        flux = data[key]
                    except:
                        continue
                    else:
                        break
                error = None
                for key in ('ERR', 'SIGMA', 'NOISE'):
                    try:
                        error = data[key]
                    except:
                        continue
                    else:
                        break

                if waveobs is not None and flux is not None:
                    try:
                        spectrum = create_spectrum_structure(waveobs, flux, error)
                        break
                    except:
                        continue
                else:
                    continue
        if spectrum is None:
            raise Exception("Unknown FITS file format")
    else:
        raise Exception("Unknown FITS file format")

    hdulist.close()

    return spectrum

def __read_spectrum(spectrum_filename):
    try:
        spectrum = np.array([tuple(line.rstrip('\r\n').split("\t")) for line in open(spectrum_filename,)][1:], dtype=[('waveobs', float),('flux', float),('err', float)])
        if len(spectrum) == 0:
            raise Exception("Empty spectrum or incompatible format")
    except Exception as err:
        # try narval plain text format:
        # - ignores 2 first lines (header)
        # - ignores last line (empty)
        # - lines separated by \r
        # - columns separated by space
        try:
            narval = open(spectrum_filename,).readlines()[0].split('\r')
            spectrum = np.array([tuple(line.rstrip('\r').split()) for line in narval[2:-1]], dtype=[('waveobs', float),('flux', float),('err', float)])
            if len(spectrum) == 0:
                raise Exception("Empty spectrum or incompatible format")
        except:
            # try espadons plain text format:
            # - ignores 2 first lines (header)
            # - columns separated by space
            espadons = open(spectrum_filename,).readlines()
            spectrum = np.array([tuple(line.rstrip('\r').split()) for line in espadons[2:]], dtype=[('waveobs', float),('flux', float),('err', float)])

    if len(spectrum) == 0:
        raise Exception("Empty spectrum or incompatible format")
    return spectrum

def read_spectrum(spectrum_filename, apply_filters=True, sort=True, regions=None):
    """
    Return spectrum recarray structure from a filename.
    The file format shouldd be plain text files with **tab** character as column delimiter.
    Three columns should exists: wavelength, flux and error (although this last one is not a relevant value
    for the editor and it can be set all to zero).
    The first line should contain the header names 'waveobs', 'flux' and 'err' such as in the following example:
    ::
        waveobs       flux          err
        370.000000000 1.26095742505 1.53596736433
        370.001897436 1.22468868618 1.55692475754
        370.003794872 1.18323884263 1.47304952231
        370.005692308 1.16766911881 1.49393329036
    To save space, the file can be compressed in gzip format.
    If the specified file does not exists, it checks if there is a compressed version
    with the extension '.gz' (gzip) and if it exists, it will be automatically uncompressed.
    ** It can recognise FITS files by the filename (extensions .FITS or .FIT), if this is
    the case, then it tries to load the PRIMARY spectra by default and tries to search the errors
    in the extensions of the FITS file. In case the PRIMARY is empty, it searches for binary tables
    with wavelengths, fluxes and errors.
    """
    # If it is not compressed
    if os.path.exists(spectrum_filename) and (spectrum_filename[-4:].lower() == ".fit" or spectrum_filename[-5:].lower() == ".fits" or \
            spectrum_filename[-7:].lower() == ".fit.gz" or spectrum_filename[-8:].lower() == ".fits.gz"):
        spectrum = __read_fits_spectrum(spectrum_filename)
    elif os.path.exists(spectrum_filename) and spectrum_filename[-3:].lower() != ".gz":
        spectrum = __read_spectrum(spectrum_filename)
    elif (os.path.exists(spectrum_filename) and spectrum_filename[-3:].lower() == ".gz") or (os.path.exists(spectrum_filename.lower() + ".gz")):
        if spectrum_filename[-3:] != ".gz":
            spectrum_filename = spectrum_filename + ".gz"

        tmp_spec = tempfile.mktemp() + str(int(random.random() * 100000000))
        # Uncompress to a temporary file
        f_out = open(tmp_spec, 'wb')
        f_in = gzip.open(spectrum_filename, 'rb')
        f_out.writelines(f_in)
        f_out.close()
        f_in.close()

        spectrum = __read_spectrum(tmp_spec)
        os.remove(tmp_spec)
    else:
        raise Exception("Spectrum file does not exists: %s" %(spectrum_filename))

    if apply_filters:
        # Filtering...
        valid = ~np.isnan(spectrum['flux'])

        if len(spectrum[valid]) > 2:
            # Find duplicate wavelengths
            dups, dups_index = find_duplicates(spectrum, 'waveobs')

            # Filter all duplicates except the first one
            last_wave = None
            for i in np.arange(len(dups)):
                if last_wave is None:
                    last_wave = dups[i]['waveobs']
                    continue
                if last_wave == dups[i]['waveobs']:
                    pos = dups_index[i]
                    valid[pos] = False
                else:
                    # Do not filter the first duplicated value
                    last_wave = dups[i]['waveobs']

        # Filter invalid and duplicated values
        spectrum = spectrum[valid]

    if sort:
        spectrum.sort(order='waveobs') # Make sure it is ordered by wavelength

    if regions is not None:
        # Wavelengths to be considered: segments
        wfilter = create_wavelength_filter(spectrum, regions=regions)
        spectrum = spectrum[wfilter]

    return spectrum

############## [start] Radial velocity

def __cross_correlation_function_template(spectrum, template, lower_velocity_limit, upper_velocity_limit, velocity_step, frame=None):
    """
    Calculates the cross correlation value between the spectrum and the specified template
    by shifting the template from lower to upper velocity.
    - The spectrum and the template should be uniformly spaced in terms of velocity (which
      implies non-uniformly distributed in terms of wavelength).
    - The velocity step used for the construction of the template should be the same
      as the one specified in this function.
    - The lower/upper/step velocity is only used to determine how many shifts
      should be done (in array positions) and return a velocity grid.
    """

    last_reported_progress = -1
    if frame is not None:
        frame.update_progress(0)

    # Speed of light in m/s
    c = 299792458.0

    velocity = np.arange(lower_velocity_limit, upper_velocity_limit+velocity_step, velocity_step)
    # 1 shift = 0.5 km/s (or the specified value)
    shifts = np.int32(velocity/ velocity_step)

    num_shifts = len(shifts)
    # Cross-correlation function
    ccf = np.zeros(num_shifts)
    ccf_err = np.zeros(num_shifts)
    depth = np.abs(np.max(template['flux']) - template['flux'])
    for i, vel in enumerate(velocity):
        factor = np.sqrt((1.-(vel*1000.)/c)/(1.+(vel*1000./c)))
        shifted_template = np.interp(spectrum['waveobs'], template['waveobs']/factor, depth, left=0.0, right=0.0)
        ccf[i] = np.correlate(spectrum['flux'], shifted_template)[0]
        ccf_err[i] = np.correlate(spectrum['err'], shifted_template)[0] # Propagate errors

    max_ccf = np.max(ccf)
    ccf = ccf/max_ccf # Normalize
    ccf_err = ccf_err/max_ccf # Propagate errors

    return velocity, ccf, ccf_err


def _sampling_uniform_in_velocity(wave_base, wave_top, velocity_step):
    """
    Create a uniformly spaced grid in terms of velocity:
    - An increment in position (i => i+1) supposes a constant velocity increment (velocity_step).
    - An increment in position (i => i+1) does not implies a constant wavelength increment.
    - It is uniform in log(wave) since:
          Wobs = Wrest * (1 + Vr/c)^[1,2,3..]
          log10(Wobs) = log10(Wrest) + [1,2,3..] * log10(1 + Vr/c)
      The last term is constant when dealing with wavelenght in log10.
    - Useful for building the cross correlate function used for determining the radial velocity of a star.
    """
    # Speed of light
    c = 299792.4580 # km/s
    #c = 299792458.0 # m/s

    ### Numpy optimized:
    # number of elements to go from wave_base to wave_top in increments of velocity_step
    i = int(np.ceil( (c * (wave_top - wave_base))/ (wave_base*velocity_step)))
    grid = wave_base * np.power((1 + (velocity_step/ c)), np.arange(i)+1)

    # Ensure wavelength limits since the "number of elements i" tends to be overestimated
    wfilter = grid <= wave_top
    grid = grid[wfilter]

    ### Non optimized:
    #grid = []
    #next_wave = wave_base
    #while next_wave <= wave_top:
        #grid.append(next_wave)
        ### Newtonian version:
        #next_wave = next_wave + next_wave * ((velocity_step) / c) # nm
        ### Relativistic version:
        ##next_wave = next_wave + next_wave * (1.-np.sqrt((1.-(velocity_step*1000.)/c)/(1.+(velocity_step*1000.)/c)))

    return np.asarray(grid)

def __cross_correlation_function_uniform_in_velocity(spectrum, mask, lower_velocity_limit, upper_velocity_limit, velocity_step, mask_size=2.0, mask_depth=0.01, template=False, fourier=False, frame=None):
    """
    Calculates the cross correlation value between the spectrum and the specified mask
    by shifting the mask from lower to upper velocity.
    - The spectrum and the mask should be uniformly spaced in terms of velocity (which
      implies non-uniformly distributed in terms of wavelength).
    - The velocity step used for the construction of the mask should be the same
      as the one specified in this function.
    - The lower/upper/step velocity is only used to determine how many shifts
      should be done (in array positions) and return a velocity grid.
    If fourier is set, the calculation is done in the fourier space. More info:
        VELOCITIES FROM CROSS-CORRELATION: A GUIDE FOR SELF-IMPROVEMENT
        CARLOS ALLENDE PRIETO
        http://iopscience.iop.org/1538-3881/134/5/1843/fulltext/205881.text.html
        http://iopscience.iop.org/1538-3881/134/5/1843/fulltext/sourcecode.tar.gz
    """

    last_reported_progress = -1
    if frame is not None:
        frame.update_progress(0)

    # Speed of light in m/s
    c = 299792458.0

    # 1 shift = 1.0 km/s (or the specified value)
    shifts = np.arange(np.int32(np.floor(lower_velocity_limit)/velocity_step), np.int32(np.ceil(upper_velocity_limit)/velocity_step)+1)
    velocity = shifts * velocity_step

    waveobs = _sampling_uniform_in_velocity(np.min(spectrum['waveobs']), np.max(spectrum['waveobs']), velocity_step)
    flux = np.interp(waveobs, spectrum['waveobs'], spectrum['flux'], left=0.0, right=0.0)
    err = np.interp(waveobs, spectrum['waveobs'], spectrum['err'], left=0.0, right=0.0)


    if template:
        depth = np.abs(np.max(mask['flux']) - mask['flux'])
        resampled_mask = np.interp(waveobs, mask['waveobs'], depth, left=0.0, right=0.0)
    else:
        selected = __select_lines_for_mask(mask, minimum_depth=mask_depth, velocity_mask_size = mask_size, min_velocity_separation = 1.0)
        resampled_mask = __create_mask(waveobs, mask['wave_peak'][selected], mask['depth'][selected], velocity_mask_size=mask_size)

    if fourier:
        # Transformed flux and mask
        tflux = fft(flux)
        tresampled_mask = fft(resampled_mask)
        conj_tresampled_mask = np.conj(tresampled_mask)
        num = len(resampled_mask)/2+1
        tmp = abs(ifft(tflux*conj_tresampled_mask))
        ccf = np.hstack((tmp[num:], tmp[:num]))

        # Transformed flux and mask powered by 2 (second)
        #ccf_err = np.zeros(len(ccf))
        # Conservative error propagation
        terr = fft(err)
        tmp = abs(ifft(terr*conj_tresampled_mask))
        ccf_err = np.hstack((tmp[num:], tmp[:num]))
        ## Error propagation
        #tflux_s = fft(np.power(flux, 2))
        #tresampled_mask_s = fft(np.power(resampled_mask, 2))
        #tflux_err_s = fft(np.power(err, 2))
        #tresampled_mask_err_s = fft(np.ones(len(err))*0.05) # Errors of 5% for masks

        #tmp = abs(ifft(tflux_s*np.conj(tresampled_mask_err_s)))
        #tmp += abs(ifft(tflux_err_s*np.conj(tresampled_mask_s)))
        #ccf_err = np.hstack((tmp[num:], tmp[:num]))
        #ccf_err = np.sqrt(ccf_err)

        # Velocities
        velocities = velocity_step * (np.arange(len(resampled_mask), dtype=float)+1 - num)

        # Filter to area of interest
        xfilter = np.logical_and(velocities >= lower_velocity_limit, velocities <= upper_velocity_limit)
        ccf = ccf[xfilter]
        ccf_err = ccf_err[xfilter]
        velocities = velocities[xfilter]
    else:
        num_shifts = len(shifts)
        # Cross-correlation function
        ccf = np.zeros(num_shifts)
        ccf_err = np.zeros(num_shifts)

        for shift, i in zip(shifts, np.arange(num_shifts)):
            #shifted_mask = resampled_mask
            if shift == 0:
                shifted_mask = resampled_mask
            elif shift > 0:
                #shifted_mask = np.hstack((shift*[0], resampled_mask[:-1*shift]))
                shifted_mask = np.hstack((resampled_mask[-1*shift:], resampled_mask[:-1*shift]))
            else:
                #shifted_mask = np.hstack((resampled_mask[-1*shift:], -1*shift*[0]))
                shifted_mask = np.hstack((resampled_mask[-1*shift:], resampled_mask[:-1*shift]))
            #ccf[i] = np.correlate(flux, shifted_mask)[0]
            #ccf_err[i] = np.correlate(err, shifted_mask)[0] # Propagate errors
            ccf[i] = np.average(flux*shifted_mask)
            ccf_err[i] = np.average(err*shifted_mask) # Propagate errors
            #ccf[i] = np.average(np.tanh(flux*shifted_mask))
            #ccf_err[i] = np.average(np.tanh(err*shifted_mask)) # Propagate errors

    max_ccf = np.max(ccf)
    ccf = ccf/max_ccf # Normalize
    ccf_err = ccf_err/max_ccf # Propagate errors

    return velocity, ccf, ccf_err, len(flux)



def create_filter_for_regions_affected_by_tellurics(wavelengths, linelist_telluric, min_velocity=-30.0, max_velocity=30.0, frame=None):
    """
    Returns a boolean array of the same size of wavelengths. True will be assigned
    to those positions thay may be affected by telluric lines in a range from
    min_velocity to max_velocity
    """
    # Light speed in vacuum
    c = 299792458.0 # m/s

    tfilter = wavelengths == np.nan
    tfilter2 = wavelengths == np.nan
    wave_bases = linelist_telluric['wave_peak'] * np.sqrt((1.-(max_velocity*1000.)/c)/(1.+(max_velocity*1000.)/c))
    wave_tops = linelist_telluric['wave_peak'] * np.sqrt((1.-(min_velocity*1000.)/c)/(1.+(min_velocity*1000.)/c))
    last_reported_progress = -1
    total_regions = len(wave_bases)
    last = 0 # Optimization
    for i, (wave_base, wave_top) in enumerate(zip(wave_bases, wave_tops)):
        begin = wavelengths[last:].searchsorted(wave_base)
        end = wavelengths[last:].searchsorted(wave_top)
        tfilter[last+begin:last+end] = True
        #wfilter = np.logical_and(wavelengths >= wave_base, wavelengths <= wave_top)
        #tfilter = np.logical_or(wfilter, tfilter)

        last += end
    return tfilter



try:
    import pyximport
    import numpy as np
    pyximport.install(setup_args={'include_dirs':[np.get_include()]})
    from .lines_c import create_mask as __create_mask
except:

    def __create_mask(spectrum_wave, mask_wave, mask_values, velocity_mask_size=2.0):
        """
        It constructs a zero flux spectrum and assign mask values to the wavelengths
        belonging to that value and its surounds (determined by the velocity_mask_size).
        """
        ## Speed of light in m/s
        c = 299792458.0

        resampled_mask = np.zeros(len(spectrum_wave))

        # Mask limits
        mask_wave_step = (mask_wave * (1.-np.sqrt((1.-(velocity_mask_size*1000.)/c)/(1.+(velocity_mask_size*1000.)/c))))/2.0
        mask_wave_base = mask_wave - 1*mask_wave_step
        mask_wave_top = mask_wave + 1*mask_wave_step

        i = 0
        j = 0
        for i in range(len(mask_wave)):
            #j = 0
            while j < len(spectrum_wave) and spectrum_wave[j] < mask_wave_base[i]:
                j += 1
            while j < len(spectrum_wave) and spectrum_wave[j] >= mask_wave_base[i] and spectrum_wave[j] <= mask_wave_top[i]:
                resampled_mask[j] = mask_values[i]
                j += 1

        return resampled_mask

def __select_lines_for_mask(linemasks, minimum_depth=0.01, velocity_mask_size = 2.0, min_velocity_separation = 1.0):
    """
    Select the lines that are goint to be used for building a mask for doing
    cross-correlation. It filters by depth and validate that the lines are
    suficiently apart from its neighbors to avoid overlapping.
    For that purpose, 'velocity_mask_size' represents the masks size in km/s
    around the peak and optionally, 'min_velocity_separation' indicates the
    minimum separation needed between two consecutive masks.
    It returns a boolean array indicating what lines have been finally selected.
    """
    total_velocity_separation = velocity_mask_size + min_velocity_separation / 2.0
    selected = linemasks['depth'] >= minimum_depth

    ## Speed of light in m/s
    c = 299792458.0

    # Mask limits
    mask_wave_step = (linemasks['wave_peak'] * (1.-np.sqrt((1.-(total_velocity_separation*1000.)/c)/(1.+(total_velocity_separation*1000.)/c))))/2.0
    mask_wave_base = linemasks['wave_peak'] - 1*mask_wave_step
    mask_wave_top = linemasks['wave_peak'] + 1*mask_wave_step
    #mask_wave_base = linemasks['wave_base'] - 1*mask_wave_step
    #mask_wave_top = linemasks['wave_top'] + 1*mask_wave_step

    i = 0
    while i < len(linemasks):
        if selected[i]:
            # Right
            r = i
            max_r = r
            max_depth_r = linemasks['depth'][i]
            while r < len(linemasks) and mask_wave_base[r] <= mask_wave_top[i]:
                if selected[r] and linemasks['depth'][r] > max_depth_r:
                    max_depth_r = linemasks['depth'][r]
                    max_r = r
                r += 1
            # Left
            l = i
            max_l = l
            max_depth_l = linemasks['depth'][i]
            while l >= 0 and mask_wave_top[l] >= mask_wave_base[i]:
                if selected[l] and linemasks['depth'][l] > max_depth_l:
                    max_depth_l = linemasks['depth'][l]
                    max_l = l
                l -= 1

            if i - 1 == l and i + 1 == r:
                # No conflict
                i += 1
            else:
                #print "*",
                if i + 1 != l and i - 1 != r:
                    #print "both", i, i - l, r - i
                    for x in range(r - i):
                        selected[i+x] = False
                    for x in range(i - l):
                        selected[i-x] = False
                    if max_depth_l > max_depth_r:
                        selected[max_l] = True
                    else:
                        selected[max_r] = True
                elif i + 1 != l:
                    #print "left"
                    for x in range(i - l):
                        selected[i-x] = False
                    selected[max_l] = True
                else:
                    #print "right"
                    for x in range(r - i):
                        selected[i+x] = False
                    selected[max_r] = True
                i = r
        else:
            i += 1
    return selected



def cross_correlate_with_mask(spectrum, linelist, lower_velocity_limit=-200, upper_velocity_limit=200, velocity_step=1.0, mask_size=None, mask_depth=0.01, fourier=False, only_one_peak=False, model='2nd order polynomial + gaussian fit', peak_probability=0.75, frame=None):
    """
    Determines the velocity profile by cross-correlating the spectrum with
    a mask built from a line list mask.
    If mask_size is not specified, the double of the velocity step will be taken,
    which generally it is the recommended value.
    :returns:
        - Array with fitted gaussian models sorted by depth (deepest at position 0)
        - CCF structure with 'x' (velocities), 'y' (relative intensities), 'err'
    """
    if mask_size is None:
        mask_size = 2*velocity_step # Recommended
    return __cross_correlate(spectrum, linelist=linelist, template=None, \
                lower_velocity_limit=lower_velocity_limit, upper_velocity_limit = upper_velocity_limit, \
                velocity_step=velocity_step, \
                mask_size=mask_size, mask_depth=mask_depth, fourier=fourier, \
                only_one_peak=only_one_peak, peak_probability=peak_probability, model=model, \
                frame=None)

def cross_correlate_with_template(spectrum, template, lower_velocity_limit=-200, upper_velocity_limit=200, velocity_step=1.0, fourier=False, only_one_peak=False, model='2nd order polynomial + gaussian fit', peak_probability=0.75, frame=None):
    """
    Determines the velocity profile by cross-correlating the spectrum with
    a spectrum template.
    :returns:
        - Array with fitted gaussian models sorted by depth (deepest at position 0)
        - CCF structure with 'x' (velocities), 'y' (relative intensities), 'err'
    """
    return __cross_correlate(spectrum, linelist=None, template=template, \
            lower_velocity_limit=lower_velocity_limit, upper_velocity_limit = upper_velocity_limit, \
            velocity_step=velocity_step, \
            mask_size=None, mask_depth=None, fourier=fourier, \
            only_one_peak=only_one_peak, peak_probability=peak_probability, model=model, \
            frame=None)

def __cross_correlate(spectrum, linelist=None, template=None, lower_velocity_limit = -200, upper_velocity_limit = 200, velocity_step=1.0, mask_size=2.0, mask_depth=0.01, fourier=False, only_one_peak=False, peak_probability=0.75, model='2nd order polynomial + gaussian fit', frame=None):
    ccf, nbins = __build_velocity_profile(spectrum, \
            linelist = linelist, template = template, \
            lower_velocity_limit = lower_velocity_limit, upper_velocity_limit = upper_velocity_limit, \
            velocity_step=velocity_step, \
            mask_size=mask_size, mask_depth=mask_depth, \
            fourier=fourier, frame=frame)

    models = __model_velocity_profile(ccf, nbins, only_one_peak=only_one_peak, \
                                            peak_probability=peak_probability, model=model)
    # We have improved the peak probability detection using RLM, a priori it is not needed
    # this best selection:
    #best = select_good_velocity_profile_models(models, ccf)
    #return models[best], ccf
    return models, ccf

def __build_velocity_profile(spectrum, linelist=None, template=None, lower_velocity_limit = -200, upper_velocity_limit = 200, velocity_step=1.0, mask_size=2.0, mask_depth=0.01, fourier=False, frame=None):
    """
    Determines the velocity profile by cross-correlating the spectrum with:
    * a mask built from a line list if linelist is specified
    * a spectrum template if template is specified
    :returns:
        CCF structure with 'x' (velocities), 'y' (relative intensities), 'err'
        together with the number of spectrum's bins used in the cross correlation.
    """
    if linelist is not None:

        linelist = linelist[linelist['depth'] > 0.01]
        lfilter = np.logical_and(linelist['wave_peak'] >= np.min(spectrum['waveobs']), linelist['wave_peak'] <= np.max(spectrum['waveobs']))
        linelist = linelist[lfilter]

        velocity, ccf, ccf_err, nbins = __cross_correlation_function_uniform_in_velocity(spectrum, linelist, lower_velocity_limit, upper_velocity_limit, velocity_step, mask_size=mask_size, mask_depth=mask_depth, fourier=fourier, frame=frame)
    elif template is not None:
        ## Obtain the cross-correlate function by shifting the template
        velocity, ccf, ccf_err, nbins = __cross_correlation_function_uniform_in_velocity(spectrum, template, lower_velocity_limit, upper_velocity_limit, velocity_step, template=True, fourier=False, frame=frame)
        #velocity, ccf, ccf_err = __cross_correlation_function_template(spectrum, template, lower_velocity_limit = lower_velocity_limit, upper_velocity_limit=upper_velocity_limit, velocity_step = velocity_step, frame=frame)

    else:
        raise Exception("A linelist or template should be specified")

    ccf_struct = np.recarray((len(velocity), ), dtype=[('x', float),('y', float), ('err', float)])
    ccf_struct['x'] = velocity
    ccf_struct['y'] = ccf
    ccf_struct['err'] = ccf_err
    return ccf_struct, nbins


def __model_velocity_profile(ccf, nbins, only_one_peak=False, peak_probability=0.55, model='2nd order polynomial + gaussian fit'):
    """
    Fits a model ('Gaussian' or 'Voigt') to the deepest peaks in the velocity
    profile. If it is 'Auto', a gaussian and a voigt will be fitted and the best
    one used.
    In all cases, the peak is located by fitting a 2nd degree polynomial. Afterwards,
    the gaussian/voigt fitting is done for obtaining more info (such as sigma, etc.)
    * For Radial Velocity profiles, more than 1 outlier peak implies that the star is a spectroscopic binary.
    WARNING: fluxes and errors are going to be modified by a linear normalization process
    Detected peaks are evaluated to discard noise, a probability is assigned to each one
    in function to a linear model. If more than one peak is found, those with a peak probability
    lower than the specified by the argument will be discarded.
    :returns:
        Array of fitted models and an array with the margin errors for model.mu() to be able to know the interval
        of 99% confiance.
    """
    models = []
    if len(ccf) == 0:
        return models
    xcoord = ccf['x']
    fluxes = ccf['y']
    errors = ccf['err']

    # Smooth flux
    sig = 1
    smoothed_fluxes = scipy.ndimage.filters.gaussian_filter1d(fluxes, sig)
    #smoothed_fluxes = fluxes
    # Finding peaks and base points
    peaks, base_points = __find_peaks_and_base_points(xcoord, smoothed_fluxes)

    if len(peaks) == 0 or len(base_points) == 0:
        return models

    if len(peaks) != 0:
        base = base_points[:-1]
        top = base_points[1:]
        # Adjusting edges
        new_base = np.zeros(len(base), dtype=int)
        new_top = np.zeros(len(base), dtype=int)
        for i in np.arange(len(peaks)):
            new_base[i], new_top[i] = __improve_linemask_edges(xcoord, smoothed_fluxes, base[i], top[i], peaks[i])
            #new_base[i] = base[i]
            #new_top[i] = top[i]
        base = new_base
        top = new_top

        if only_one_peak:
            # Just try with the deepest line
            selected_peaks_indices = []
        else:
            import statsmodels.api as sm
            #x = np.arange(len(peaks))
            #y = fluxes[peaks]
            x = xcoord
            y = fluxes
            # RLM (Robust least squares)
            # Huber's T norm with the (default) median absolute deviation scaling
            # - http://en.wikipedia.org/wiki/Huber_loss_function
            # - options are LeastSquares, HuberT, RamsayE, AndrewWave, TrimmedMean, Hampel, and TukeyBiweight
            x_c = sm.add_constant(x, prepend=False) # Add a constant (1.0) to have a parameter base
            huber_t = sm.RLM(y, x_c, M=sm.robust.norms.HuberT())
            linear_model = huber_t.fit()
            selected_peaks_indices = np.where(linear_model.weights[peaks] < 1. - peak_probability)[0]

        if len(selected_peaks_indices) == 0:
            # Try with the deepest line
            sorted_peak_indices = np.argsort(fluxes[peaks])
            selected_peaks_indices = [sorted_peak_indices[0]]
        else:
            # Sort the interesting peaks from more to less deep
            sorted_peaks_indices = np.argsort(fluxes[peaks[selected_peaks_indices]])
            selected_peaks_indices = selected_peaks_indices[sorted_peaks_indices]
    else:
        # If no peaks found, just consider the deepest point and mark the base and top
        # as the limits of the whole data
        sorted_fluxes_indices = np.argsort(fluxes)
        peaks = sorted_fluxes_indices[0]
        base = 0
        top = len(xcoord) - 1
        selected_peaks_indices = [0]

    for i in np.asarray(selected_peaks_indices):
        #########################################################
        ####### 2nd degree polinomial fit to determine the peak
        #########################################################
        poly_step = 0.01
        # Use only 9 points for fitting (4 + 1 + 4)
        diff_base = peaks[i] - base[i]
        diff_top = top[i] - peaks[i]
        if diff_base > 4 and diff_top > 4:
            poly_base = peaks[i] - 4
            poly_top = peaks[i] + 4
        else:
            # There are less than 9 points but let's make sure that there are
            # the same number of point in each side to avoid asymetries that may
            # affect the fitting of the center
            if diff_base >= diff_top:
                poly_base = peaks[i] - diff_top
                poly_top = peaks[i] + diff_top
            elif diff_base < diff_top:
                poly_base = peaks[i] - diff_base
                poly_top = peaks[i] + diff_base
        p = np.poly1d(np.polyfit(xcoord[poly_base:poly_top+1], fluxes[poly_base:poly_top+1], 2))
        poly_vel = np.arange(xcoord[poly_base], xcoord[poly_top]+poly_step, poly_step)
        poly_ccf = p(poly_vel)
        mu = poly_vel[np.argmin(poly_ccf)]
        # Sometimes the polynomial fitting can give a point that it is not logical
        # (far away from the detected peak), so we do a validation check
        if mu < xcoord[peaks[i]-1] or mu > xcoord[peaks[i]+1]:
            mu = xcoord[peaks[i]]
            poly_step = xcoord[peaks[i]+1] - xcoord[peaks[i]] # Temporary just to the next iteration

        #########################################################
        ####### Gaussian/Voigt fit to determine other params.
        #########################################################
        # Models to fit
        gaussian_model = GaussianModel()
        voigt_model = VoigtModel()

        # Parameters estimators
        baseline = np.median(fluxes[base_points])
        A = fluxes[peaks[i]] - baseline
        sig = np.abs(xcoord[top[i]] - xcoord[base[i]])/3.0

        parinfo = [{'value':0., 'fixed':False, 'limited':[False, False], 'limits':[0., 0.]} for j in np.arange(5)]
        parinfo[0]['value'] = 1.0 #fluxes[base[i]] # baseline # Continuum
        parinfo[0]['fixed'] = True
        #parinfo[0]['limited'] = [True, True]
        #parinfo[0]['limits'] = [fluxes[peaks[i]], 1.0]
        parinfo[1]['value'] = A # Only negative (absorption lines) and greater than the lowest point + 25%
        parinfo[1]['limited'] = [False, True]
        parinfo[1]['limits'] = [0., 0.]
        parinfo[2]['value'] = sig # Only positives (absorption lines)
        parinfo[2]['limited'] = [True, False]
        parinfo[2]['limits'] = [1.0e-10, 0.]
        parinfo[3]['value'] = mu # Peak only within the xcoord slice
        #parinfo[3]['fixed'] = True
        parinfo[3]['fixed'] = False
        parinfo[3]['limited'] = [True, True]
        #parinfo[3]['limits'] = [xcoord[base[i]], xcoord[top[i]]]
        #parinfo[3]['limits'] = [xcoord[peaks[i]-1], xcoord[peaks[i]+1]]
        parinfo[3]['limits'] = [mu-poly_step, mu+poly_step]

        # Only used by the voigt model (gamma):
        parinfo[4]['value'] = (xcoord[top[i]] - xcoord[base[i]])/2.0 # Only positives (not zero, otherwise its a gaussian) and small (for nm, it should be <= 0.01 aprox but I leave it in relative terms considering the spectrum slice)
        parinfo[4]['fixed'] = False
        parinfo[4]['limited'] = [True, True]
        parinfo[4]['limits'] = [0.0, xcoord[top[i]] - xcoord[base[i]]]

        f = fluxes[base[i]:top[i]+1]
        min_flux = np.min(f)
        # More weight to the deeper fluxes
        if min_flux < 0:
            weights = f + -1*(min_flux) + 0.01 # Above zero
            weights = np.min(weights)/ weights
        else:
            weights = min_flux/ f
        weights -= np.min(weights)
        weights = weights/np.max(weights)


        try:
            # Fit a gaussian and a voigt, but choose the one with the best fit
            if model in ['2nd order polynomial + auto fit', '2nd order polynomial + gaussian fit']:
                gaussian_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo[:4]), weights=weights)
                #gaussian_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo[:4]))
                rms_gaussian = np.sqrt(np.sum(np.power(gaussian_model.residuals(), 2)) / len(gaussian_model.residuals()))
            if model in ['2nd order polynomial + auto fit', '2nd order polynomial + voigt fit']:
                voigt_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo), weights=weights)
                #voigt_model.fitData(xcoord[base[i]:top[i]+1], fluxes[base[i]:top[i]+1], parinfo=copy.deepcopy(parinfo))
                rms_voigt = np.sqrt(np.sum(np.power(voigt_model.residuals(), 2)) / len(voigt_model.residuals()))

            if model == '2nd order polynomial + voigt fit' or (model == '2nd order polynomial + auto fit' and rms_gaussian > rms_voigt):
                final_model = voigt_model
            else:
                final_model = gaussian_model
#
            # Calculate velocity error based on:
            # Zucker 2003, "Cross-correlation and maximum-likelihood analysis: a new approach to combining cross-correlation functions"
            # http://adsabs.harvard.edu/abs/2003MNRAS.342.1291Z
            inverted_fluxes = 1-fluxes
            distance = xcoord[1] - xcoord[0]
            first_derivative = np.gradient(inverted_fluxes, distance)
            second_derivative = np.gradient(first_derivative, distance)
            ## Using the exact velocity, the resulting error are less coherents (i.e. sometimes you can get lower errors when using bigger steps):
            #second_derivative_peak = np.interp(final_model.mu(), xcoord, second_derivative)
            #inverted_fluxes_peak = final_model.mu()
            ## More coherent results:
            peak = xcoord.searchsorted(final_model.mu())
            inverted_fluxes_peak = inverted_fluxes[peak]
            second_derivative_peak = second_derivative[peak]
            if inverted_fluxes_peak == 0:
                inverted_fluxes_peak = 1e-10
            if second_derivative_peak == 0:
                second_derivative_peak = 1e-10
            sharpness = second_derivative_peak/ inverted_fluxes_peak
            line_snr = np.power(inverted_fluxes_peak, 2) / (1 - np.power(inverted_fluxes_peak, 2))
            # Use abs instead of a simple '-1*' because sometime the result is negative and the sqrt cannot be calculated
            error = np.sqrt(np.abs(1 / (nbins * sharpness * line_snr)))

            final_model.set_emu(error)
            models.append(final_model)
        except Exception as e:
            print(type(e), e.message)


    return np.asarray(models)

def select_good_velocity_profile_models(models, ccf):
    """
    Select the modeled peaks that are not deeper than mean flux + 6*standard deviation
    unless it is the only detected peak.
    """
    accept = []
    if len(models) == 0:
        return accept

    xcoord = ccf['x']
    fluxes = ccf['y']

    ## We want to calculate the mean and standard deviation of the velocity profile
    ## but discounting the effect of the deepest detected lines:
    # Build the fluxes for the composite models
    line_fluxes = None
    for model in models:
        if line_fluxes is None:
            # first peak
            line_fluxes = model(xcoord)
            continue

        current_line_fluxes = model(xcoord)
        wfilter = np.where(line_fluxes > current_line_fluxes)[0]
        line_fluxes[wfilter] = current_line_fluxes[wfilter]
    ### Substract the line models conserving the base level
    if line_fluxes is not None:
        values = 1 + fluxes - line_fluxes
    else:
        values = fluxes
    ## Finally, calculate the mean and standard deviation
    check_mean = np.mean(values)
    check_std = np.std(values)
    for (i, model) in enumerate(models):
        # The first peak is always accepted
        if i == 0:
            accept.append(True)
            continue

        mu = model.mu()
        peak = model(mu)

        # Discard peak if it is not deeper than mean flux + 6*standard deviation
        limit = check_mean - 6*check_std
        if limit < 0.05 or peak >= limit:
            accept.append(False)
        else:
            accept.append(True)
    return np.asarray(accept)




############## [end] Radial velocity

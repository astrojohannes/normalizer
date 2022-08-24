#
# The methods in this file were originally written by Sergi Blanco-Cuaresma (http://www.blancocuaresma.com/s/) for iSpec.
# They were modified to be used in the normalizer tool.
#

def air_to_vacuum(spectrum):
    """
    It converts spectrum's wavelengths (nm) from air to vacuum
    """
    # Following the air to vacuum conversion from VALD3 (computed by N. Piskunov) http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    wave_air = spectrum['waveobs'] * 10. # Angstroms
    s_square = np.power(1.e4/ wave_air, 2)
    n2 = 1. + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s_square) + 0.0001599740894897 / (38.92568793293 - s_square)
    wave_vacuum = wave_air*n2 # Angstroms
    converted_spectrum = create_spectrum_structure(wave_vacuum / 10., spectrum['flux'], spectrum['err'])
    return converted_spectrum


def vacuum_to_air(spectrum):
    """
    It converts spectrum's wavelengths from vacuum to air
    """
    # Following the vacuum to air conversion the formula from Donald Morton (2000, ApJ. Suppl., 130, 403) which is also a IAU standard
    # - More info: http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    wave_vacuum = spectrum['waveobs'] * 10. # Angstroms
    s_square = np.power(1.e4/ wave_vacuum, 2)
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s_square) + 0.00015998 / (38.9 - s_square)
    wave_air = wave_vacuum/n # Angstroms
    converted_spectrum = create_spectrum_structure(wave_air / 10., spectrum['flux'], spectrum['err'])

    return converted_spectrum

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets
import sys,os
import normalizer as n
import getopt

os.environ["XDG_SESSION_TYPE"]='x11'    # should enable correct window geometry etc. when Wayland is used
os.environ['LANG'] = u'en_US.UTF-8'     # force locale to English, to assure dots as decimal sepatators
os.environ['NORMALIZER_DIR'] = os.path.dirname(os.path.realpath(__file__))

## Print usage
def usage():
    print("Usage:")
    print(sys.argv[0], "[--help] [--wave=min,max] [spectrum_file]")

## Interpret arguments
def get_arguments():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hw", ["help","wave="])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(1)

    wave_min=0
    wave_max=0
        
    spectrum_file = None
    has_wave=False

    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            exit(0)
            
        elif o in ("-w", "--wave"):
            wave = str(a)
            if len(wave.split(','))==2:
                try:
                    wave_min=float(wave.split(',')[0])
                    wave_max=float(wave.split(',')[1])
                except:
                    print("Wavelength range is not numeric. Exiting.")
                    wave_min=0
                    wave_max=0
                    exit(1)
                                        
                if (wave_max-wave_min) > 1:
                    has_wave=True
                else:
                    print("Wavelength range is too short or not recognized. Exiting.")
                    exit(1)
            else:
                print("Wavelength range is too short or not recognized. Exiting.")
                exit(1)

        else:
            print("Argument", o, "not recognized. Exiting.")
            usage()
            sys.exit(1)

    # Open spectrum
    has_spec=False
    for arg in args:
        spectrum_file = arg
        if not os.path.exists(spectrum_file):
            print("Spectrum file", spectrum_file, "missing. Exiting.")
            sys.exit(1)
        else:
            has_spec=True

    if has_spec and has_wave:
        return wave_min,wave_max,spectrum_file
    elif has_spec and not has_wave:
        return 0,0,spectrum_file
    else:
        return 0,0,None


if __name__ == "__main__":
    wave_min,wave_max,spectrum_file=get_arguments()        
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Breeze')  # ['Breeze', 'Oxygen', 'QtCurve', 'Windows', 'Fusion', 'Cleanlooks']
    norm = n.start(os.environ['NORMALIZER_DIR']+"/dev/gui.ui")
    norm.connect_buttons()
    if not spectrum_file == None and wave_min>0 and wave_max>0:
        norm.readfits(spectrum_file)
        norm.gui.lbl_fname.setText(spectrum_file)
        norm.make_fig(0)
        norm.zoom_fig(wave_min,wave_max)
        norm.on_slice_pressed()
    elif not spectrum_file == None:
        norm.readfits(spectrum_file)
        norm.gui.lbl_fname.setText(spectrum_file)
        norm.make_fig(0)
        #norm.fit_spline()
    sys.exit(app.exec_())


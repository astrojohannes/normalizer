import numpy as np
from scipy.ndimage import gaussian_filter1d

class PeakMask:
    def __init__(self, data, sigma_smooth, sigma_threshold, rms_tolerance):
        self.data = data
        self.sigma_smooth = sigma_smooth
        self.sigma_threshold = sigma_threshold
        self.rms_tolerance = rms_tolerance
        self.mask = None
        self.snr = None
    
    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def create_mask(self):
        # Preprocessing
        ysmooth = 1 + self.smooth(self.data, self.sigma_smooth) * -1
        rms = np.std(ysmooth)
        masked_ysmooth = np.copy(ysmooth)
        mask = np.zeros_like(ysmooth,dtype=int)
 
        i=0
        iterlimit=60
        while True:
            self.snr = np.ma.masked_array(masked_ysmooth,mask) / float(rms)
            mask = self.snr > self.sigma_threshold
            masked_ysmooth = np.ma.masked_array(ysmooth, mask)
            
            new_rms = np.ma.std(masked_ysmooth)
            percent_change = np.ma.abs((new_rms - rms) / rms) * 100
            
            if percent_change < self.rms_tolerance or i >= iterlimit:
                self.mask = mask
                break
            
            rms = new_rms

            i+=1
        return np.array(self.mask,dtype=int)

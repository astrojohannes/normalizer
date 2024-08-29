import numpy as np
import copy
from scipy.ndimage import gaussian_filter1d

class PeakMask:
    def __init__(self, data, sigma_smooth, sigma_threshold, rms_tolerance, maxnumber_iterations):
        self.data = data
        self.sigma_smooth = sigma_smooth
        self.sigma_threshold = sigma_threshold
        self.rms_tolerance = rms_tolerance
        self.maxnumber_iterations = maxnumber_iterations
        self.mask = None
    
    def smooth(self, y, box_pts):
        if box_pts > 1:
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        else:
            return y
   
    def create_mask(self):
        # Preprocessing
        ysmooth = np.nanmax(self.data) + 0.01 - self.smooth(self.data, self.sigma_smooth)
        rms = np.nanstd(ysmooth)
        new_rms = rms
        last_rms = rms
        masked_ysmooth = np.copy(ysmooth)
        mask = np.zeros_like(ysmooth, dtype=bool)

        i = 0
        last_acceptable_mask = None  # to hold the last acceptable (not fully True) mask
        while i < self.maxnumber_iterations and new_rms > 0:
            i += 1
            snr = np.ma.masked_array(masked_ysmooth, mask) / float(rms)
            mask_new = np.array(snr > self.sigma_threshold,dtype=bool)
            updated_mask = mask | mask_new

            # Check if updated_mask is all True
            if not np.all(updated_mask):  
                # If not all True, update the working masks and proceed
                mask = updated_mask
                last_acceptable_mask = np.copy(mask)  # Update last acceptable mask
                last_rms = new_rms  # Update last RMS before changing
                masked_ysmooth = np.ma.masked_array(ysmooth, mask)
                new_rms = np.ma.std(masked_ysmooth)
            else:
                # If all True, break the loop and revert to last acceptable states
                print("All values in mask are True. Stopping here.")
                new_rms = last_rms  # Revert RMS to last acceptable value
                break  # Exit the loop

            if new_rms > 0:
                percent_change = np.ma.abs((new_rms - rms) / rms) * 100
                #print(f"RMS measured outside line masks: {round(new_rms,4)}")
                self.mask = copy.deepcopy(last_acceptable_mask if last_acceptable_mask is not None else mask)
                rms = new_rms
                if percent_change < self.rms_tolerance:
                    break

            else:   # rms = 0
                print("RMS is zero. Stopping here.")
                break
    
        return np.array(self.mask if last_acceptable_mask is not None else mask, dtype=bool), i, last_rms 

#
#
#   DEF FOR FRAC SCORE ID, LABELS FROM DYNAMIC PRESSURE
#
#  Things to consider before upgrading to production:

## 1. OUTPUT AS .PY, PUT IT IN GIT, SOME KIND OF VERSION CONTROL.
## 2. ARE WE OVERFITTING TO CHUPA 04? TRY ON DIFFERENT WELLS

def frac_score(dyn_presure_displacement, sampling_rate=43000, window_in_seconds=1, STD_MULTIPLIER=10):

    # This method takes a dynamic pressure signal and time np array, along with some calibrated parameters, and returns a label array (of equivalent size) for ML.
    #
    #      method inputs:
    #
    #   dyn_pressure_time, obtained from dyamic pressure data, assumed to be a numpy array
    #   dyn_presure_displacement, obtained from dyanmic pressure data, assumed to be a numpy array
    #   sampling_rate, assumed to ~ 43 kHz
    #   window_in_seconds, for calibration purposes, this is 10 seconds, but can be adjusted to 1 second
    #   TIME_WIDTH, 1's padding around detected event, set to 0.25 ms
    #   STD_MULTIPLIER, the detector function calculates the stanard deviation of the dynamic pressure data. the Multipler is a scalar used to determine the detector threshold
    #                   we anticipate that most of the broad spectral pops occur beyond 4 STD vals of the dynamic pressure distribution.
    #
    #
    #      method outputs:
    #
    #   sqr_wave_label_time, a numpy array with equivalent length and size as the dyn_presure_displacement input array
    #   sqr_wave_label_vals, a numpy array with equivalent length and size as the dyn_presure_displacement input array
    #   detectorStatsTxt, a string that reports the detected events in seconds with microsecond precision.

    # 1. Import necessary libraries

    import scipy.signal as signal
    import numpy as np

    # 2. Define a highpass, butterworth filter, used to remove the seismic, low requency response

    sos = signal.butter(10,250,btype='highpass',fs=sampling_rate,output='sos')

    # 3. Filtered_xd is the dynamic pressure response, with f < 250 Hz contributions removed.

    filtered_xd = signal.sosfilt(sos, dyn_presure_displacement)

    # 4. Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response

    std_dynamic_window = np.std(np.absolute(filtered_xd))


    # 6. Distance defines how each peak needs to be from each other, this avoids overcounting, see the scipy.signal docs for more info.

    distanceVal = 1000

    # 7. The height threshold is the accpetable signal to noise ration, or relative magnitude of the filtered signal

    HEIGHT_THRESHOLD = STD_MULTIPLIER*std_dynamic_window

    # 8. Using the signal find peaks method, see the scipy.signal docs for more info.

    peaks, _ = signal.find_peaks(filtered_xd, height=HEIGHT_THRESHOLD, distance=distanceVal)

    return len(peaks)

class FracScore():
    def __init__(self, window_size):
        self.window_size = window_size

    def infer(self, dynamic_data, static_data):
        if len(dynamic_data) > 0:
            num_fracs = frac_score(dynamic_data, window_in_seconds=self.window_size, STD_MULTIPLIER=10, sampling_rate=40000)
            result = {"event": "frac count",
                      "value": num_fracs}
            return result

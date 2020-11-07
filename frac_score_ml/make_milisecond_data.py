import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
import matplotlib.image as img
import tensorflow as tf
import datetime as dt

from ml_utils import frac_score_detector, butter_bandpass_filter
from usefulmethods import mk_data_time_from_str, dt_obj_to_str

from scipy import signal

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

"""
Use this to quickly visualize the event datasets stored as .npy files
9/23 - NOw looking at 1 sec windows, NNs will classify on milisecond timescale 
"""

# # open file 7000 pops, 5.2 GB
# detectedPops = np.load('raw_ml_data/fracScore_allPads_08_09_pop_events.npy', allow_pickle=True)
# # need to adjust, 5788 nonevents, filesize is 4.3 GB
# notPops = np.load('raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy', allow_pickle=True)

# 1 - second windows
#
# # open file 4245 pops, 661.7 MB
# detectedPops = np.load('raw_ml_data/fracScore_allPads_23_09_2020_T01_05_44.npy', allow_pickle=True)
# # need to adjust, 4348 nonevents, filesize is 677.3 MB
# notPops = np.load('raw_ml_data/fracScore_allPads_non_events23_09_2020_T01_06_36.npy', allow_pickle=True)


# 9/24 - New dataset, higher std_m

# std_m = 12
# open file 5618 pops, 875 MB
#detectedPops = np.load('raw_ml_data/fracScore_allPads_11_10_2020_T00_04_07.npy', allow_pickle=True)
# need to adjust, 4343 nonevents, filesize is 676 MB
#notPops = np.load('raw_ml_data/fracScore_allPads_non_events24_09_2020_T13_49_41.npy', allow_pickle=True)

# 11/3 new detected pops

detectedPops = np.load('raw_ml_data/raw_ml_pops_Nov_3.npy', allow_pickle=True)
notPops = np.load('raw_ml_data/fracScore_allPads_non_events24_09_2020_T13_49_41.npy', allow_pickle=True)

save_filename_pops = 'millisecond_data_pops_3_Nov_2020.npy'
save_filename_not_pops = 'millisecond_data_not_pops.npy'

print('number of pops detected: ' + str(len(detectedPops)))
print('number of nonevents detected: ' + str(len(notPops)))

# set axis dims

fig_x = 25
fig_y = 10
F_MIN = 1
# F_MAX = 2000
F_MAX = 5000
# CUTOFF_FREQ = 10
COLORMAP = 'jet_r'
TIME_SAMPLE_WINDOW = .0075  # 0.01 is good for 1 sec
OVERLAP_FACTOR = 50  # 10 is good for 1 sec
sampling_rate = 40000
n_FFT = int(sampling_rate * TIME_SAMPLE_WINDOW)  # 5ms window
n_overlap = int(sampling_rate * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

# the goal is to extract 0.02 seconds of data from, what is assumed, to be a pop, at the 0.5s mark of the data set

millisecond_data_obj = []

print('building 20ms datasets, pops...')

for i in range(len(detectedPops)):
    well_name = detectedPops[i][0]
    api = detectedPops[i][1]
    stage = detectedPops[i][2]
    start = detectedPops[i][3]
    stop = detectedPops[i][4]
    values_array = detectedPops[i][5]

    # show 0.02 portion

    ten_milisecs = int(0.01 * sampling_rate)
    mid_point = int(len(values_array) / 2)

    ten_milisecond_window = values_array[mid_point - ten_milisecs:mid_point + ten_milisecs]

    new_start = mk_data_time_from_str(start) + dt.timedelta(seconds=0.49)
    new_stop = mk_data_time_from_str(start) + dt.timedelta(seconds=0.51)

    millisecond_data_obj.append([well_name,
                                api,
                                stage,
                                dt_obj_to_str(new_start),
                                dt_obj_to_str(new_stop),
                                ten_milisecond_window])

print('reformating of data complete')

np.save(save_filename_pops, millisecond_data_obj)


# now do it for the non events
millisecond_data_not_pops_obj = []

print('building 20ms datasets, not pops...')

for i in range(len(notPops)):
    well_name = notPops[i][0]
    api = notPops[i][1]
    stage = 0
    start = notPops[i][3]
    stop = notPops[i][4]
    values_array = notPops[i][5]

    # show 0.02 portion

    ten_milisecs = int(0.01 * sampling_rate)
    mid_point = int(len(values_array) / 2)

    ten_milisecond_window = values_array[mid_point - ten_milisecs:mid_point + ten_milisecs]

    new_start = mk_data_time_from_str(start) + dt.timedelta(seconds=0.49)
    new_stop = mk_data_time_from_str(start) + dt.timedelta(seconds=0.51)

    millisecond_data_not_pops_obj.append([well_name,
                                api,
                                stage,
                                dt_obj_to_str(new_start),
                                dt_obj_to_str(new_stop),
                                ten_milisecond_window])

print('reformating of non events datadata complete')

np.save(save_filename_not_pops, millisecond_data_not_pops_obj)




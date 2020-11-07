import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
import tensorflow as tf
import scipy.stats as ss
from ml_utils import frac_score_detector, butter_bandpass_filter, make_spectrogram_image_matrix

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

"""
Use this to quickly visualize the event datasets stored as .npy files
9/23 - Looking at 1 sec windows, NNs will classify on millisecond timescale 
9/24 - Looking at 50 20ms windows, applying 1d and 2d convo models  
"""

# 1 - second windows

# # std_m = 10
# # open file 4245 pops, 661.7 MB
# detectedPops = np.load('raw_ml_data/fracScore_allPads_23_09_2020_T01_05_44.npy', allow_pickle=True)
# # need to adjust, 4348 nonevents, filesize is 677.3 MB
# notPops = np.load('raw_ml_data/fracScore_allPads_non_events23_09_2020_T01_06_36.npy', allow_pickle=True)

# 9/24 - New dataset, higher std_m

# std_m = 12
# open file 5618 pops, 875 MB
#detectedPops = np.load('raw_ml_data/fracScore_allPads_24_09_2020_T13_49_09.npy', allow_pickle=True)
# need to adjust, 4343 nonevents, filesize is 676 MB
notPops = np.load('raw_ml_data/fracScore_allPads_non_events24_09_2020_T13_49_41.npy', allow_pickle=True)

print('number of nonevents detected: ' + str(len(notPops)))


# new pops dataset, with gradient update, fracScore_allPads_07_10_2020_T21_40_00

detectedPops = np.load('raw_ml_data/fracScore_allPads_07_10_2020_T21_40_00.npy', allow_pickle=True)
print('number of pops detected: ' + str(len(detectedPops)))

# set axis dims
class_labels = ['pops', 'not_pops']

fig_size_x_in = 6
fig_size_y_in = 3
fig_x = 25
fig_y = 10
f_min = 1
f_max = 5000
COLORMAP = 'jet_r'
TIME_SAMPLE_WINDOW = .0075  # 0.01 is good for 1 sec
OVERLAP_FACTOR = 50  # 10 is good for 1 sec
sampling_rate = 40000
n_FFT = int(sampling_rate * TIME_SAMPLE_WINDOW)  # 5ms window
n_overlap = int(sampling_rate * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))


# view detected pop events

def plot_event(index, data_object, event_type):
    well_name = data_object[index][0]
    api = data_object[index][1]
    start = data_object[index][3]
    stop = data_object[index][4]
    values_array = data_object[index][5]

    out_text = 'pop event: ' + str(index) + '\n'
    out_text += 'WN: ' + well_name + '\n'
    out_text += 'API: ' + api + '\n'
    out_text += 'start: ' + start + '\n'
    out_text += 'stop: ' + stop + '\n'

    if event_type == 'pop':
        stage = data_object[i][2]
        out_text += 'stage: ' + str(stage) + '\n'

    measured_time = len(values_array) / sampling_rate
    time_array = np.linspace(0, measured_time, len(values_array))

    # subplot stuff

    fig, axs = plt.subplots(4, 1, figsize=(fig_x, fig_y))

    # 1st plot, dynamic signal

    axs[0].set_title('Dynamic Pressure Signal: ' + str(event_type), fontsize=18, color='black')
    axs[0].plot(time_array, values_array, color='black', label='dyn, raw', zorder=1)

    # filtered signal for frac score

    filtered_xd = butter_bandpass_filter(values_array, 150, 500, sampling_rate, order=9)

    axs[0].plot(time_array, filtered_xd, color='blue', label='bp filtered', zorder=1)

    axs[0].axis([0, measured_time, np.nanmin(values_array), np.nanmax(values_array)])
    axs[0].legend(loc='upper right')
    axs[0].text(1, 0.6 * min(values_array), out_text, fontsize=11, verticalalignment='center')
    axs[0].set_ylabel('Rel. Mag.')

    # spectrogram. Specgram method

    axs[1].set_title('Spectrogram: ' + str(event_type), fontsize=18, color='black')

    axs[1].specgram(values_array,
                    NFFT=n_FFT,
                    Fs=sampling_rate,
                    noverlap=n_overlap,
                    cmap=pylab.get_cmap(COLORMAP))

    axs[1].set_ylabel('Frequency in Hz')
    axs[1].axis([0, measured_time, f_min, f_max])

    # show 0.02 portion

    ten_millisecs = int(0.01 * sampling_rate)
    mid_point = int(len(values_array) / 2)

    ten_milisecond_window = values_array[mid_point - ten_millisecs:mid_point + ten_millisecs]

    # plot it
    axs[2].set_title('signal: 0.5s +/- 0.01s', fontsize=18, color='black')
    axs[2].plot(time_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
                values_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
                color='black',
                label='dyn,',
                zorder=1)

    # look at pop spectro
    axs[3].set_title('Spectrogram: 0.5s +/- 0.01s', fontsize=18, color='black')

    # remove strong seismic

    values_quick = butter_bandpass_filter(values_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
                                          200,
                                          1000,
                                          sampling_rate,
                                          order=9)
    axs[3].specgram(values_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
                    NFFT=int(n_FFT * 0.05),
                    Fs=sampling_rate,
                    noverlap=int(n_overlap * 0.1),
                    cmap=pylab.get_cmap(COLORMAP))

    axs[3].set_ylabel('Frequency in Hz')
    # axs[3].axis([time_array[mid_point - ten_millisecs], time_array[mid_point + ten_millisecs], f_min, f_max])
    # axs[3].set_ylim(f_min, f_max)

    plt.show()


for z in range(50):
    i = random.randrange(len(detectedPops))

    print('look at: ' + str(i))

    plot_event(i, detectedPops, 'pop')


# now view non events

print('number of non pops detected: ' + str(len(notPops)))

# look at 20 arrays, pick randomly

for r in range(20):
    i = random.randrange(len(notPops))
    print('look at nonevent: ' + str(i))

    plot_event(i, notPops, 'non')

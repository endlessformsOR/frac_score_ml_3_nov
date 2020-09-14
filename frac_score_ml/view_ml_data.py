import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

"""
Use this to quickly visualize the event datasets stored as .npy files
"""

# open file 7000 pops, 5.2 GB

detectedPops = np.load('raw_ml_data/fracScore_allPads_08_09_pop_events.npy', allow_pickle=True)

# need to adjust, 5788 nonevents, filesize is 4.3 GB

notPops = np.load('raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy', allow_pickle=True)

# set axis dims

fig_x = 25
fig_y = 10
F_MIN = 1
F_MAX = 2000
CUTOFF_FREQ = 10
COLORMAP = 'jet_r'
TIME_SAMPLE_WINDOW = .1
OVERLAP_FACTOR = 10
sampling_rate = 40000
n_FFT = int(sampling_rate * TIME_SAMPLE_WINDOW)  # 5ms window
n_overlap = int(sampling_rate * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

# view detected pop events

print('number of pops detected: ' + str(len(detectedPops)))


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

    fig, axs = plt.subplots(2, 1, figsize=(fig_x, fig_y))

    # spectrogram
    axs[0].set_title('Spectrogram ', fontsize=18, color='black')
    axs[0].specgram(values_array, NFFT=n_FFT, Fs=sampling_rate, noverlap=n_overlap, cmap=pylab.get_cmap(COLORMAP))
    axs[0].set_ylabel('Frequency in Hz')
    axs[0].axis([0, measured_time, F_MIN, F_MAX])

    # 1st plot, dynamic signal

    axs[1].set_title('Dynamic Pressure Signal ', fontsize=18, color='black')
    axs[1].plot(time_array, values_array, color='black', label='dyn, raw', zorder=1)
    axs[1].axis([0, measured_time, np.nanmin(values_array), np.nanmax(values_array)])
    axs[1].legend(loc='upper right')
    axs[1].text(1, 0.6 * min(values_array), out_text, fontsize=11, verticalalignment='center')
    axs[1].set_ylabel('Rel. Mag.')

    plt.show()


# look at 20 random events in list

for z in range(20):
    i = random.randrange(len(detectedPops))

    print('look at: ' + str(i))

    plot_event(i, detectedPops, 'pop')

# now view non events

print('number of non pops detected: ' + str(len(notPops)))

# look at 20 arrays, pick randomly

for r in range(20):
    i = random.randrange(len(notPops))
    print('look at nonevent: ' + str(i))

    plot_event(i, detectedPops, 'non')

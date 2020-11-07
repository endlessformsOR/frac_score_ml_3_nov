"""

Use this to view Norman State BASIS pops during pumping events. Include seismic sensor

"""

from dynamic_utils import interval_to_flat_array_resample, parse_time_string_with_colon_offset
from usefulmethods import fetch_sensor_db_data
import os, sys, datetime, subprocess, time, db, statistics, ipywidgets
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import scipy.fftpack as sfp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib.figure import Figure
import datetime as dt
from numpy.fft import fft, fftfreq, ifft
import pylab

from ml_utils import frac_score_detector

#
plt.rcParams.update({'figure.max_open_warning': 0})
# base_folder = '/notebooks/'
# TITLE = 'Conditioned Data Viewer for Event: '
#
# # Set Working Directory
#
# base_path = f'/home/ubuntu/legend-analytics/' + base_folder
# os.chdir(base_path)
# if base_path not in sys.path:
#     sys.path.append(base_path)
# os.chdir(base_path)
# cwd = os.getcwd()
#
# LEGEND_LIVE_DIR = '/home/ubuntu/legend-live'
# DATA_RELATIVE_DIR = '../legend-analytics/' #relative to legend-live-repo

# Look at norman state spectros

# static, dynamic, acc

Norman_State_14_3HA = ['2a61310f-ada8-48b7-ba15-c37b25c85631','1beb7d86-d8db-4639-8113-2896c51031b9','e610e3e2-9eb2-4936-96d8-27f1f7b12ac3']
Norman_State_14_4HA =	['64835a1a-c535-4867-8aae-4f5c56e0fdc9','554763ef-0144-4b47-be2e-1c02b3948742','e8e5d2d3-a9c8-4a80-aae1-7d18c30e7d42']
Norman_State_14_6HA =	['a1137138-780f-4e5f-baf5-a86c576bab45','65c19746-2acf-4817-a43d-490ff8beb763','9ec79ba7-ec6b-478a-b32e-a57941c0f031']

# start stop times, don't go over an hr!

startTime = '2020-08-21T10:54:30-05:00'
stopTime = '2020-08-21T11:04:30-05:00'

# plot dims

fig_x = 25
fig_y = 20
WN = 'Norman State'

# spectro details


sampling_rate = 40000
TIME_SAMPLE_WINDOW = 0.5
OVERLAP_FACTOR = 500
NFFT = int(sampling_rate*TIME_SAMPLE_WINDOW)
COLORMAP = 'jet_r'

# sampling_rate = 40000
# TIME_SAMPLE_WINDOW = 2.5
# OVERLAP_FACTOR = .25
# NFFT = int(sampling_rate*TIME_SAMPLE_WINDOW)
# COLORMAP = 'jet_r'

def makeData(accID, dynID, statID, startSTR, stopSTR):
    # pull dyn pressure data

    start_converted = parse_time_string_with_colon_offset(startSTR)
    stop_converted = parse_time_string_with_colon_offset(stopSTR)
    delta = stop_converted - start_converted
    seconds = int(delta.total_seconds())
    x_d = interval_to_flat_array_resample(dynID, start_converted, stop_converted).values()

    sampling_rate = int(len(x_d) / seconds)

    t_d = np.linspace(0, seconds, len(x_d))

    # make static data

    stat_obj = fetch_sensor_db_data(statID, startSTR, stopSTR)

    x_s = stat_obj['max'].to_numpy()
    t_s = np.linspace(0, len(x_s), len(x_s))

    # make accelerometer data

    acc_d = interval_to_flat_array_resample(accID, start_converted, stop_converted).values()

    return acc_d, x_d, t_d, x_s, t_s, sampling_rate, seconds


# make data

statID = Norman_State_14_6HA[0]
dynID = Norman_State_14_6HA[1]
accID = Norman_State_14_6HA[2]

WNnum = ' 6HA'

acc_d, x_d, t_d, x_s, t_s, sampling_rate, seconds = makeData(accID, dynID, statID, startTime, stopTime)

fig, axs = plt.subplots(4, 1, figsize=(fig_x, fig_y))

NOVERLAP = int(seconds * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

titleTxt = 'Pressure Response and Accelerometer Behavior for Norman State ' + WNnum + ' start: ' + startTime + ' stop: ' + stopTime

axs[0].set_title(titleTxt, fontsize=18)
axs[0].get_yaxis().set_visible(False)
axs[0].get_xaxis().set_visible(True)
# axs[0].set_xlabel('Elapsed time in secs', fontsize = 18, color='black')
axs[0].plot(t_d, x_d, color='gainsboro', label='dynamic', zorder=1)
axs[0].set_xlim([0, seconds])
# axs[0].set_ylim([-100,100])
# axs[0].set_ylim([np.nanmin(x_d),np.nanmax(x_d)])

# fig_pressureResponse_HA.axis([0 ,seconds, x_d_A_FINAL.min(), x_d_A_FINAL.max()])
axs[0].legend(loc='right')

static_plt = axs[0].twinx()

static_plt.plot(t_s, x_s, color='black', label='static', zorder=3)
static_plt.set_xlabel('Elapsed Time In secs', fontsize=18, color='black')
static_plt.set_ylabel('PSI', fontsize=18, color='black')
static_plt.yaxis.set_label_position("left")
static_plt.yaxis.tick_left()
static_plt.legend(loc='upper right')
static_plt.set_xlim([0, seconds])

# 2nd, spectro

axs[1].specgram(x_d, NFFT=NFFT, Fs=sampling_rate, noverlap=NOVERLAP, cmap=pylab.get_cmap(COLORMAP))
# axs[1].specgram(filtered_dyn, NFFT=NFFT, Fs=sampling_rate, noverlap=NOVERLAP, cmap=pylab.get_cmap(COLORMAP))
axs[1].set_title('Dynamic Pressure Spectro')
# axs[1].set_xlabel('Elapsed time in seconds')
axs[1].set_ylabel('Frequency in Hz')
axs[1].axis([0, seconds, 0, 500])

axs[2].set_title('Accelerometer Signal Response')

axs[2].plot(t_d, acc_d, color='green', label='accelerometer', zorder=3)
axs[2].set_xlim([0, seconds])
axs[2].set_ylim([np.nanmin(acc_d), np.nanmax(acc_d)])
axs[2].set_xlabel('Elapsed time in seconds')
axs[2].set_ylabel('Accelerometer Data')

axs[3].specgram(acc_d, NFFT=NFFT, Fs=sampling_rate, noverlap=NOVERLAP, cmap=pylab.get_cmap(COLORMAP))
# axs[1].specgram(filtered_dyn, NFFT=NFFT, Fs=sampling_rate, noverlap=NOVERLAP, cmap=pylab.get_cmap(COLORMAP))
axs[3].set_title('Accelerometer Spectro')
# axs[1].set_xlabel('Elapsed time in seconds')
axs[3].set_ylabel('Frequency in Hz')
axs[3].axis([0, seconds, 0, 250])
axs[3].set_xlabel('Elapsed time in secs', fontsize=18, color='black')

# heuristic frac score for dyn

num_events = 30
std_m = 9.5
distance = int(sampling_rate * 0.25)
fracs_dyn = frac_score_detector(x_d, sampling_rate, 100, 800, std_m, distance)
colors = plt.cm.winter(np.linspace(0, 1, num_events + 1))
cnt = 0

for pk in fracs_dyn:

    # add a line on the plot

    if cnt < num_events:
        cnt += 1

        axs[0].axvline(x=t_d[pk],
                       ls='--',
                       linewidth=2,
                       color=colors[cnt],
                       label='heuristic')

# pool dyn and acc data for all pops detected

output_list = []
millisecond = sampling_rate * 0.001
num_ms = 30

for pk in fracs_dyn:
    dyn_out = x_d[pk - int(num_ms * millisecond):pk + int(num_ms * millisecond)]
    acc_out = acc_d[pk - int(num_ms * millisecond):pk + int(num_ms * millisecond)]

    output_list.append(['Norman State 14-6HA', 'Stage 1', startTime, stopTime, dyn_out, acc_out])

# save file:

np.save('Norman_State_pops_stage_1.npy', output_list)
print('saved data for norman state complete.')

# # use frac score on acc data, use different cutoff frequencies
# std_m_acc = 15
# fracs_acc = frac_score_detector(acc_d, sampling_rate, 100, 200, std_m_acc, distance)
# cnt_acc = 0
# for pk in fracs_acc:

#     # add a line on the plot

#     if cnt_acc < num_events:

#         cnt_acc += 1

#         axs[2].axvline(x=t_d[pk],
#                        ls='--',
#                        linewidth=2,
#                        color=colors[cnt_acc],
#                        label='heuristic')
"""

Use this file to quickly look at pops

"""

import pandas as pd
import numpy as np
import datetime as datetime
import datetime as dt
from ml_utils import print_out_well_job_info, mk_utc_time, \
    fetch_sensor_ids_from_apis, make_data, interval_to_flat_array_resample, \
    mk_data_time_from_str, dt_obj_to_str, frac_score_detector
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
from scipy.signal import find_peaks

# pops = np.load('raw_ml_data/pops_to_observe.npy', allow_pickle=True)
# pops = np.load('raw_ml_data/pops_to_observe_chupa04_stage_1.npy', allow_pickle=True)
pops = np.load('raw_ml_data/Norman_State_pops_stage_1.npy', allow_pickle=True)


# updated frac score

def frac_score_v3(dynamic_pressure_signal, sampling_freq):
    # for a 20ms window, force any non smooth signal to be a basis pop

    dist = 0.02 * sampling_freq
    height = 0.115

    max_dyn = np.nanmax(np.absolute(dynamic_pressure_data))
    dynamic_normalized = np.absolute(dynamic_pressure_data) / np.absolute(max_dyn)
    dynamic_gradient = np.gradient(dynamic_normalized)
    peaks_dynamic_g, _dict = find_peaks(np.absolute(dynamic_gradient), height=height, distance=dist)

    return peaks_dynamic_g


rows = 6
columns = 5

fig, axes1 = plt.subplots(rows, columns, figsize=(25, 20))

i = 0

# plot all raw signals
#
# for j in range(rows):
#
#     for k in range(columns):
#         well_name = pops[i][0]
#         stage = pops[i][1]
#         dynamic_pressure_data = pops[i][4]
#         dynamic_seismic_data = pops[i][5]
#
#         total_time = len(dynamic_pressure_data) / 40000
#
#         time_window = np.linspace(0, total_time * 1000, int(len(dynamic_seismic_data)))
#
#         axes1[j][k].plot(time_window, dynamic_pressure_data, label='dyn', color='red')
#         axes1[j][k].plot(time_window, dynamic_seismic_data, label='acc', color='blue')
#         axes1[j][k].set(xlabel='time in ms',
#                         ylabel='arb. units',
#                         title=well_name + ' , stage: ' + stage + ' , pop: ' + str(i + 1))
#
#         axes1[j][k].set_xlim([20, 30])
# #        axes1[j][k].set_xlim([27.50, 27.70])
#
#         axes1[j][k].legend()
#
#         i += 1
#
# plt.cla()
# plt.show()

# plot normalized vals of signals
# plt.cla()

i = 0

# plot all raw signals

for j in range(rows):

    for k in range(columns):
        well_name = pops[i][0]
        stage = pops[i][1]
        dynamic_pressure_data = pops[i][4]
        dynamic_seismic_data = pops[i][5]

        total_time = len(dynamic_pressure_data) / 40000

        time_window = np.linspace(0, total_time * 1000, int(len(dynamic_seismic_data)))

        # normalize signals

        max_dyn_p = np.nanmax(np.absolute(dynamic_pressure_data))
        max_dyn_s = np.nanmax(np.absolute(dynamic_seismic_data))

        n_dyn = np.absolute(dynamic_pressure_data) / np.absolute(max_dyn_p)
        n_acc = np.absolute(dynamic_seismic_data) / np.absolute(max_dyn_s)

        axes1[j][k].plot(time_window, n_dyn, label='dyn_n', color='red', lw=3)
        axes1[j][k].plot(time_window, n_acc, label='acc_n', color='blue', lw=3)
        axes1[j][k].set(xlabel='time in ms',
                        ylabel='arb. units',
                        title=well_name + ' , stage: ' + stage + ' , pop: ' + str(i + 1))

        # axes1[j][k].set_xlim([27.4, 27.8])
        #        axes1[j][k].set_xlim([27.50, 27.70])
        axes1[j][k].set_xlim([0, 60])

        # add updated frac score here
        sampling_frequency = 40000
        new_frac_peaks = frac_score_v3(dynamic_pressure_data, sampling_frequency)

        for pks in new_frac_peaks:
            # plot peaks
            time_detected = time_window[pks]
            label_text = 'frac v3 pk: ' + str(round(time_detected, 3))
            axes1[j][k].axvline(x=time_detected, linestyle='--', color='black', label=label_text, lw=5)

        axes1[j][k].legend()

        i += 1

plt.show()

# clear plots, plot the np.gradient of all signals

plt.cla()

i = 0
delta_t_list = []
neg_vals_list = []
fig, axes1 = plt.subplots(rows, columns, figsize=(25, 20))

for j in range(rows):

    for k in range(columns):

        well_name = pops[i][0]
        stage = pops[i][1]
        dynamic_pressure_data = pops[i][4]
        dynamic_seismic_data = pops[i][5]

        total_time = len(dynamic_pressure_data) / 40000

        time_window = np.linspace(0, total_time * 1000, int(len(dynamic_seismic_data)))

        # # limit view to 20 to 40 ms window
        # event_smaller_window_start = 27
        # event_smaller_window_stop = 28
        #
        # # approx index of val
        #
        # index_start = int(len(time_window) * (event_smaller_window_start / (total_time * 1000)))
        # index_stop = int(len(time_window) * (event_smaller_window_stop / (total_time * 1000)))
        #
        # # now run peak detect stuff on smaller window
        #
        # peak_det_dyn_pre = dynamic_pressure_data[index_start:index_stop]
        # peak_det_acc = dynamic_seismic_data[index_start:index_stop]
        #
        # short_time = time_window[index_start:index_stop]

        # normalize signals

        max_dyn_p = np.nanmax(np.absolute(dynamic_pressure_data))
        max_dyn_s = np.nanmax(np.absolute(dynamic_seismic_data))

        n_dyn = np.absolute(dynamic_pressure_data) / np.absolute(max_dyn_p)
        n_acc = np.absolute(dynamic_seismic_data) / np.absolute(max_dyn_s)

        # measure delta T for pop between sensors

        dyn_gradient = np.gradient(n_dyn)
        acc_gradient = np.gradient(n_acc)

        # std_m = 10
        # std_m_dyn = 2.65
        # std_m_acc = 2.00

        # std_m_dyn = 4.25
        # std_m_acc = 3.75

        std_m_dyn = 2
        std_m_acc = 2

        distance = 200
        # height = 0.15
        # axes1[j][k].axhline(y=height, color='black', label='cutoff')

        dyn_g_height = std_m_dyn * np.std(np.absolute(n_dyn))
        acc_g_height = std_m_acc * np.std(np.absolute(n_acc))

        peaks_dyn_g, _ = find_peaks(np.absolute(dyn_gradient), height=0.15, distance=distance)
        peaks_acc_g, _1 = find_peaks(np.absolute(acc_gradient), height=0.05, distance=distance)

        print('pop:' + str(i) + ' , number of dyn pks:' + str(len(peaks_dyn_g)) + ' , number of acc peaks: ' + str(
            len(peaks_acc_g)))

        axes1[j][k].plot(time_window, np.absolute(dyn_gradient), label='dyn', color='red', lw=3)
        axes1[j][k].plot(time_window, np.absolute(acc_gradient), label='acc', color='blue', lw=3)
        axes1[j][k].set(xlabel='time in ms',
                        ylabel='d/dt, normalized',
                        title=well_name + ' , stage: ' + stage + ' , pop: ' + str(i + 1))

        # axes1[j][k].axis([27, 28, 0, 1000])
        # axes1[j][k].set_xlim([27.50, 27.70])

        axes1[j][k].axis([27.4, 27.8, 0, 0.5])
        # axes1[j][k].axis([27.50, 27.70, 0, 2500])

        # now calculate difference

        acc_time = -999
        dyn_time = -999
        delta_t = -999

        for pk in peaks_dyn_g:

            print('dyn peak at:' + str(time_window[pk]))

            if dyn_time == -999:
                dyn_time = time_window[pk]

                axes1[j][k].axvline(x=dyn_time, linestyle='-.', color='green', label='dyn pk', lw=3)

        for pk1 in peaks_acc_g:

            print('acc peak at:' + str(time_window[pk1]))

            if acc_time == -999:
                acc_time = time_window[pk1]

                axes1[j][k].axvline(x=acc_time, linestyle='--', color='orange', label='acc pk', lw=3)

        if acc_time != -999 and dyn_time != -999:

            delta_t = np.abs(acc_time - dyn_time)

            # delta_t = acc_time - dyn_time

            # delta_t = acc_time - dyn_time
            # delta_t = dyn_time - acc_time

            if delta_t > 0:

                print('difference is: ' + str(delta_t) + ' ms')
                delta_t_list.append(delta_t)

            else:

                print('detected negative delta t:' + str(delta_t))
                neg_vals_list.append((i, delta_t))

        # output_text = 'dt: ' + str(round(delta_t, 3))
        output_text = 'dt: ' + str(delta_t)

        axes1[j][k].text(27.5, 0.2, output_text)

        axes1[j][k].legend()

        # axes1[j][k].plot(time_window[peaks_dyn_g], dyn_gradient[peaks_dyn_g], "x", color='black')
        # axes1[j][k].plot(time_window[peaks_acc_g], acc_gradient[peaks_acc_g], "x", color='green')

        i += 1

plt.show()

# close and restart, plot distribution

plt.cla()
print('size of diffs:' + str(len(delta_t_list)))
print(delta_t_list)

plt.hist(delta_t_list)
plt.xlabel('time difference in milliseconds')
plt.ylabel('count')
plt.title('delta t  distribution' + ' , number of events: ' + str(len(delta_t_list)))
plt.show()

# close and restart, plot val per iteration

plt.cla()
print('size of diffs:' + str(len(delta_t_list)))
print(delta_t_list)

plt.plot(delta_t_list)
plt.xlabel('pop number')
plt.ylabel('delta t, in milliseconds')
plt.title('calculated delta t  per pop event')
plt.show()

# # plot indies
#
# plt.cla()
#
# for i in range(len(pops)):
#
#     fig, ax = plt.subplots(1,1)
#     well_name = pops[i][0]
#     stage = pops[i][1]
#     dynamic_pressure_data = pops[i][4]
#     dynamic_seismic_data = pops[i][5]
#
#     total_time = len(dynamic_pressure_data) / 40000
#
#     time_window = np.linspace(0,total_time*1000, int(len(dynamic_seismic_data)))
#
#     ax.plot(time_window, dynamic_pressure_data, label='dyn', color='red')
#     ax.plot(time_window, dynamic_seismic_data, label='acc', color='blue')
#
#     ax.set(xlabel='time in ms',
#            ylabel='arb. units',
#            title=well_name + ' , stage: ' + stage + ' , pop: ' + str(i+1))
#
#     ax.legend()
#
#     plt.show()

#
# # plot indies with spectros
#
# sampling_rate = 40000
# TIME_SAMPLE_WINDOW = .06
# OVERLAP_FACTOR = .25
# NFFT = int(int(sampling_rate*TIME_SAMPLE_WINDOW) / 60)
# COLORMAP = 'jet_r'
#
# #plt.cla()
#
# for i in range(18):
# #for i in range(len(pops)):
#
#     fig, axs = plt.subplots(3,1)
#     well_name = pops[i][0]
#     stage = pops[i][1]
#     dynamic_pressure_data = pops[i][4]
#     dynamic_seismic_data = pops[i][5]
#
#     total_time = len(dynamic_pressure_data) / 40000
#
#     NOVERLAP = int(total_time * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))
#
#     time_window = np.linspace(0,total_time*1000, int(len(dynamic_seismic_data)))
#
#     axs[0].plot(time_window, dynamic_pressure_data, label='dyn', color='red')
#     axs[0].plot(time_window, dynamic_seismic_data, label='acc', color='blue')
#     axs[0].set_xlim([20,37])
#     #axs[0].axis([27, 28, -10000, 10000])
#     #NOVERLAP = int(total_time * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))
#
#     axs[0].set(xlabel='time in ms',
#                ylabel='arb. units',
#                title=well_name + ' , stage: ' + stage + ' , pop: ' + str(i+1))
#
#     axs[0].legend()
#
#     axs[1].specgram(dynamic_pressure_data, NFFT=NFFT, Fs=sampling_rate, noverlap=NOVERLAP, cmap=pylab.get_cmap(COLORMAP))
#     axs[1].set_title('Dynamic Pressure Data Spectro')
#     axs[1].set_ylabel('Frequency in Hz')
#     axs[1].set_xlabel('Elapsed time in secs', fontsize=12, color='black')
#     axs[1].axis([0, 0.06, 0, 10000])
#
#     axs[2].specgram(dynamic_seismic_data, NFFT=NFFT, Fs=sampling_rate, noverlap=NOVERLAP, cmap=pylab.get_cmap(COLORMAP))
#     axs[2].set_title('Seismic Accelerometer Data Spectro')
#     axs[2].set_ylabel('Frequency in Hz')
#     axs[2].set_xlabel('Elapsed time in secs', fontsize=12, color='black')
#     axs[2].axis([0, 0.06, 0, 5000])
#
#     plt.show()
#
# print('all doneskis.')

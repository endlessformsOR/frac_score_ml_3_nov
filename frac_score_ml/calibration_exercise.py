"""

Use this file to pull 10 mins of dynamic data from the center of each pumping event in the Chupadera 04 HA well

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


## Some defs that will end up in ml_utils at some point


def find_middle_time(window_m, stamped_time, utc_start):
    half_time = int(stamped_time) / 2

    window_utc_start = mk_data_time_from_str(utc_start) + dt.timedelta(seconds=60 * half_time)
    window_utc_stop = window_utc_start + dt.timedelta(seconds=window_m * 60)

    return dt_obj_to_str(window_utc_start), dt_obj_to_str(window_utc_stop)


def make_all_stage_data(df_well_ids, df_eff_report, window_mins, save_file_name):
    output_array = []

    # now_str = datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")

    print('saving data under the filename: ' + save_file_name)

    total_events = len(df_eff_report)

    for i in range(total_events):

        # print('analyzing record: ' + str(i) + ' of: ' + str(total_events))

        well_name = str(df_eff_report.loc[i, "Well"])
        api = str(df_eff_report.loc[i, "API #"])
        stage = str(df_eff_report.loc[i, "Stage"])
        start_date = str(df_eff_report.loc[i, "Start Date"])
        start_time = str(df_eff_report.loc[i, "Start Time"])
        end_date = str(df_eff_report.loc[i, "End Date"])
        end_time = str(df_eff_report.loc[i, "End Time"])
        reason = str(df_eff_report.loc[i, "Reason"])
        comments = str(df_eff_report.loc[i, "Comments"])
        stamped_time = str(df_eff_report.loc[i, "TimeInMin"])

        # find chupa 04 HA and

        if api == API or well_name == Well_Name:

            if reason == 'Pumping':
                # collect data

                print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage,
                                        stamped_time, comments)

                utc_start = mk_utc_time(start_date, start_time)

                # make sure time is greater than 10 minutes

                if float(stamped_time) > 10:

                    # now pull IDs

                    static_id, dynamic_id = fetch_sensor_ids_from_apis(df_well_ids, api)

                    if dynamic_id != '-1' and static_id != '-1':

                        # find center of event

                        time_start, time_stop = find_middle_time(window_mins, float(stamped_time), utc_start)

                        # now make data

                        x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dynamic_id,
                                                                               static_id,
                                                                               time_start,
                                                                               time_stop)

                        if len(x_d) > 0:

                            # write to disk

                            output_array.append([well_name,
                                                 api,
                                                 stage,
                                                 time_start,
                                                 time_stop,
                                                 x_d])
                        else:

                            print('unable to evaluate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

    # print('percent complete: ' + str(round(100 * (i / total_events), 2)))

    # save the data while you can!

    print('frac data build complete')

    np.save(save_file_name, output_array)

    return output_array


# # UNCOMMENT TO BUILD ALL STAGES FOR WELL

# Well_Name = 'Briscoe Catarina West 33 HU'
# API = '42127379820000'
# save_file_name = Well_Name + '_frac_dataset' + '.npy'
# window_minutes = 10
# df_well_ids = pd.read_csv('wells3Sept2020.csv')
# df_eff_report = pd.read_csv('fracPops_effReport_allPads_4Sept2020.csv')
#
# # make the data
# fracs_obj = make_all_stage_data(df_well_ids, df_eff_report, window_minutes, save_file_name)
#
# # did we get it?
#
# print('length of pops array:', len(fracs_obj))

### LOAD AND VIEW DATA

# load data

#fracs_obj = np.load('raw_ml_data/Norman State 14-6HA_frac_dataset_v1.npy', allow_pickle=True)
fracs_obj = np.load('raw_ml_data/Chupadera_04_HA_frac_all_stages_dataset.npy', allow_pickle=True)
#fracs_obj = np.load('raw_ml_data/Briscoe Catarina West 33 HU_frac_dataset.npy', allow_pickle=True)


print('frac obj loaded, size of obj: ' , len(fracs_obj))


# view fracs from Norman State

i = 0

well_name = fracs_obj[i][0]
stage = fracs_obj[i][2]
start_time = fracs_obj[i][3]
stop_time = fracs_obj[i][4]

fig_x = 25
fig_y = 10
f_min = 1
f_max = 500
COLORMAP = 'jet_r'
TIME_SAMPLE_WINDOW = 0.5  # 0.01 is good for 1 sec
OVERLAP_FACTOR = 500  # 10 is good for 1 sec
sampling_rate = 40000

n_FFT = int(sampling_rate * TIME_SAMPLE_WINDOW)  # 5ms window
n_overlap = int(sampling_rate * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

data_array = fracs_obj[i][5]

measured_time = len(data_array) / sampling_rate
time_array = np.linspace(0, measured_time, len(data_array))

# heuristic frac score

std_m = 10
distance = int(sampling_rate * 0.25)
fracs = frac_score_detector(data_array, sampling_rate, 100, 800, std_m, distance)

fig, axs = plt.subplots(3, 1, figsize=(fig_x, fig_y))

# 1st plot, dynamic signal

axs[0].set_title(well_name + ', ' + 'stage: ' + stage +  ' start: ' + start_time + ' , stop:' + stop_time + ' , std: ' + str(std_m) + ' index in data obj: ' + str(i),
                 fontsize=18, color='black')
axs[0].plot(time_array, data_array, color='black', label='dyn, raw', zorder=1)

# filtered signal for frac score

# filtered_xd = butter_bandpass_filter(data_array, 150, 500, sampling_rate, order=9)
# axs[0].plot(time_array, filtered_xd, color='blue', label='bp filtered', zorder=1)

axs[0].axis([0, measured_time, np.nanmin(data_array), np.nanmax(data_array)])
axs[0].legend(loc='upper right')
# axs[0].text(1, 0.6 * min(data_array), out_text, fontsize=11, verticalalignment='center')
axs[0].set_ylabel('Rel. Mag.')
axs[0].set_xlabel('Window sampling size in minutes: ' + str(round(measured_time/60,0)) + '. Elapsed time in seconds')

# add frac score picks

millisecond = sampling_rate * 0.001
num_ms = 30
cnt = 0
num_events = 10
colors = plt.cm.winter(np.linspace(0, 1, num_events+1))

millisecond_time_array = np.linspace(0, 2*num_ms, int(millisecond*2*num_ms))

print('largest val:' + str(2*num_ms))
print('number of elements:' + str(millisecond*2*num_ms))

collect_pops = []

for pk in fracs:
    axs[0].axvline(x=time_array[pk],
                   ls='--',
                   linewidth=2,
                   color=colors[cnt],
                   label='heuristic')

    # plot first 10 events, centered around a window

    if cnt < num_events:
        # plot
        event = data_array[pk - int(num_ms * millisecond):pk + int(num_ms * millisecond)]
        #normalize
        rawMax = np.nanmax(np.absolute(event))
        normalizedRaw = [np.absolute(x / rawMax) for x in event]

        # slightly increase vals to stack on top

        adjustedVals = np.asarray(normalizedRaw) + int(cnt*1.1)
        axs[2].plot(millisecond_time_array, adjustedVals, label='event: ' + str(cnt + 1), color=colors[cnt])
        cnt += 1
        collect_pops.append([well_name, stage, start_time, stop_time, millisecond_time_array, adjustedVals])

# spectrogram. Specgram method

axs[1].set_title('Spectrogram: ' + 'number of peaks: ' + str(len(fracs)), fontsize=18, color='black')

axs[1].specgram(data_array,
                NFFT=n_FFT,
                Fs=sampling_rate,
                noverlap=n_overlap,
                cmap=pylab.get_cmap(COLORMAP))

axs[1].set_ylabel('Frequency in Hz')
axs[1].axis([0, measured_time, f_min, f_max])
axs[1].set_xlabel('Elapsed time in seconds')

# individual pops window

axs[2].set_title('pops for 1st ' + str(cnt) + ' peaks, ' + 'total window size: ' + str(2 * num_ms) + 'ms',
                 fontsize=18, color='black')
axs[2].axis([0, 2*num_ms, 0, num_events])
axs[2].set_xlabel('Elapsed time in milliseconds')
axs[2].set_ylabel('Detected signal, normalized')

np.save('pops_to_observe.npy', collect_pops)

plt.show()



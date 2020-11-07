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

#pops = np.load('raw_ml_data/pops_to_observe.npy', allow_pickle=True)
#pops = np.load('raw_ml_data/pops_to_observe_chupa04_stage_1.npy', allow_pickle=True)
pops = np.load('raw_ml_data/Norman_State_pops_stage_1.npy', allow_pickle=True)

rows = 6
columns = 5

# plot the np.gradient of all signals

i = 0
delta_t_list = []
fig, axes1 = plt.subplots(rows,columns,figsize=(25, 20))

for j in range(rows):

    for k in range(columns):

        well_name = pops[i][0]
        stage = pops[i][1]
        dynamic_pressure_data = pops[i][4]
        dynamic_seismic_data = pops[i][5]

        total_time = len(dynamic_pressure_data) / 40000

        time_window = np.linspace(0, total_time * 1000, int(len(dynamic_seismic_data)))

        # limit view to 20 to 40 ms window
        event_smaller_window_start = 25
        event_smaller_window_stop = 30

        # approx index of val

        index_start = int(len(time_window)*(event_smaller_window_start/(total_time*1000)))
        index_stop = int(len(time_window)*(event_smaller_window_stop/(total_time*1000)))

        # now run peak detect stuff on smaller window

        peak_det_dyn_pre = dynamic_pressure_data[index_start:index_stop]
        peak_det_acc = dynamic_seismic_data[index_start:index_stop]

        short_time = time_window[index_start:index_stop]

        # measure delta T for pop between sensors

        dyn_gradient = np.gradient(dynamic_pressure_data)
        acc_gradient = np.gradient(dynamic_seismic_data)

        std_m_dyn = 2.65
        std_m_acc = 2.00
        distance = 200

        dyn_g_height = std_m_dyn*np.std(np.absolute(peak_det_dyn_pre))
        acc_g_height = std_m_acc*np.std(np.absolute(peak_det_acc))

        peaks_dyn_g, _ = find_peaks(peak_det_dyn_pre, height=dyn_g_height, distance=distance)
        peaks_acc_g, _1 = find_peaks(peak_det_acc, height=acc_g_height, distance=distance)

        print('pop:' + str(i) + ' , number of dyn pks:' + str(len(peaks_dyn_g)) + ' , number of acc peaks: ' + str(len(peaks_acc_g)))

        axes1[j][k].plot(time_window, dyn_gradient, label='dyn', color='red')
        axes1[j][k].plot(time_window, acc_gradient, label='acc', color='blue')
        axes1[j][k].set(xlabel='time in ms',
                        ylabel='arb. units',
                        title=well_name + ' , stage: ' + stage + ' , pop: ' + str(i + 1))

        # now calculate difference

        acc_time = -999
        dyn_time = -999

        for pk in peaks_dyn_g:

            print('dyn peak at:' + str(short_time[pk]))

            if dyn_time == -999:
                dyn_time = short_time[pk]

        for pk1 in peaks_acc_g:

            print('acc peak at:' + str(short_time[pk1]))

            if acc_time == -999:
                acc_time = short_time[pk1]

        if acc_time != -999 and dyn_time != -999:

            delta_t = np.abs(acc_time - dyn_time)

            print('difference is: ' + str(delta_t) + ' ms')

            delta_t_list.append(delta_t)

        output_text = 'test me'

        axes1[j][k].text(50, 0, output_text)

        axes1[j][k].legend()

        #axes1[j][k].plot(time_window[peaks_dyn_g], dyn_gradient[peaks_dyn_g], "x", color='black')
        #axes1[j][k].plot(time_window[peaks_acc_g], acc_gradient[peaks_acc_g], "x", color='green')

        i += 1

plt.show()
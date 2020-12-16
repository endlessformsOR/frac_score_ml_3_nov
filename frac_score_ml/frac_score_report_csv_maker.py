import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time

import os
from os import listdir
from os.path import isfile, join


def make_numpy_dataobj(static_id, dynamic_id, WN, event, api, start, stop, sampling_rate, output_file_name):
    output_data = []

    x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dynamic_id, static_id, start, stop)

    all_dynamic_data = np.asarray(x_d, dtype=np.float32)
    all_static_data = np.asarray(x_s, dtype=np.float32)

    #     all_dynamic_data = np.array([], dtype=np.float32)
    #     all_static_data = np.array([], dtype=np.float32)

    result = np.array([WN,
                       api,
                       event,
                       start,
                       sampling_rate,
                       all_dynamic_data.astype(np.float32),
                       all_static_data.astype(np.float32)])

    # output_data.append([WN, event, api, start, stop, x_d, t_d, x_s, t_s, sampling_rate, seconds])

    np.save(output_file_name, result)

    print('created:' + output_file_name + ' successfully.')


def round_preds(preds, threshold=0.5):
    preds = preds.reshape(-1)

    if threshold == 0.5:
        return np.round(preds).astype(np.int)
    else:
        rounded = np.zeros(len(preds), dtype=np.int)
        for i in range(len(rounded)):
            if preds[i] >= threshold:
                rounded[i] = 1
            else:
                rounded[i] = 0

        return rounded


def calc_fracs_per_second(frac_score):
    frac_score = frac_score.reshape(-1)
    preds_per_sec = 50
    num_chunks = len(frac_score) // preds_per_sec

    fps = []
    for i in range(num_chunks):
        start_idx = i * preds_per_sec
        end_idx = start_idx + preds_per_sec
        chunk = frac_score[start_idx: end_idx]
        fps.append(sum(chunk))

    return np.array(fps, dtype=np.int)


def plot_frac_score(df, save_report_file_name, dynamic_data=None):
    fps = df['frac_per_sec']
    fpm = df['frac_per_sec'].groupby(df.index // 60).sum()
    cum_fracs = fps.cumsum()
    static = df['static_pressure']

    total_fracs = np.sum(fps.values)
    max_fpm = np.max(fpm.values)
    avg_fpm = np.mean(fps.values)

    fig = plt.figure(figsize=(22, 14))
    grid = fig.add_gridspec(2, 3)
    fig_text = fig.add_subplot(grid[0, 0])
    fig_fpm = fig.add_subplot(grid[0, 1:])
    fig_cumfrac = fig.add_subplot(grid[1, :])
    # fig_spectro = fig.add_subplot(grid[2, :])

    text_str = f"""
    Total Fracs: {total_fracs}
    Max Fracs/min: {max_fpm}
    Average Fracs/min: {avg_fpm}
    """
    fig_text.axis('off')
    fig_text.text(0, 0.5, text_str, fontsize='large')

    fig_fpm.plot(static, color='black')
    fig_fpm.set_ylabel('Static Pressure (psi)', color='black')
    fig_fpm.set_xlabel('Elapsed Seconds')
    fig_fpm.set_title('Frac Score Per Min')
    fig_fpm_d = fig_fpm.twinx()
    fig_fpm_d.plot(range(0, len(fps), 60), fpm, color='red')
    fig_fpm_d.set_ylabel('Frac Score per min', color='black')

    fig_cumfrac.plot(static, color='black')
    fig_cumfrac.set_ylabel('Static Pressure (psi)', color='black')
    fig_cumfrac.set_xlabel('Elapsed Seconds')
    fig_cumfrac.set_title('Cumulative Frac Score')
    fig_cumfrac_d = fig_cumfrac.twinx()
    fig_cumfrac_d.plot(cum_fracs, color='red')
    fig_cumfrac_d.set_ylabel('Total Frac Score', color='black')

    #     if len(dynamic_data) > 0:
    #         f_max = 500
    #         sampling_rate = 40000
    #         num_seconds = len(dynamic_data) // sampling_rate
    #         TIME_SAMPLE_WINDOW = 2.5
    #         OVERLAP_FACTOR = .25

    #         NFFT = int(sampling_rate*TIME_SAMPLE_WINDOW)
    #         COLORMAP = 'gist_ncar'
    #         NOVERLAP = int(num_seconds*(TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

    #         fig_spectro.specgram(dynamic_data, NFFT=NFFT, Fs=sampling_rate,
    #                              noverlap=NOVERLAP, cmap=COLORMAP)
    #         fig_spectro.set_title('0 < f < 500 Hz')
    #         fig_spectro.set_xlabel('Elapsed time in seconds')
    #         fig_spectro.set_ylabel('Frequency in Hz')
    #         fig_spectro.axis([0, num_seconds, 0, f_max])

    plt.tight_layout()
    plt.savefig(save_report_file_name + '.pdf')
    plt.show()

    return fps, fpm, cum_fracs, static, total_fracs, max_fpm, avg_fpm


def calculate_and_plot_frac_score(stage_data, model):
    print('making data frame ')

    well_name = stage_data[0]
    api = stage_data[1]
    event = stage_data[2]
    start = stage_data[3]
    num_seconds = stage_data[4]
    dynamic_pressure = stage_data[5]
    static_pressure = stage_data[6]

    print('making model prediction')

    t0 = time.time()
    preds = model.predict(dynamic_pressure.reshape(-1, 800, 1))
    print("Inference took:", time.time() - t0, "seconds")

    frac_score = round_preds(preds)
    fps = calc_fracs_per_second(frac_score)

    data = {'frac_per_sec': fps,
            'static_pressure': static_pressure[0:len(fps)]}

    df = pd.DataFrame(data=data)
    save_report_file_name = well_name + '_' + start
    fps, fpm, cum_fracs, static, total_fracs, max_fpm, avg_fpm = plot_frac_score(df, save_report_file_name,
                                                                                 dynamic_pressure)

    # output lists

    fps_list = fps.tolist()
    fpm_list = fpm.tolist()
    cum_fracs_list = cum_fracs.tolist()
    static_list = static.tolist()

    return fps_list, fpm_list, cum_fracs_list, static_list, total_fracs, max_fpm, avg_fpm


import os
base_folder = '/notebooks/'

base_path = f'/home/ubuntu/legend-analytics/' + base_folder
os.chdir(base_path)
if base_path not in sys.path:
    sys.path.append(base_path)
os.chdir(base_path)
cwd = os.getcwd()
print(cwd)

from ml_utils import make_data

# make fpm, cum fracs, static profile for each stage per well


sampling_rate = 40000

data_path = 'Robertson_pad/'
report_file_name = 'stage_1_raw_data'
output_data_file_name = data_path + report_file_name + '.npy'
event = '1'
api = 'n/a'
start = '2020-12-15T03:35:00-05:00'
stop = '2020-12-15T05:45:00-05:00'

# we are using dan's model: 1d_cnn_dans_dec_12.h5
model_folder = 'model_evaluation/models_1d_cnn_evaluation/'
model_path = '1d_cnn_dans_dec_12.h5'
mdl = tf.keras.models.load_model(model_folder + model_path)

# pad ID info

robertson_wellpad_data_IDs = []

robertson_wellpad_data_IDs.append(
    ['Robertson 0304 1H-33X', 'e16f9912-b32d-434a-a4fe-790e60a29343', 'dc14ac08-2731-4912-a308-82c72aa24441'])
robertson_wellpad_data_IDs.append(
    ['Robertson 0304 2H-33X', 'f29e147e-c127-4e50-b5be-1bb136398032', 'e1ada825-e79b-41c1-b1b3-058304927c81'])
robertson_wellpad_data_IDs.append(
    ['Robertson 0304 3H-33X', '8438fb82-2697-43d3-84f3-0ecf0f7aa025', '09a85caa-d523-4ea7-9260-eaf51a47885f'])

# run on 1H, first stage

WN = robertson_wellpad_data_IDs[0][0]
static_id = robertson_wellpad_data_IDs[0][1]
dynamic_id = robertson_wellpad_data_IDs[0][2]

make_numpy_dataobj(static_id, dynamic_id, WN, event, api, start, stop, sampling_rate, output_data_file_name)

ds = np.load(output_data_file_name, allow_pickle=True)

print('building report for: ' + model_path + ', for dataset: ' + output_data_file_name)

fps, fpm, cum_fracs, static, total_fracs, max_fpm, avg_fpm = calculate_and_plot_frac_score(ds, mdl)

print('ready to csv it out', total_fracs)

print('len of static', len(static))
print('len of fps', len(fps))

# invoke writer

import csv

file_name = 'Robertson_pad/robertson_1H_frac_score_stage_1.csv'

#
# stuff to output: fps, fpm, cum_fracs, static, total_fracs, max_fpm, avg_fpm
#
data_path = 'Robertson_pad/'
output_file_name = data_path + 'frac_score_data.npy'
event = '1'
api = 'n/a'
start = '2020-12-15T05:45:00-05:00'
stop = '2020-12-15T07:09:00-05:00'

WN = robertson_wellpad_data_IDs[0][0]

with open(file_name, mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')

    # write header out first

    writer.writerow(['WN:' + ',' + WN])
    writer.writerow(['STAGE:' + ',' + event])
    writer.writerow(['START:' + ',' + start])
    writer.writerow(['STOP:' + ',' + stop])
    writer.writerow(['TOTAL_FRACS:' + ',' + str(total_fracs)])
    writer.writerow(['MAX_FPM:' + ',' + str(max_fpm)])
    writer.writerow(['AVG_FPM:' + ',' + str(avg_fpm)])

    writer.writerow([' '])
    writer.writerow([' '])

    # now Write out data header

    writer.writerow(['STAT_PRES_PSI' + ',' + 'FRACS_PER_SEC' + ',' + 'CUM_FRACS'])

    for i in range(len(static) - 1):
        fps_val = str(round(fps[i], 2))
        cum_fracs_val = str(round(cum_fracs[i], 2))
        static_val = str(round(static[i], 2))

        output_string = static_val + ',' + fps_val + ',' + cum_fracs_val

        writer.writerow([output_string])

    # now write out fracs per min stuff

    writer.writerow([' '])
    writer.writerow([' '])

    writer.writerow(['TIME_MINS' + ',' + 'FRACS_PER_MIN'])

    for i in range(len(fpm) - 1):
        elapsed_min = str(i + 1)
        fps_val = str(round(fpm[i], 2))

        output_string = elapsed_min + ',' + fps_val

        writer.writerow([output_string])

print('file made, look here:' + file_name)
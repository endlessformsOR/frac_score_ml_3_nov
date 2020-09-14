import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from datetime import datetime
from dynamic_utils import interval_to_flat_array_resample, parse_time_string_with_colon_offset
rom usefulmethods import fetch_sensor_db_data, parse_time_string_with_colon_offset, \
    mk_data_time_from_str, dt_obj_to_strimport datetime
import random
import scipy.signal as signal
from scipy.signal import butter, sosfilt, lfilter
import datetime as dt

"""
Use this file to build ml training and testing datasets from all pads and wells with efficiency reports. 
"""


# build training and testing data with this function

def load_ml_data_pops(pops_filename, non_pops_filename, window_size_secs, train_test_split_percent):
    # load npy file WN,Api,stage,dtObj2str(window_start),dtObj2str(window_stop),x_d_win

    print('loading data, this may take some time...')

    input_pop_event = np.load(pops_filename, allow_pickle=True)
    input_not_pops = np.load(non_pops_filename, allow_pickle=True)

    # force size of array, looks like some array sizes are less than expected

    forced_size_of_array = 40000 * window_size_secs

    # setup output lists

    training_labels = []
    testing_labels = []
    training_values = []
    testing_values = []

    print('data loaded, now building training and testing datasets')

    # cycle through events, use ratio to split between training and testing

    ratio = int(100 / (100 - train_test_split_percent))

    for i in range(len(input_pop_event)):

        array_in_question = input_pop_event[i][5]

        if len(array_in_question) == forced_size_of_array:

            if i % ratio == 0:

                training_labels.append([1])
                training_values.append(array_in_question)

            else:

                testing_labels.append([1])
                testing_values.append(array_in_question)

    for j in range(len(input_not_pops)):

        array_in_question = input_not_pops[j][5]

        if len(array_in_question) == forced_size_of_array:

            # reshape

            if j % ratio == 0:

                training_labels.append([0])
                training_values.append(array_in_question)

            else:

                testing_labels.append([0])
                testing_values.append(array_in_question)

    # convert output arrays to numpy arrays with appropriate dimensions

    print('shape of training_values: ' + str(np.shape(training_values)))

    training_arrays = np.expand_dims(np.asarray(training_values), axis=2)
    testing_arrays = np.expand_dims(np.asarray(testing_values), axis=2)
    training_labels = np.asarray(training_labels)
    testing_labels = np.asarray(testing_labels)

    print('testing and training data building complete.' + '\n')

    return training_arrays, testing_arrays, training_labels, testing_labels


def dynamic_model(n_outputs, window_size):
    # static model
    m = Sequential()
    m.add(layers.InputLayer(input_shape=(window_size, 1)))
    m.add(layers.BatchNormalization(name='batch_norm_1'))
    # new, added padding='same' instead of default
    # m.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='conv_1', padding='same'))
    m.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='conv_1'))
    m.add(layers.BatchNormalization(name='batch_norm_2'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_1'))
    m.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='conv_2'))
    m.add(layers.BatchNormalization(name='batch_norm_3'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_2'))
    m.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='conv_3'))
    m.add(layers.BatchNormalization(name='batch_norm_4'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_3'))
    m.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', name='conv_4'))
    m.add(layers.BatchNormalization(name='batch_norm_5'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_4'))
    m.add(layers.Flatten(name='flatten'))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(128, activation='relu', name='dense_1'))
    m.add(layers.Dense(n_outputs, activation='softmax', name='output'))

    return m


def evaluate_model(input_model, values_train, labels_train, values_test, labels_test):
    input_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'],
                        )
    # fit network
    input_model.fit(x=values_train,
                    y=labels_train,
                    validation_data=(values_test, labels_test),
                    # callbacks=[monitor],
                    epochs=20,
                    verbose=2)

    # evaluate model
    loss_values, accuracy = input_model.evaluate(x=values_test,
                                                 y=labels_test,
                                                 verbose=2)
    return accuracy


def make_frac_score_data_non_events(df_well_ids, df_eff_report, num_windows_per_stage, counting_window_secs,
                                    time_before_pop_secs, time_after_pop_secs):
    now_str = datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    save_file_name = 'fracScore_allPads_non_events' + now_str + '.npy'

    print('saving data under the filename: ' + save_file_name)

    non_events_obj = []

    total_events = len(df_eff_report)

    for i in range(total_events):

        print('analyzing record: ' + str(i) + ' of: ' + str(total_events) + '\n')

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

        if reason != 'Pumping':

            # Gather Well info for output text

            # print out well and job info

            print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage,
                                    stamped_time, comments)

            utc_start = mk_utc_time(start_date, start_time)
            utc_end = mk_utc_time(end_date, end_time)

            # make sure time is greater than 10 minutes

            if float(stamped_time) > 10:

                # now pull IDs

                static_id, dynamic_id = fetch_sensor_ids_from_apis(df_well_ids, api)

                if dynamic_id != '-1' and static_id != '-1':

                    # we wish to draw X number of samples, or num_windows_per_stage

                    for j in range(num_windows_per_stage):

                        print(' j - ', j)
                        print('num windows per stage: ', num_windows_per_stage)
                        print('counting windows secs: ', counting_window_secs)
                        print('float of stamped time : ' + stamped_time)
                        print('utc start: ' + utc_start)
                        print('utc end: ' + utc_end)

                        rdm_window_utc_start, rdm_window_utc_stop = mk_random_windows(counting_window_secs,
                                                                                    int(float(stamped_time)),
                                                                                    utc_start)

                        # now make data

                        x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dynamic_id,
                                                                               static_id,
                                                                               rdm_window_utc_start,
                                                                               rdm_window_utc_stop)

                        if len(x_d) > 0:

                            # just pull a window

                            activation_time = mkDataTimeFromStr(rdm_window_utc_start)
                            window_start = activation_time - dt.timedelta(seconds=time_before_pop_secs)
                            window_stop = activation_time + dt.timedelta(seconds=time_after_pop_secs)
                            # pull new data

                            x_d_win = interval_to_flat_array_resample(dynamic_id, window_start, window_stop).values()

                            # now add all the good stuff

                            non_events_obj.append(
                                [well_name, api, stage, dtObj2str(window_start), dtObj2str(window_stop), x_d_win])

                        else:

                            print('unable to evaluate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100 * (i / total_events), 2)) + '\n\n')

    # save the data while you can!

    print('frac data build complete')

    np.save(save_file_name, non_events_obj)


def make_data(dyn_id, stat_id, start_string, stop_string):
    # pull dyn pressure data. Does not include accelerometer data

    start_converted = parse_time_string_with_colon_offset(start_string)
    stop_converted = parse_time_string_with_colon_offset(stop_string)
    delta = stop_converted - start_converted
    seconds = int(delta.total_seconds())
    x_d = interval_to_flat_array_resample(dyn_id, start_converted, stop_converted).values()
    # x_d = interval_to_flat_array_resample(dynID, start_converted, stop_converted)

    sampling_rate = int(len(x_d) / seconds)

    t_d = np.linspace(0, seconds, len(x_d))

    # make static data

    stat_obj = fetch_sensor_db_data(stat_id, start_string, stop_string)

    x_s = stat_obj['max'].to_numpy()
    t_s = np.linspace(0, len(x_s), len(x_s))

    return x_d, t_d, x_s, t_s, sampling_rate, seconds


def frac_score_detector(dynamic_window, sampling_rate, f_low, f_high, std_m, distance_val):
    # Frac score method, updated version with bandpass filter

    filtered_xd = butter_bandpass_filter(dynamic_window, f_low, f_high, sampling_rate, order=9)

    # Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response

    std_dynamic_window = np.std(np.absolute(filtered_xd))

    # see the scipy.signal docs for more info.
    # STD_MULTIPLIER = 14.5
    # distance = 1000

    # 7. The height threshold is the acceptable signal to noise ration, or relative magnitude of the filtered signal

    height_threshold = std_m * std_dynamic_window

    # 8. Using the signal find peaks method, see the scipy.signal docs for more info.

    peaks, _ = signal.find_peaks(np.absolute(filtered_xd), height=height_threshold, distance=distance_val)

    return peaks


def fetch_sensor_ids_from_apis(df_well_ids, api_to_match):
    stat_id = ''
    dyn_id = ''

    for i in range(len(df_well_ids)):

        api_raw = str(df_well_ids.loc[i, "api"])
        api_mod = api_raw.replace('-', '')
        api_mod = api_mod[:-4]
        api_to_match = api_to_match.replace('-', '')

        if api_to_match == api_mod:
            stat_id = str(df_well_ids.loc[i, "static_id"])
            dyn_id = str(df_well_ids.loc[i, "dynamic_id"])

    return stat_id, dyn_id


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    sos = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def butter_low_pass_filter(data, cutoff, fs, order=5):
    b, a = butter_low_pass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_low_pass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def mk_utc_time(str_date, str_time):
    # combine strings:

    date = str_date.split('/')
    time = str_time.split(':')

    mth = str(date[0])
    day = str(date[1])
    yr = str(date[2])
    hr = str(time[0])
    mn = str(time[1])
    # sec = '00.000000Z'
    # sec = '00.000000000Z'
    sec = '00'
    fixed_utc = '05:00'

    if len(date[0]) != 2:
        mth = '0' + mth

    if len(date[1]) != 2:
        day = '0' + day

    if len(time[0]) != 2:
        hr = '0' + hr

    if len(time[1]) != 2:
        mn = '0' + mn

    out_str = '20' + yr + '-' + mth + '-' + day + 'T' + hr + ':' + mn + ':' + sec + '-' + fixed_utc

    # '2020-07-29T14:45:00-05:00'
    # print('new utc time: ' + out_str)

    return out_str


def mk_random_windows(window_size, stamped_time, utc_start):
    new_start = int(random.uniform(0, int(stamped_time)))

    # print('new start: ', new_start)

    rdm_window_utc_start = mkDataTimeFromStr(utc_start) + dt.timedelta(seconds=60 * new_start)
    rdm_window_utc_stop = rdm_window_utc_start + dt.timedelta(seconds=window_size)

    return dtObj2str(rdm_window_utc_start), dtObj2str(rdm_window_utc_stop)


def print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage, stamped_time,
                            comments):
    out_text = 'Well Name: ' + well_name + '\n'
    out_text += 'API: ' + api + '\n'
    out_text += 'Event Type: ' + reason + '\n'
    out_text += 'Start Date: ' + start_date + '\n'
    out_text += 'Start Time: ' + start_time + '\n'
    out_text += 'End Date: ' + end_date + '\n'
    out_text += 'End Time: ' + end_time + '\n'
    out_text += 'Stage Number: ' + stage + '\n'
    out_text += 'Time ' + '(' + 'minutes' + ')' + ':' + stamped_time + '\n'
    out_text += 'Comments: ' + comments + '\n'

    # make UTC start stop time

    utc_start = mk_utc_time(start_date, start_time)
    utc_end = mk_utc_time(end_date, end_time)

    out_text += 'UTC start dateTime: ' + utc_start + '\n'
    out_text += 'UTC stop dateTime: ' + utc_end

    print(out_text)


def make_frac_score_data_set(df_well_ids, df_eff_report,
                             num_windows_per_stage,
                             counting_window_secs,
                             std_m, distance_val,
                             time_before_pop_secs,
                             time_after_pop_secs):
    now_str = datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    save_file_name = 'fracScore_allPads_' + now_str + '.npy'

    print('saving data under the filename: ' + save_file_name)

    frac_score_pops_all_pads = []

    total_events = len(df_eff_report)

    for i in range(total_events):

        print('analyzing record: ' + str(i) + ' of: ' + str(total_events) + '\n')

        well_name = str(df_eff_report.loc[i, "Well"])
        api = str(df_eff_report.loc[i, "API #"])
        stage = str(df_eff_report.loc[i, "Stage"])
        # crew = str(df_eff_report.loc[i, "Crew"])
        start_date = str(df_eff_report.loc[i, "Start Date"])
        start_time = str(df_eff_report.loc[i, "Start Time"])
        end_date = str(df_eff_report.loc[i, "End Date"])
        end_time = str(df_eff_report.loc[i, "End Time"])
        reason = str(df_eff_report.loc[i, "Reason"])
        comments = str(df_eff_report.loc[i, "Comments"])
        stamped_time = str(df_eff_report.loc[i, "TimeInMin"])

        if reason == 'Pumping':

            # print out well and job info

            print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage,
                                    stamped_time, comments)

            utc_start = mk_utc_time(start_date, start_time)
            utc_end = mk_utc_time(end_date, end_time)

            # make sure time is greater than 10 minutes

            if float(stamped_time) > 10:

                # pull IDs

                stat_id, dyn_id = fetch_sensor_ids_from_apis(df_well_ids, api)

                # draw X number of samples, or num_windows_per_stage

                for j in range(num_windows_per_stage):

                    print(' j - ', j)
                    print('num windows per stage: ', num_windows_per_stage)
                    print('counting windows secs: ', counting_window_secs)
                    print('float of stamped time : ' + stamped_time)
                    print('utc start: ' + utc_start)
                    print('utc end: ' + utc_end)

                    rdm_window_utc_start, rdm_window_utc_stop = mk_random_windows(counting_window_secs,
                                                                                int(float(stamped_time)),
                                                                                utc_start)
                    # now make data

                    x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dyn_id,
                                                                           stat_id,
                                                                           rdm_window_utc_start,
                                                                           rdm_window_utc_stop)
                    if len(x_d) > 0:

                        # We assume that the frac score is pretty decent at finding BASIS pops...

                        peaks_bp = frac_score_detector(x_d, sampling_rate, 150, 500, std_m, distance_val)

                        for pk in peaks_bp:

                            # ID activation second, build data before and after peak, then draw new values

                            if pk > 0.01 * sampling_rate:
                                activation_second = t_d[pk]

                                activation_time = mkDataTimeFromStr(rdm_window_utc_start) + dt.timedelta(
                                    seconds=activation_second)

                                # write time before and after

                                window_start = activation_time - dt.timedelta(seconds=time_before_pop_secs)
                                window_stop = activation_time + dt.timedelta(seconds=time_after_pop_secs)

                                # pull new data

                                x_d_win = interval_to_flat_array_resample(dyn_id, window_start, window_stop).values()

                                # now add all the good stuff

                                frac_score_pops_all_pads.append([well_name,
                                                                 api,
                                                                 stage,
                                                                 dtObj2str(window_start),
                                                                 dtObj2str(window_stop),
                                                                 x_d_win])

                    else:

                        print('unable to evaluate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100 * (i / total_events), 2)) + '\n\n')

    # save the data

    print('frac data build complete')

    np.save(save_file_name, frac_score_pops_all_pads)

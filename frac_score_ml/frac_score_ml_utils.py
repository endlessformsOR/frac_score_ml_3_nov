import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from numpy import dstack
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from dynamic_utils import interval_to_flat_array_resample, parse_time_string_with_colon_offset
from usefulmethods import dynamic2dashboard, fetch_sensor_db_data, logFFT, logFFT2Dmatrix, Welch_matrix_methods, quickFFTobjs, parse_time_string_with_colon_offset, mkDataTimeFromStr,dtObj2str
import os, sys, datetime, subprocess, time, db, statistics
import random
import scipy.signal as signal
import datetime as dt

# build training and testing data with this function

def load_ML_data_pops(POPS_FILENAME,NON_POPS_FILENAME,WINDOW_SIZE_SECS,TRAIN_TEST_SPLIT_PERCENT):

    # load npy file WN,Api,stage,dtObj2str(window_start),dtObj2str(window_stop),x_d_win

    print('loading data, this may take some time...')

    input_pop_event = np.load(POPS_FILENAME, allow_pickle=True)
    input_notpops = np.load(NON_POPS_FILENAME, allow_pickle=True)

    # force size of array, looks like some vals are less than expected

    forced_size_of_array = 40000*WINDOW_SIZE_SECS

    # setup output lists

    training_labels = []
    testing_labels = []
    training_vals = []
    testing_vals = []

    print('data loaded, now building training and testing datasets')

    # cycle through events, use ratio to split between training and testing

    ratio = int(100/(100-TRAIN_TEST_SPLIT_PERCENT))

    for i in range(len(input_pop_event)):

        array_in_question = input_pop_event[i][5]

        if len(array_in_question) == forced_size_of_array:

            if i%ratio==0:

                training_labels.append([1])
                training_vals.append(array_in_question)

            else:

                testing_labels.append([1])
                testing_vals.append(array_in_question)

    for j in range(len(input_notpops)):

        array_in_question = input_notpops[j][5]

        if len(array_in_question) == forced_size_of_array:

            # reshape

            if j%ratio==0:

                training_labels.append([0])
                training_vals.append(array_in_question)

            else:

                testing_labels.append([0])
                testing_vals.append(array_in_question)

    # convert output arrats to numpy arrays with appropriate dimensions

    print('shape of training_vals: ' + str(np.shape(training_vals)))

    training_vals = np.expand_dims(np.asarray(training_vals), axis=2)
    testing_vals = np.expand_dims(np.asarray(testing_vals), axis=2)
    training_labels = np.asarray(training_labels)
    testing_labels = np.asarray(testing_labels)

    print('testing and training data building complete.' + '\n')

    return training_vals, testing_vals, training_labels, testing_labels

def dynamic_model(n_outputs, window_size):
    # static model
    m = Sequential()
    m.add(layers.InputLayer(input_shape=(window_size, 1)))
    m.add(layers.BatchNormalization(name='batch_norm_1'))
    # new, added padding='same' instead of validm default
    #m.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='conv_1', padding='same'))
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

def evaluate_model(input_model,vals_train, labels_train, vals_test, labels_test):

    batch_size = 1
    n_outputs = 8
    n_classes = 8
    denseLayerNumber = 128


    input_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'],
                       )
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)
    # fit network
    input_model.fit(x=vals_train,
                   y=labels_train,
                   validation_data=(vals_test, labels_test),
                   #callbacks=[monitor],
                   epochs=20,
                   verbose=2)

    # evaluate model
    loss_vals, accuracy = input_model.evaluate(x=vals_test,
                                      y=labels_test,
                                      verbose=2)
    return accuracy

def make_frac_score_data_non_events(df_wellIDs, df_effReport, num_windows_per_stage, counting_window_secs, STD_M, distanceVal, timeBeforePop_secs, timeAfterPop_secs ):

    nowstr= datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    saveFileName = 'fracScore_allPads_non_events' + nowstr +  '.npy'

    print('saving data under the filename: ' + saveFileName)

    non_events_obj = []

    totalEvents = len(df_effReport)

    for i in range(totalEvents):

        print('analyzing record: ' + str(i) + ' of: ' + str(totalEvents) + '\n')

        WN = str(df_effReport.loc[i,"Well"])
        Api = str(df_effReport.loc[i, "API #"])
        stage = str(df_effReport.loc[i,"Stage"])
        Crew = str(df_effReport.loc[i,"Crew"])
        startDate = str(df_effReport.loc[i,"Start Date"])
        startTime = str(df_effReport.loc[i,"Start Time"])
        endDate = str(df_effReport.loc[i,"End Date"])
        endTime = str(df_effReport.loc[i,"End Time"])
        Reason = str(df_effReport.loc[i,"Reason"])
        Comments = str(df_effReport.loc[i,"Comments"])
        stampedTime = str(df_effReport.loc[i,"TimeInMin"])

        if Reason != 'Pumping':

            # Gather Well info for output text

            textstr = 'Well Name: ' + WN + '\n'
            textstr +='API: ' + Api + '\n'
            textstr +='Event Type: ' + Reason + '\n'
            textstr +='Start Date: ' + startDate + '\n'
            textstr +='Start Time: ' + startTime + '\n'
            textstr +='End Date: ' + endDate + '\n'
            textstr +='End Time: ' + endTime + '\n'
            textstr +='Stage Number: ' + stage + '\n'
            textstr +='Time ' + '(' + 'mins' + ')'+ ':'  + stampedTime + '\n'
            textstr +='Comments: ' + Comments + '\n'

            # make UTC start stop time

            utcStart = mkUTCtime(startDate,startTime)
            utcEnd = mkUTCtime(endDate,endTime)

            textstr +='UTC start dateTime: ' + utcStart + '\n'
            textstr +='UTC stop dateTime: ' + utcEnd

            print(textstr)

            # make sure time is greater than 10 mins!

            if float(stampedTime) > 10:

                # now pull IDs

                statID, dynID = fetchSensorIDsFromAPIs(df_wellIDs, Api)

                if dynID != '-1' and statID != '-1':

                    # we wish to draw X number of samples, or num_windows_per_stage

                    for j in range(num_windows_per_stage):

                        print(' j - ', j)
                        print('num windows per stage: ' , num_windows_per_stage)
                        print('counting windows secs: ',counting_window_secs)
                        print('float of stamped time : ' + stampedTime)
                        print('utc start: ' + utcStart)
                        print('utc end: ' + utcEnd)

                        rdmWindowUTCstart, rdmWindowUTCstop, minsIntoEvent =  mkRndmWindows(counting_window_secs, int(float(stampedTime)), utcStart, utcEnd)

                        # now make data

                        x_d,t_d,x_s,t_s,sampling_rate,seconds = makeData(dynID,statID,rdmWindowUTCstart,rdmWindowUTCstop)

                        if len(x_d) > 0:

                            # just pull a window

                            activationTime = mkDataTimeFromStr(rdmWindowUTCstart)
                            window_start = activationTime -  dt.timedelta(seconds=timeBeforePop_secs)
                            window_stop = activationTime +  dt.timedelta(seconds=timeAfterPop_secs)
                            # pull new data

                            x_d_win = interval_to_flat_array_resample(dynID,window_start,window_stop).values()

                            # now add all the good stuff

                            non_events_obj.append([WN,Api,stage,dtObj2str(window_start),dtObj2str(window_stop),x_d_win])

                        else:

                            print('unable to evaulate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100*(i/totalEvents),2)) + '\n\n')

    # save the data while you can!

    print('frac data build complete')

    np.save(saveFileName,non_events_obj)

def makeData(dynID, statID, startSTR, stopSTR):

    # pull dyn pressure data. Does not include accelerometer data

    start_converted = parse_time_string_with_colon_offset(startSTR)
    stop_converted = parse_time_string_with_colon_offset(stopSTR)
    delta = stop_converted - start_converted
    seconds = int(delta.total_seconds())
    x_d = interval_to_flat_array_resample(dynID,start_converted,stop_converted).values()
    #x_d = interval_to_flat_array_resample(dynID, start_converted, stop_converted)

    sampling_rate = int(len(x_d) / seconds)

    t_d = np.linspace(0,seconds,len(x_d))

    # make static data

    stat_obj = fetch_sensor_db_data(statID,startSTR,stopSTR)

    x_s = stat_obj['max'].to_numpy()
    t_s = np.linspace(0,len(x_s),len(x_s))

    return x_d,t_d,x_s,t_s,sampling_rate,seconds

def fracScoreDetector_v2(dynamic_window, sampling_rate, f_low, f_high, STD_M, distanceVal):

    ## Frac score method, updated version with bandpass filter

    filtered_xd = butter_bandpass_filter(dynamic_window,f_low,f_high,sampling_rate,order=9)

    # Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response

    std_dynamic_window = np.std(np.absolute(filtered_xd))

    # 6. Distance defines how each peak needs to be from each other, this avoids overcounting, see the scipy.signal docs for more info.
    #STD_MULTIPLIER = 14.5
    #distance = 1000

    # 7. The height threshold is the accpetable signal to noise ration, or relative magnitude of the filtered signal

    HEIGHT_THRESHOLD = STD_M*std_dynamic_window

    # 8. Using the signal find peaks method, see the scipy.signal docs for more info.

    peaks, _ = signal.find_peaks(np.absolute(filtered_xd), height=HEIGHT_THRESHOLD, distance=distanceVal)

    return peaks

def fetchSensorIDsFromAPIs(df_wellIDs, API2match):

    for i in range(len(df_wellIDs)):

        api_raw = str(df_wellIDs.loc[i,"api"])
        api_mod = api_raw.replace('-','')
        api_mod = api_mod[:-4]
        API2match = API2match.replace('-','')

        #print('analyzing record: ' + str(i) + ' of: ' + str(len(df_wellIDs))  + ' API2match: ' + API2match + ' APIfound: ' + api_mod)

        if API2match == api_mod:

            statID = str(df_wellIDs.loc[i,"static_id"])
            dynID = str(df_wellIDs.loc[i,"dynamic_id"])

    return statID, dynID

def mkUTCtime(strDate,strTime):

    # combine strings:

    date = strDate.split('/')
    time = strTime.split(':')

    mth = str(date[0])
    day = str(date[1])
    yr = str(date[2])
    hr = str(time[0])
    mn = str(time[1])
    #sec = '00.000000Z'
    #sec = '00.000000000Z'
    sec = '00'

    if len(date[0]) != 2:

        mth = '0' + mth

    if len(date[1]) != 2:

        day = '0' + day

    if len(time[0]) != 2:

        hr = '0' + hr

    if len(time[1]) != 2:

        mn = '0' + mn

    outSTR = '20' + yr + '-' + mth + '-' + day + 'T' + hr + ':' + mn + ':' + sec + '-' + '05:00'
    #outSTR = yr + '-' + mth + '-' + day + 'T' + hr + ':' + mn + ':' + sec + '-' + '05:00'

    #'2020-07-29T14:45:00-05:00'
    print('new utc time: ' + outSTR)
    outSTR

    return outSTR

def mkRndmWindows(windowSize, stampedTime, utcStart, utcEnd):

    newStart = int(random.uniform(0,int(stampedTime)))

    print('new start: ' , newStart)

    rdmWindowUTCstart = mkDataTimeFromStr(utcStart) + dt.timedelta(seconds=60*newStart)
    #rdmWindowUTCstart = parse_time_string_with_colon_offset(utcStart) + dt.timedelta(seconds=60*newStart)

    #print ('new random window UTC start: ' + rdmWindowUTCstart)

    rdmWindowUTCstop = rdmWindowUTCstart + dt.timedelta(seconds=windowSize)

    #print ('new random window UTC stop: ' + rdmWindowUTCstop)

    return dtObj2str(rdmWindowUTCstart), dtObj2str(rdmWindowUTCstop), newStart

def make_frac_score_data_set(df_wellIDs, df_effReport, num_windows_per_stage, counting_window_secs, STD_M, distanceVal, timeBeforePop_secs, timeAfterPop_secs ):

    nowstr= datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    saveFileName = 'fracScore_allPads_' + nowstr +  '.npy'

    print('saving data under the filename: ' + saveFileName)

    fracScore_popsAllPads = []

    totalEvents = len(df_effReport)

    for i in range(totalEvents):

        print('analyzing record: ' + str(i) + ' of: ' + str(totalEvents) + '\n')

        WN = str(df_effReport.loc[i,"Well"])
        Api = str(df_effReport.loc[i, "API #"])
        stage = str(df_effReport.loc[i,"Stage"])
        Crew = str(df_effReport.loc[i,"Crew"])
        startDate = str(df_effReport.loc[i,"Start Date"])
        startTime = str(df_effReport.loc[i,"Start Time"])
        endDate = str(df_effReport.loc[i,"End Date"])
        endTime = str(df_effReport.loc[i,"End Time"])
        Reason = str(df_effReport.loc[i,"Reason"])
        Comments = str(df_effReport.loc[i,"Comments"])
        stampedTime = str(df_effReport.loc[i,"TimeInMin"])

        if Reason == 'Pumping':

            # Gather Well info for output text

            textstr = 'Well Name: ' + WN + '\n'
            textstr +='API: ' + Api + '\n'
            textstr +='Event Type: ' + Reason + '\n'
            textstr +='Start Date: ' + startDate + '\n'
            textstr +='Start Time: ' + startTime + '\n'
            textstr +='End Date: ' + endDate + '\n'
            textstr +='End Time: ' + endTime + '\n'
            textstr +='Stage Number: ' + stage + '\n'
            textstr +='Time ' + '(' + 'mins' + ')'+ ':'  + stampedTime + '\n'
            textstr +='Comments: ' + Comments + '\n'

            # make UTC start stop time

            utcStart = mkUTCtime(startDate,startTime)
            utcEnd = mkUTCtime(endDate,endTime)

            textstr +='UTC start dateTime: ' + utcStart + '\n'
            textstr +='UTC stop dateTime: ' + utcEnd

            print(textstr)

            # make sure time is greater than 10 mins!

            if float(stampedTime) > 10:

                # now pull IDs

                statID, dynID = fetchSensorIDsFromAPIs(df_wellIDs, Api)

                # we wish to draw X number of samples, or num_windows_per_stage

                for j in range(num_windows_per_stage):

                    print(' j - ', j)
                    print('num windows per stage: ' , num_windows_per_stage)
                    print('counting windows secs: ',counting_window_secs)
                    print('float of stamped time : ' + stampedTime)
                    print('utc start: ' + utcStart)
                    print('utc end: ' + utcEnd)

                    rdmWindowUTCstart, rdmWindowUTCstop, minsIntoEvent =  mkRndmWindows(counting_window_secs, int(float(stampedTime)), utcStart, utcEnd)

                    # now make data

                    x_d,t_d,x_s,t_s,sampling_rate,seconds = makeData(dynID,statID,rdmWindowUTCstart,rdmWindowUTCstop)

                    if len(x_d) > 0:

                        # We assume that the frac score is pretty decent at finding BASIS pops...

                        peaks_bp = fracScoreDetector_v2(x_d, sampling_rate, 150, 500, STD_M, distanceVal)

                        for pk in peaks_bp:

                            # ID activation second, build data before and after peak, then draw new vals

                            if pk > 0.01*sampling_rate:

                                activation_second = t_d[pk]

                                activationTime = mkDataTimeFromStr(rdmWindowUTCstart) + dt.timedelta(seconds=activation_second)

                                # write time before and after

                                window_start = activationTime -  dt.timedelta(seconds=timeBeforePop_secs)
                                window_stop = activationTime +  dt.timedelta(seconds=timeAfterPop_secs)

                                # pull new data!

                                x_d_win = interval_to_flat_array_resample(dynID,window_start,window_stop).values()

                                # now add all the good stuff

                                fracScore_popsAllPads.append([WN,Api,stage,dtObj2str(window_start),dtObj2str(window_stop),x_d_win])

                    else:

                        print('unable to evaulate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100*(i/totalEvents),2)) + '\n\n')

    # save the data while you can!

    print('frac data build complete')

    np.save(saveFileName,fracScore_popsAllPads)

from scipy.signal import butter, sosfilt, sosfreqz, lfilter, freqz

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

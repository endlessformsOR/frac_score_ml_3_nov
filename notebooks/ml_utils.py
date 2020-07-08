
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd
import datetime as dt
from numpy import mean
from numpy import std
from numpy import dstack
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import scipy.signal as signal

from usefulmethods import fetch_sensor_db_data, parse_time_string_with_colon_offset, mkDataTimeFromStr,dtObj2str 
from dynamic_utils import interval_to_flat_array, parse_time_string_with_colon_offset

def loadDataset(STATIC_DATA_FILENAME, DYNAMIC_DATA_FILENAME, STATIC_WINDOW_SIZE, DYNAMIC_WINDOW_SIZE):
    
    print('loading dataset...')
    
    input_static_data = np.load(STATIC_DATA_FILENAME, allow_pickle=True)
    input_dynamic_data = np.load(DYNAMIC_DATA_FILENAME, allow_pickle=True)
    
    # reformat data for training and testing
    
    numEvents_static = len(input_static_data)
    numEvents_dynamic = len(input_dynamic_data)

    print('number of static events: ' + str(numEvents_static))
    print('number of dynamic events: ' + str(numEvents_dynamic))

    dataset_static = []
    dataset_dynamic = []

    #  input_data[i][j][k]
    
    #   ([WN,static_window_start_str,static_window_stop_str,sampling_rate,seconds,activation_second,static_event_array, outputLabel,Api,EVENT])
    
    #   number of data objects is represented by the -i index. in our case, we have 31 so far...
    #   data objects have a 10-parameter space shape. these are indexed within the -j column
    #
    #
    # j = 0 <---- WELL NAME
    # j = 1 <---- Start Time, UTC adjusted
    # j = 2 <---- End Time, UTC adjusted
    # j = 3 <---- Sampling Frequency in Hz
    # j = 4 <---- Total Elapsed Time of Event, in Seconds
    # J = 5 <---- Activation second where event begins on conditioned data set 
    # j = 6 <---- training vals, np array 
    # J = 7 <---- training labels, np array
    # j = 8 <---- API, str
    # J = 9 <---- EVENT TYPE, str
    
    ###NEW
    
    # populate list for static and dynamic stuff
    
    static_vals = []
    static_labels = []
    static_events = []
    
    dyn_vals = []
    dyn_labels = []
    dyn_events = []
    
    for i in range(len(input_static_data)):
        
        static_vals.append(input_static_data[i][6])
        static_labels.append(input_static_data[i][7])
        static_events.append(input_static_data[i][9])
        
    for j in range(len(input_dynamic_data)):
        
        dyn_vals.append(input_dynamic_data[j][6])
        dyn_labels.append(input_dynamic_data[j][7])
        dyn_events.append(input_dynamic_data[j][9])
        
    # 70-30 split for training/testing
    
    split_point_static = int(0.7 * len(input_static_data))
    
    static_vals_train = static_vals[:split_point_static]
    static_labels_train = static_labels[:split_point_static]
    static_events_train = static_events[:split_point_static]
    
    static_vals_test = static_vals[split_point_static:]
    static_labels_test = static_labels[split_point_static:]
    static_events_test = static_events[split_point_static:]
    
    split_point_dyn = int(0.7 * len(input_dynamic_data))
    
    dyn_vals_train = dyn_vals[:split_point_dyn]
    dyn_labels_train = dyn_labels[:split_point_dyn]
    dyn_events_train = dyn_events[:split_point_dyn]
    
    dyn_vals_test = dyn_vals[split_point_dyn:]
    dyn_labels_test = dyn_labels[split_point_dyn:]
    dyn_events_test = dyn_events[split_point_dyn:]
    
    # convert output arrays to numpy format
    
    static_vals_train = np.expand_dims(np.asarray(static_vals_train), axis=2)
    dyn_vals_train = np.expand_dims(np.asarray(dyn_vals_train), axis=2)

    static_vals_test = np.expand_dims(np.asarray(static_vals_test), axis=2)
    dyn_vals_test = np.expand_dims(np.asarray(dyn_vals_test), axis=2)
    
    static_labels_train = np.asarray(static_labels_train)
    dyn_labels_train = np.asarray(dyn_labels_train)
    
    static_labels_test = np.asarray(static_labels_test)
    dyn_labels_test = np.asarray(dyn_labels_test)
    
    static_events_train = np.asarray(static_events_train)
    dyn_events_train = np.asarray(dyn_events_train)
    
    static_events_test = np.asarray(static_events_test)
    dyn_events_test = np.asarray(dyn_events_test)
    
    return static_vals_train, dyn_vals_train, static_vals_test, dyn_vals_test, static_labels_train, dyn_labels_train, static_labels_test, dyn_labels_test, static_events_train, dyn_events_train, static_events_test,dyn_events_test

#
# CURRENT 1D CONVO MODEL TEMPLATES
# 

# STATIC PRESSURE MODEL 

def static_model(n_outputs,STATIC_WINDOW_SIZE):
    # static model 
    m = Sequential()
    m.add(layers.InputLayer(input_shape=(STATIC_WINDOW_SIZE,1)))
    m.add(layers.BatchNormalization(name='batch_norm_1'))
    m.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='conv_1'))
    m.add(layers.BatchNormalization(name='batch_norm_2'))
    m.add(layers.MaxPooling1D(pool_size=2, name='max_pool_1'))
    m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='conv_2'))
    m.add(layers.BatchNormalization(name='batch_norm_3'))
    m.add(layers.MaxPooling1D(pool_size=2, name='max_pool_2'))
    m.add(layers.Flatten(name='flatten'))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(32, activation='relu', name='dense_1'))
    m.add(layers.Dense(n_outputs, activation='softmax', name='output'))
    
    return m

# DYNAMIC PRESSURE MODEL

def dynamic_model(n_outputs,DYNAMIC_WINDOW_SIZE):
    # static model 
    m = Sequential()
    m.add(layers.InputLayer(input_shape=(DYNAMIC_WINDOW_SIZE,1)))
    m.add(layers.BatchNormalization(name='batch_norm_1'))
    m.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='conv_1'))
    m.add(layers.BatchNormalization(name='batch_norm_2'))
    m.add(layers.MaxPooling1D(pool_size=2, name='max_pool_1'))
    m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='conv_2'))
    m.add(layers.BatchNormalization(name='batch_norm_3'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_2'))
    m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='conv_3'))
    m.add(layers.BatchNormalization(name='batch_norm_4'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_3'))
    m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='conv_4'))
    m.add(layers.BatchNormalization(name='batch_norm_5'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_4'))
    m.add(layers.Flatten(name='flatten'))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(128, activation='relu', name='dense_1'))
    m.add(layers.Dense(n_outputs, activation='softmax', name='output'))
    
    return m

# evaluate model

def evaluate_model(input_model,vals_train, labels_train, vals_test, labels_test):
    
    batch_size = 1
    n_outputs = 8
    n_classes = 8
    denseLayerNumber = 128
    
    input_model.compile(loss='categorical_crossentropy', 
                       optimizer='adam', 
                       metrics=['accuracy'])
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

# summarize scores

def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# this is for buildTrainingDataSet

def greatestNegVal(inputList):
    
    output = 0
    for val in inputList:
        if (val < 0):
            if output == 0 or val < output:
                output = val
    return output

def buildTrainingDataSet(dataLoadOjb, df, STATIC_WINDOW_SEC, DYNAMIC_WINDOW_SEC,RESAMPLEDFREQ):
    
    # Build a general purpose static and dynamic training dataset for the following events:
        
    #EVENT = 'PUMPDOWN_PERFS_PLUGS_START'
    #EVENT = 'PUMPDOWN_PERFS_PLUGS_STOP'
    #EVENT = 'PERF_GUN_FIRING'
    #EVENT = 'ENTIRE_PERF_EVENT'
    #EVENT = 'FRAC_STAGE_START'
    #EVENT = 'FRAC_STAGE_STOP'
    #EVENT = 'FORMATION_BREAKDOWN'
    #EVENT = 'PRESSURIZATION STEP'
    #EVENT = 'FRACS_ON_ACTIVE_WELL'
    #EVENT = 'FRACS_ON_OFFSET_WELL'
    #EVENT = 'GEAR_SHIFT'
    #EVENT = 'WELL_COMMUNICATION'
    #EVENT = 'FRICTION'
    #EVENT = 'OTHER_NEEDS_CLASSIFICATION'
    #EVENT = 'PERF_ENTIRE_WINDOW'
    #EVENT = 'ENTIRE_WINDOW'
    #EVENT = 'STUCK_WIRELINE'
    
    ## add events to eventList if there are heuristic conditions applied to the if elif block!
    
    eventList = ['PUMPDOWN_PERFS_PLUGS_START' , 'PUMPDOWN_PERFS_PLUGS_STOP' , 'PERF_GUN_FIRING' , 'FRAC_STAGE_START' , 'FRAC_STAGE_STOP', 'FORMATION_BREAKDOWN', 'PRESSURIZATION STEP', 'GEAR_SHIFT']
    
    # create id matrix, use rows as output arrays for each class
    
    classLabelMatrix = np.identity(len(eventList))

    # Total size of objects
    
    sizeOFstaticObjs = STATIC_WINDOW_SEC + 1
    sizeOfDynamicObjs = RESAMPLEDFREQ*DYNAMIC_WINDOW_SEC

    # Instantiate dataclass

    c = dataLoadOjb

    # create output master training data object, this will be populated and saved as a .npz file

    output_training_static_data_obj = []
    output_training_dynamic_data_obj = []

    # iterate through dataset 

    seconds_list = []
    stat_evnCntr = 0
    dyn_evnCntr = 0

    for i in range(len(df)):

        EVENT = str(df.loc[i,"EVENT_CLASS"])
        startTime = str(df.loc[i,"START_TIME"])
        endTime = str(df.loc[i, "STOP_TIME"])
        Api = str(df.loc[i, "WELL_API"])
        WN = str(df.loc[i,"WELL_NAME"])
        staticID = str(df.loc[i,"STATIC_SENSOR_ID"])
        dynamicID = str(df.loc[i,"DYNAMIC_SENSOR_ID"])
        print(staticID, dynamicID)
        metadata = str(df.loc[i,"METADATA"])
        tags = str(df.loc[i,"TAGS"])
        stage = str(df.loc[i,"STAGE"])

        # Well Info 

        textstr = 'Well Name: ' + WN + '\n' 
        textstr +='API: ' + Api + '\n' 
        textstr +='Event Type: ' + EVENT + '\n' 
        textstr +='Start: ' + startTime + '\n' 
        textstr +='End: ' + endTime + '\n'
        textstr +='Stage Number: ' + stage + '\n'
        textstr +='class: ' + tags + '\n'
        textstr +='comments: ' + metadata + '\n'

        # generate labeled data from conditioned events
            
        print('---record number: ' + str(i) + '  --- looking at event:' + EVENT + ' for: ' + WN)
        
        if EVENT in eventList:
            
            t_s, x_s, t_d, x_d, sampling_rate, seconds, textstr, startTime_df, endTime_df, staticID, dynamicID  = c.pullDataObjsFromDF(df, i)
            
            activation_second = 0
            indexOfEventLabel = eventList.index(EVENT)

            if len(x_d) > 0:

	            for sec in range(seconds):
	                                
	                # do analysis by the second, determine time window for each well

	                dynamic_window = x_d[int(sec*sampling_rate):int((sec+1)*sampling_rate)]

	                # criteria for IDing initation of event

	                var_dyn = np.var(dynamic_window)
	                mean_dyn = np.mean(dynamic_window)
	                max_amplitude = np.amax(np.absolute(dynamic_window))

	                # for start/stop stage

	                earlyPart = np.mean(dynamic_window[0:int(len(dynamic_window)/5)])
	                latePart = np.mean(dynamic_window[int(4+len(dynamic_window)/5):-1])

	                # look at static pressure, 10 second window, look for overall increase during interval

	                init_stat_press = x_s[0]
	                final_stat_press = x_s[-1]
	                    
	                # HEURISTIC BLOCK FOR CENTERING ARRAYS WRT EVENT
	                
	                if EVENT == 'PUMPDOWN_PERFS_PLUGS_START':

	                    outputLabel = classLabelMatrix[indexOfEventLabel]

	                    if var_dyn > 130:

	                        if activation_second == 0:

	                            activation_second = sec

	                elif EVENT == 'PUMPDOWN_PERFS_PLUGS_STOP':

	                    outputLabel = classLabelMatrix[indexOfEventLabel]

	                    if greatestNegVal(dynamic_window) < -70:

	                        if activation_second == 0:

	                            activation_second = sec

	                elif EVENT == 'PERF_GUN_FIRING':

	                    outputLabel = classLabelMatrix[indexOfEventLabel]

	                    if max_amplitude > 250:

	                        if activation_second == 0:

	                            activation_second = sec

	                elif EVENT == 'FRAC_STAGE_START':

	                    outputLabel = classLabelMatrix[indexOfEventLabel]

	                    if var_dyn > 15 and mean_dyn > 10 and max_amplitude > 10 and 5*earlyPart > latePart and final_stat_press > 1.02*init_stat_press:
	                            
	                        if activation_second == 0:
	                                
	                            activation_second = sec

	                elif EVENT == 'FRAC_STAGE_STOP':
	                    
	                    outputLabel = classLabelMatrix[indexOfEventLabel]

	                    if var_dyn > 15 and mean_dyn > 10 and max_amplitude > 10 and 5*earlyPart > latePart and final_stat_press < 1.02*init_stat_press:

	                        if activation_second == 0:

	                            activation_second = sec
	                            
	                elif EVENT == 'FORMATION_BREAKDOWN':
	                    
	                    outputLabel = classLabelMatrix[indexOfEventLabel]
	                    
	                    # this occurs typically after the largest pressurization step. for now, we center event around middle of window... 
	                    
	                    activation_second = int(seconds / 2)
	                    
	                elif EVENT == 'PRESSURIZATION STEP':

	                    outputLabel = classLabelMatrix[indexOfEventLabel]

	                    if len(x_s) != 0:
	                        
	                        if (sec - 5) < 0:
	                            
	                            init_stat_press = x_s[0]
	                            final_stat_press = x_s[5]
	                        
	                        elif (sec + 5) > len(x_s):
	                            
	                            init_stat_press = x_s[len(x_s)-5]
	                            final_stat_press = x_s[len(x_s)-1]
	                        
	                        elif (sec - 5) < len(x_s) and ( sec - 5 ) > 0:
	                            
	                            init_stat_press = x_s[sec-5]
	                            final_stat_press = x_s[sec]
	                        
	                        else:
	                            
	                            init_stat_press = x_s[0]
	                            final_stat_press = x_s[0]
	                        
	                        Pf_div_Pi = final_stat_press / init_stat_press
	                        
	                        if var_dyn > 120 and Pf_div_Pi > 1.04 and max_amplitude > 30:
	                            
	                            if activation_second == 0:
	                                activation_second = sec


	                elif EVENT == 'GEAR_SHIFT':
	                    
	                    outputLabel = classLabelMatrix[indexOfEventLabel]
	                    
	                    # use frac score logic to filter out low frequency noise
	                            
	                    sos = signal.butter(10,800,btype='highpass',fs=sampling_rate,output='sos')

	                    # Filtered_xd is the dynamic pressure response, with f < 800 Hz contributions removed. 

	                    filtered_xd = signal.sosfilt(sos,dynamic_window)

	                    # Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response

	                    std_dynamic_window = np.std(np.absolute(filtered_xd))

	                    # Distance defines how each peak needs to be from each other, this avoids overcounting, see the scipy.signal docs for more info. 
	                    STD_MULTIPLIER = 10
	                    distanceVal = 1000 

	                    # The height threshold is the accpetable signal to noise ration, or relative magnitude of the filtered signal 

	                    HEIGHT_THRESHOLD = STD_MULTIPLIER*std_dynamic_window 

	                    # Using the signal find peaks method, see the scipy.signal docs for more info. 

	                    peaks, _ = signal.find_peaks(filtered_xd, height=HEIGHT_THRESHOLD, distance=distanceVal)
	                    
	                    # we have seen that gear shifts are two quick spikes or pops of equal magnitude, so look for this here:
	                    
	                    if len(peaks) > 2:
	                        
	                        # we have a gear shift
	                        
	                        if activation_second == 0:

	                            activation_second = sec
	                    
	                else:
	                    
	                    # we have an event we have yet to heuristically interpret for training purposes...

	                    print('unable to ID event for training, placing activation second at 0. ')

	                    activation_second = 0
	                    
	                    # output array of zeroes
	                    
	                    outputLabel = np.zeros(len(eventList))
	                    
	                    
            # only write out files with an activation second
                
            if activation_second != 0:
            
                #reposition at center and save, define the activation second as a time string

                activationTimeStr = dtObj2str(mkDataTimeFromStr(startTime_df) +  dt.timedelta(seconds=activation_second))

                # new window will be centered around this new start time. 

                # 2. STATIC: revise start/stop times around activtion second

                static_window_half = int(STATIC_WINDOW_SEC/2)

                static_window_start_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) -  dt.timedelta(seconds=static_window_half))
                static_window_stop_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) +  dt.timedelta(seconds=static_window_half))

                # 3. pull static data from this window

                stat_obj = fetch_sensor_db_data(staticID,static_window_start_str,static_window_stop_str)
                static_event_array = stat_obj['max'].to_numpy()

                # 4. pull dynamic data from this window

                dynamic_window_half = int(DYNAMIC_WINDOW_SEC/2)
                dyn_window_start_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) -  dt.timedelta(seconds=dynamic_window_half))
                dyn_window_stop_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) +  dt.timedelta(seconds=dynamic_window_half))
                dynamic_event_array = interval_to_flat_array(dynamicID, parse_time_string_with_colon_offset(dyn_window_start_str), parse_time_string_with_colon_offset(dyn_window_stop_str)).values()

                if len(dynamic_event_array) != 0:

                    dyn_evnt_downsampled = signal.resample(dynamic_event_array,sizeOfDynamicObjs)

                # 5. Make sure arrays meet standard length requirement

                print('static_len array:' , len(static_event_array))
                print('REQUIRED array size:' , sizeOFstaticObjs)

                ##### NEED TO ADDRESS STATIC PRESSURE ARRAY ISSUE, PADDING FOR NOW

                if len(static_event_array) != 0:

                    # sometimes the array is smaller

                    if len(static_event_array) < sizeOFstaticObjs:

                        diff = np.absolute(sizeOFstaticObjs - len(static_event_array))
                        
                        print('diff: ' , diff)

                        lastVal = static_event_array[-1]

                        for i in range(diff):
                            print('i = ', i)
                            print('len of array before:', len(static_event_array))
                            #np.append(static_event_array,lastVal,axis=0) # doesnt work
                            s_list = static_event_array.tolist()
                            s_list.append(lastVal)
                            static_event_array = np.asarray(s_list)
                            print('len of array after:', len(static_event_array))
                        
                        print('new static len: ' + str(len(static_event_array)))

                    # sometimes the array is larger

                    if len(static_event_array) > sizeOFstaticObjs:

                        diff = np.absolute(sizeOFstaticObjs - len(static_event_array))

                        static_event_array[:-diff]

                    print(' new static_len array:' , len(static_event_array))

                print('dyn_len array:' , len(dynamic_event_array))
                print('dwnsmpld array:' , len(dyn_evnt_downsampled))

                print('outputlabel: ', outputLabel)

                # CHECK STUFF HERE!!

                if len(static_event_array) == sizeOFstaticObjs:

                    # save to array

                    stat_evnCntr += 1
                    #static_data_obj = np.array([WN,static_window_start_str,static_window_stop_str,sampling_rate,seconds,activation_second,static_event_array,'1',static_non_event_array,'0'])

                    static_data_obj = np.array([WN,static_window_start_str,static_window_stop_str,sampling_rate,seconds,activation_second,static_event_array, outputLabel,Api,EVENT])
                    output_training_static_data_obj.append(static_data_obj)

                    print('static event added successfully. ')
                    print('total static events:', stat_evnCntr)

                if len(dyn_evnt_downsampled) == sizeOfDynamicObjs:

                    dynamic_data_obj = np.array([WN,dyn_window_start_str,dyn_window_stop_str,sampling_rate,seconds,activation_second,dyn_evnt_downsampled, outputLabel,Api,EVENT])
                    output_training_dynamic_data_obj.append(dynamic_data_obj)

                    dyn_evnCntr += 1

                    print('dynamic event added successfully. ')
                    print('total dynamic events:', dyn_evnCntr)


                print('end total static events:', stat_evnCntr)
                print('end total dynamic events:', dyn_evnCntr)
            
        else:
            
            print('moving on, our model has yet to train over this')
        
    return output_training_static_data_obj, output_training_dynamic_data_obj

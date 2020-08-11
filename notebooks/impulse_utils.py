import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot
import scipy.signal as signal
from usefulmethods import fetch_sensor_db_data, parse_time_string_with_colon_offset, mkDataTimeFromStr,dtObj2str, dynamic2dashboard 
from dynamic_utils import interval_to_flat_array, parse_time_string_with_colon_offset



def fracScoreDetector(dynamic_window, sampling_rate):
    
    ## Frac score method. Uses a high pass filter of 800Hz and peak detect. 
        
    sos = signal.butter(10,800,btype='highpass',fs=sampling_rate,output='sos')
    
    # Filtered_xd is the dynamic pressure response, with f < 800 Hz contributions removed. 
    
    filtered_xd = signal.sosfilt(sos,dynamic_window)
    
    # Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response
    
    std_dynamic_window = np.std(np.absolute(filtered_xd))

    # Distance defines how each peak needs to be from each other, this avoids overcounting, see the scipy.signal docs for more info.
    
    STD_MULTIPLIER = 14.5 # this should vary depeneding on order of stage...
    distanceVal = 1000 
    
    # The height threshold is the accpetable signal to noise ration, or relative magnitude of the filtered signal 

    HEIGHT_THRESHOLD = STD_MULTIPLIER*std_dynamic_window 
    
    # Using the signal find peaks method, see the scipy.signal docs for more info. 

    peaks, _ = signal.find_peaks(filtered_xd, height=HEIGHT_THRESHOLD, distance=distanceVal)
    
    return peaks

def makeData(dynID, statID, startSTR, stopSTR):
    
    # A Quick method to pull numpy arrays from datetime strings. 
    # Out:
    # x_d = dynamic pressure data numpy array
    # t_d = dynamic time data, for x_d
    # x_s = static pressure data numpy array
    # t_s = static time data for x_s
    # sampling rate is in Hz. It may vary
    # seconds - total elapsed time from strings. 
    
    # add an assert here at stop > start some point...
    
   # Determine length of time from strings
    
    start_converted = parse_time_string_with_colon_offset(startSTR)
    stop_converted = parse_time_string_with_colon_offset(stopSTR)
    delta = stop_converted - start_converted
    seconds = int(delta.total_seconds())
    
    # pull dynamic data
    # include logic for when seconds is over 3600 later
    
    x_d = interval_to_flat_array(dynID, start_converted, stop_converted).values()
    sampling_rate = int(len(x_d) / seconds)
    t_d = np.linspace(0,seconds,len(x_d))

    # make static data
    
    stat_obj = fetch_sensor_db_data(statID,startSTR,stopSTR)
    x_s = stat_obj['max'].to_numpy()
    t_s = np.linspace(0,len(x_s),len(x_s))
    
    return x_d,t_d,x_s,t_s,sampling_rate,seconds

## BANDPASS FILTER USED TO REMOVE TOOL NOISE DURING PUMPDOWN, works well since the events seem to fall within 30 - 200 Hz

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

# fft methods

def mkFFTstuff(signal, sampling_rate):
    #
    # Use this to make an FFT object as well as a Power Spectral Density Object for plotting 
    #
    n = int(len(signal))
    f = signal
    dt = 1 / sampling_rate
    
    # make the fft object, fhat
    
    fhat = np.fft.fft(f,n)
    
    # make the power spectral density object, psd
    
    PSD = fhat * np.conj(fhat) / n
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(1,np.floor(n/2),dtype='int')
    
    return L,freq, PSD

def makePerfGunDataset(RECORDS, secsBeforeEvent,secsAfterEvent):
    
    # use info from .xls to cycle through events and populate an .npy object
    
    df = pd.read_excel(RECORDS)
    
    EVENT = 'PERF_GUN_FIRING'
    
    # output data object

    sequentialPerfEvents = []

    # Count the number of current events for this event type

    print('\n\n')
    print('Total number of ' + EVENT + ' events: ' + str(len(df[df['EVENT_CLASS']==EVENT])))
    print('\n\n')

    for i in range(len(df)):
        
        eventType = str(df.loc[i,"EVENT_CLASS"])
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
        fracPlugDepth = str(df.loc[i,"FRAC_PLUG_DEPTH"])
        
        # Gather Well info for output text 

        textstr = 'Well Name: ' + WN + '\n' 
        textstr +='API: ' + Api + '\n' 
        textstr +='Event Type: ' + eventType + '\n' 
        textstr +='Start: ' + startTime + '\n' 
        textstr +='End: ' + endTime + '\n'
        textstr +='Stage Number: ' + stage + '\n'
        textstr +='frac plug depth: ' + fracPlugDepth + '\n'

        if eventType == EVENT:

            print('\n\n' + textstr)

            # generate dynamic and static data

            x_d,t_d,x_s,t_s,sampling_rate,seconds = makeData(dynamicID, staticID, startTime, endTime)
            
            print('total elapsed time in seconds: ' +str(seconds))
            print('x_s length: ' + str(len(x_s)))
            print('x_d length: ' + str(len(x_d)))
            print('sampling rate: ' + str(sampling_rate))
            
            # Use frac score method to determine pop locations
            
            peaks = fracScoreDetector(x_d, sampling_rate)

            # now plot each event 

            fracEventList = []

            for idx, pk in enumerate(peaks):

                activation_second = t_d[pk]

                print('sec is: ' + str(activation_second))
                
                if activation_second > 0.1: # remove large blip at beginning of data 

                    #reposition at center and save, define the activation second as a time string

                    activationTimeStr = dtObj2str(mkDataTimeFromStr(startTime) +  dt.timedelta(seconds=activation_second))

                    #window_start_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) -  dt.timedelta(seconds=window_half))

                    window_start_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) -  dt.timedelta(seconds=secsBeforeEvent))
                    window_stop_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) +  dt.timedelta(seconds=secsAfterEvent))
                    dynamic_event_array = interval_to_flat_array(dynamicID, parse_time_string_with_colon_offset(window_start_str), parse_time_string_with_colon_offset(window_stop_str)).values()

                    outputObj = np.array([WN,Api,window_start_str,window_stop_str,stage,idx,secsBeforeEvent+secsAfterEvent,sampling_rate,seconds,activation_second,fracPlugDepth,dynamic_event_array])

                    sequentialPerfEvents.append(outputObj)
                    
                    print('peak added to data object')
            

    return sequentialPerfEvents


def makeActiveFracsDataset(RECORDS, secsBeforeEvent,secsAfterEvent):
    
    df = pd.read_excel(RECORDS)
    
    EVENT = 'FRACS_ON_ACTIVE_WELL'
    
    # output data object

    outputFracsDetected = []
    
    plotObjects = []

    # Count the number of current events for this event type

    print('\n\n')
    print('Total number of ' + EVENT + ' events: ' + str(len(df[df['EVENT_CLASS']==EVENT])))
    print('\n\n')

    for i in range(len(df)):
        
        eventType = str(df.loc[i,"EVENT_CLASS"])
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
        fracPlugDepth = str(df.loc[i,"FRAC_PLUG_DEPTH"])
        # Gather Well info for output text 

        textstr = 'Well Name: ' + WN + '\n' 
        textstr +='API: ' + Api + '\n' 
        textstr +='Event Type: ' + eventType + '\n' 
        textstr +='Start: ' + startTime + '\n' 
        textstr +='End: ' + endTime + '\n'
        textstr +='Stage Number: ' + stage + '\n'
        textstr +='frac plug depth: ' + fracPlugDepth + '\n'

        if eventType == EVENT:

            print('\n\n' + textstr)

            # generate data
            
            x_d,t_d,x_s,t_s,sampling_rate,seconds = makeData(dynamicID, staticID, startTime, endTime)
            
            x_d_r, t_d_r = dynamic2dashboard(x_d, seconds, t_d, sampling_rate)
            
            titleTxt = 'Pressure Response for Briscoe Chupadera 04 HA' + ' start: ' + startTime + ' stop: ' + endTime

            # run through event per minute, use frac score method to pick out events

            #outputFracsDetected = []

            fpm_time = []
            fpm_vals = []
            totalFracs = 0

            totalMins = seconds / 60

            print('total mins:',totalMins)

            for min in range(int(totalMins)):

                fpm_time.append(min)

                stat_vals = x_s[(min*60):((min+1)*60)]
                stat_time = t_s[(min*60):((min+1)*60)]
                dyn_vals = x_d[(min*60)*sampling_rate:((min+1)*60)*sampling_rate]
                dyn_time = t_d[(min*60)*sampling_rate:((min+1)*60)*sampling_rate]

                # run frac score algo to find peaks

                peaks = fracScoreDetector(dyn_vals, sampling_rate)

                finalPeakNum = 0
                plotLinesList = []
                
                for c, pk in enumerate(peaks):

                    activation_second = dyn_time[pk]

                    if pk > 0.01*sampling_rate:

                        finalPeakNum += 1
                        totalFracs += 1

                        activationTimeStr = dtObj2str(mkDataTimeFromStr(startTime) + dt.timedelta(seconds=(min*60)+activation_second))

                        # write 10-sec start stop time strings
                        
                        idx = pk

                        window_start_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) -  dt.timedelta(seconds=secsBeforeEvent))
                        window_stop_str = dtObj2str(mkDataTimeFromStr(activationTimeStr) +  dt.timedelta(seconds=secsAfterEvent))
                        total_elpased_second_stage = (min*60)+activation_second
                        
                        # add frac info to list
                        
                        outputFracsDetected.append(np.array([WN,Api,window_start_str,window_stop_str,stage,idx,secsBeforeEvent+secsAfterEvent,sampling_rate,seconds,activation_second,fracPlugDepth,dynamicID]))

                        # add line to lines list

                        plotLinesList.append(total_elpased_second_stage)

                # add number of peaks 

                fpm_vals.append(finalPeakNum)

            print('frac review complete for: ' + str(WN) + ' stage: ' +str(stage) + ' number of events recorded:' + str(totalFracs))
            
            # add plot objects

            plotObjects.append([t_s, x_s,t_d_r, x_d_r,fpm_time,fpm_vals,(seconds/60),plotLinesList,titleTxt])
            
           
    return outputFracsDetected, plotObjects

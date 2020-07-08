#
#
#   DEF FOR FRAC SCORE ID, LABELS FROM DYNAMIC PRESSURE
#
#  Things to consider before upgrading to production:

## 1. OUTPUT AS .PY, PUT IT IN GIT, SOME KIND OF VERSION CONTROL. 
## 2. ARE WE OVERFITTING TO CHUPA 04? TRY ON DIFFERENT WELLS

def mkLabelArray(dyn_presure_time, dyn_presure_displacement, sampling_rate, window_in_seconds, TIME_WIDTH, STD_MULTIPLIER):
    
    # This method takes a dynamic pressure signal and time np array, along with some calibrated parameters, and returns a label array (of equivalent size) for ML. 
    # 
    #      method inputs:
    #
    #   dyn_presure_time, obtained from dyamic pressure data, assumed to be a numpy array
    #   dyn_presure_displacement, obtained from dyanmic pressure data, assumed to be a numpy array
    #   sampling_rate, assumed to ~ 43 kHz
    #   window_in_seconds, for calibration purposes, this is 10 seconds, but can be adjusted to 1 second
    #   TIME_WIDTH, 1's padding around detected event, set to 0.25 ms
    #   STD_MULTIPLIER, the detector function calculates the stanard deviation of the dynamic pressure data. the Multipler is a scalar used to determine the detector threshold
    #                   we anticipate that most of the broad spectral pops occur beyond 4 STD vals of the dynamic pressure distribution. 
    #
    #
    #      method outputs:
    #
    #   sqr_wave_label_time, a numpy array with equivalent length and size as the dyn_presure_displacement input array
    #   sqr_wave_label_vals, a numpy array with equivalent length and size as the dyn_presure_displacement input array
    #   detectorStatsTxt, a string that reports the detected events in seconds with microsecond precision. 
    
    # 1. Import necessary libraries

    import scipy.signal as signal
    import numpy as np
    
    # 2. Define a highpass, butterworth filter, used to remove the seismic, low requency response

    sos = signal.butter(10,250,btype='highpass',fs=sampling_rate,output='sos')
    
    # 3. Filtered_xd is the dynamic pressure response, with f < 250 Hz contributions removed. 
    
    filtered_xd = signal.sosfilt(sos,dyn_presure_displacement)
    
    # 4. Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response
    
    std_dynamic_window = np.std(np.absolute(filtered_xd))
    
    # 5. This is a 1-d array array of time values, used for book keeping

    time = np.linspace(0,window_in_seconds,len(filtered_xd))

    # 6. Distance defines how each peak needs to be from each other, this avoids overcounting, see the scipy.signal docs for more info. 
    
    distanceVal = 1000 
    
    # 7. The height threshold is the accpetable signal to noise ration, or relative magnitude of the filtered signal 

    HEIGHT_THRESHOLD = STD_MULTIPLIER*std_dynamic_window 
    
    # 8. Using the signal find peaks method, see the scipy.signal docs for more info. 

    peaks, _ = signal.find_peaks(filtered_xd, height=HEIGHT_THRESHOLD, distance=distanceVal)

    # 9. Output array: sqr_wave_label_time vals are from 0 to window_in_seconds, with number of elements equal to the dyn_presure_time and displacement

    sqr_wave_label_time = np.linspace(0,window_in_seconds,len(dyn_presure_time))
    
    # 10. Output array: sqr_wave_label_val values are originaly zero, with number of elements equal to the dyn_presure_time and displacement

    sqr_wave_label_vals = np.zeros(len(dyn_presure_time))

    # 11. Report peak locations to detectorStatsTxt string 

    detectorStatsTxt = 'peaks detected at times (secs): ' + '\n\n'
    cnt = 1
    
    # 12. For each detected peak, we need to pad 1's around it. 

    for pk in peaks:

        # 13. Add the detected peak time value to the output string

        detectorStatsTxt += str(cnt) + ' - ' + "{:.{}f}".format( time[pk], 6 ) + '\n'
        cnt += 1

        # 14. ID the Time in question, to begin to pad 1's around the event. 

        time_in_question = time[pk]

        # 15. Idx is the index of the time in question, with respect to the output array, sqr_wave_label_time

        idx = np.where(sqr_wave_label_time == time_in_question)
        
        # 16. This makes a spike at center of pop.
        
        sqr_wave_label_vals[idx] = 1

        # 17. Find index vals around peak and force them be 1

        index_width = int(TIME_WIDTH*(sampling_rate*window_in_seconds))

        for i in range(int(index_width)):
            
            if (int(idx[0])-i) > 0:
                
                # 18. Add 1's before the event
            
                sqr_wave_label_vals[int(idx[0])-i] = 1

                # 19. Add 1's after the event

                sqr_wave_label_vals[int(idx[0])+i] = 1
                
            if (int(idx[0])-i) < 0:
                
                # index is negative, cut off padded 1's at origin
                
                sqr_wave_label_vals[int(idx[0])+i] = 1
            
    # 20. Return the output arrays for training 
    
    return sqr_wave_label_time,sqr_wave_label_vals,detectorStatsTxt

# THE DETECTOR FUNTION

# import scipy.signal as signal 

    # Use this to determine the amount of frac hits over a time interval
    # unfiltered data is filtered with a highpass butterworth filter (250Hz cutoff)
    # then the signal is normalized and a SNR criteria is used to pick high spectral/amp pops, 
    # which may be fracture hits
    # 
    # Calibrated using picks from Reid with the following conditions:
    # window_in_seconds = 10s
    # sampling_rate ~ 43,000 Hz
    # 0.08 < signal_threshold < 0.12
    # distanceVal = 1000

def detector_function_template(sampling_rate, t_s, x_s, t_d, x_d, detection_time_seconds, STD_MULTIPLIER):

    import scipy.signal as signal
    import numpy as np

    # define lists and ints
    
    total_fracs_time = []
    total_fracs_vals = []
    
    fpm_time = []
    fpm_fracspermin = []
    
    fps_time = []
    fps_vals = []
    
    total_fracs = 0
    total_fps = 0
    fps = 0
    fpm = 0
    total_mins = 0
    total_seconds = 0
    
    std_sec_time = []
    std_sec_val = []
    
    # determine length of time period for dataobjects
    
    elapsed_time_seconds = int(len(x_d) / sampling_rate)

    print('total number of seconds is: ' + str(elapsed_time_seconds))
    
    for sec in range(elapsed_time_seconds):
        
        total_seconds += 1
        
        #### ADD DETECTOR LOGIC HERE


        sos = signal.butter(10,250,btype='highpass',fs=sampling_rate,output='sos')
    
        # 3. Filtered_xd is the dynamic pressure response, with f < 250 Hz contributions removed. 
    
        filtered_xd = signal.sosfilt(sos,x_d)
    
        # 4. Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response
    
        std_dynamic_window = np.std(np.absolute(filtered_xd))
    
        # 6. Distance defines how each peak needs to be from each other, this avoids overcounting, see the scipy.signal docs for more info. 
    
        distanceVal = 1000 
    
        # 7. The height threshold is the accpetable signal to noise ration, or relative magnitude of the filtered signal 

        HEIGHT_THRESHOLD = STD_MULTIPLIER*std_dynamic_window 
    
        # 8. Using the signal find peaks method, see the scipy.signal docs for more info. 

        peaks, _ = signal.find_peaks(filtered_xd, height=HEIGHT_THRESHOLD, distance=distanceVal)
        
        # apply high-pass filter and normalize
                
                

        print('std is: ' + str(std_dynamic_window))
                
        std_sec_time.append(sec)
        std_sec_val.append(std_dynamic_window)
            
        # add counted peaks to total frac hits
                
        total_fracs += len(peaks)
        total_fps += len(peaks)
                
        fps += len(peaks)
        
        ### IF YOU WANT TO SPECIFY DETECTION WINDOW.. right now its 1 sec of detection time
        #if total_seconds % 10 == 0:
            

        if total_seconds % 60 == 0:
            
            total_mins += 1
                    
            fpm = total_fps
            
            fpm_fracspermin.append(fpm)
            
            # clear it out
            
            fpm = 0
            total_fps = 0
            
        else:
            
            # clear out fpms for non 60 second multiples
            
            fpm_fracspermin.append(0)
        
        # report total fracs
        
        total_fracs_vals.append(total_fracs)
        
        # append time vals
        
        total_fracs_time.append(sec)
        
        fpm_time.append(sec)
        
        fps_vals.append(fps)
        
        # end of second, clear out fps
        
        fps = 0
    
    # stats for output, use format: "{:.{}f}".format( global_max_x_val_abs, 3 )
    
    total_fracs_out = total_fracs
    avg_fps_out = np.average(fps_vals)
    var_fps_out = np.var(fps_vals)
    
    # for FPM stuff, we need to remove zeros, label them as NaNs 
    
    new_fpm = fpm_fracspermin
    new_fpm = [x for x in new_fpm if x != 0]
    
    avg_fpm_out = np.nanmean(new_fpm)
    var_fpm_out = np.nanvar(new_fpm)
    
    # format output strings: 
    
    str_det_time = "{:.{}f}".format( detection_time_seconds, 0 )
    str_dyn_var_thr = "{:.{}f}".format( dyn_variance_threshold, 2 )
    str_stat_p_thre = "{:.{}f}".format( static_pressure_threshold, 0 )
    str_highpass_freq_cutoff = "{:.{}f}".format( highpass_freq_cutoff, 2 )
    str_signal_threshold = "{:.{}f}".format( signal_threshold, 3 )
    str_distanceVal = "{:.{}f}".format( distanceVal, 0 )
    str_total_fracs_out = "{:.{}f}".format( total_fracs_out, 0 )
    str_avg_fps_out = "{:.{}f}".format( avg_fps_out, 0 )
    str_var_fps_out = "{:.{}f}".format( var_fps_out, 3 )
    str_avg_fpm_out = "{:.{}f}".format( avg_fpm_out, 3 )
    str_var_fpm_out = "{:.{}f}".format( var_fpm_out, 3 )

    # output params for table 

    #string_params = [str_det_time, str_dyn_var_thr, str_stat_p_thre, str_highpass_freq_cutoff, str_signal_threshold, str_distanceVal, str_total_fracs_out, str_avg_fps_out, str_var_fps_out, str_avg_fpm_out, str_var_fpm_out]
    
    string_params = ''

    # ouput 
        
    return fpm_time, fpm_fracspermin, total_fracs_time, total_fracs_vals, string_params 
        
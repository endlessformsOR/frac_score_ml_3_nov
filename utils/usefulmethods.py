import os, sys, datetime, subprocess, time,db
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import scipy.fftpack as sfp
from numpy.fft import fft, fftfreq, ifft
import pytz
from datetime import datetime, timedelta

#change between live and master here based on what env you want to use

LEGEND_LIVE_DIR = '/home/ubuntu/legend-live'
DATA_RELATIVE_DIR = '../legend-analytics/' #relative to legend-live-repo

# Correct datetime formats

def parse_time_string_with_colon_offset(s):
    "parses timestamp string with colon in UTC offset ex: -6:00"
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    utc = local.astimezone(pytz.utc)
    return utc

def mkDataTimeFromStr(s):
    "same as parse_time_string, but includes utc offset"
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    #utc = local.astimezone(pytz.utc)
    #return utc
    return local

def dtObj2str(dtobj):
    "takes datetimeobj and returns formatted string with UTC offset"
    nst_formatted = dtobj.strftime("%Y-%m-%dT%H:%M:%S%z")
    # add the ":" to the format (couldn't find an easier solution)
    str_time = nst_formatted[:-5]
    str_utc = nst_formatted[-4:]
    str_utcUp = nst_formatted[-4:-2] + ':' + nst_formatted[-2:]
    outputSTR = str_time + '-'+ str_utcUp
    return outputSTR

def quickPressurePlotDataObjs(startTime, endTime, staticID, dynamicID):

	# quickly make np objects to plot static and dynamic pressure response

    start_converted = parse_time_string_with_colon_offset(startTime)
    stop_converted = parse_time_string_with_colon_offset(endTime)
    delta = stop_converted - start_converted
    seconds = int(delta.total_seconds())

    # this method assumes sampling rate is 57kHz, we may need to address this later

    x_d = interval_to_flat_array(dynamicID, start_converted, stop_converted).values()
    t_d = np.linspace(0,len(x_d),len(x_d))
    sampling_rate = int(len(x_d) / seconds)

    # make static data

    stat_obj = fetch_sensor_db_data(staticID,startTime,endTime)

    x_s = stat_obj['max'].to_numpy()
    t_s = np.linspace(0,len(x_s),len(x_s))

    x_d_r, t_d_r = dynamic2dashboard(x_d, seconds, t_d, sampling_rate)

    # Pressure Time Profile ouputs

    return seconds, t_s, x_s, t_d, x_d, t_d_r, x_d_r

def dynamic2dashboard(signal_array, time_in_seconds, time_array,sampling_rate):

    # determine sampling rate

    signal_array_list = signal_array.tolist()

    #sampling_rate = sampling_rate_int-1

    # for each second, find max

    x_list = []
    t_list = []

    for i in range(time_in_seconds):

        start = i*sampling_rate
        end = (i+1)*sampling_rate

        if len(signal_array[start:end]) != 0:

        	maxVal = np.amax(signal_array[start:end],axis=0)

        elif len(signal_array[start:end]) == 0:

        	maxVal = 0

        if i == time_in_seconds-1:

        	if len(signal_array[-sampling_rate:-1]) != 0:

        		maxVal = np.amax(signal_array[-sampling_rate:-1],axis=0)

        	elif len(signal_array[-sampling_rate:-1]) == 0:

        		maxVal = 0

        x_list.append(maxVal)
        t_list.append(i+1)

    x_out = np.asarray(x_list, dtype=np.float32)
    t_out = np.asarray(t_list, dtype=np.float32)

    return x_out, t_out


#Welch matrix method

def Welch_matrix_methods(signal_array, t_f,f_num,sampling_rate, seg_length):

    # find size of array

    x_vals, y_vals = signal.welch(signal_array,sampling_rate,nperseg=seg_length)

    f_vals = len(x_vals)

    #f_vals = int(seg_length/2 + 1)

    spectro_PSD = np.zeros((f_vals,t_f))
    spectro_LS = np.zeros((f_vals,t_f))

    for second in range(t_f):

        start = second*sampling_rate
        end = (second+1)*sampling_rate

        # Welch PSD

        x_psd, y_psd = signal.welch(signal_array[start:end],sampling_rate,nperseg=seg_length)

        y_psd_max = y_psd.max()

        # Welch LS

        x_ls, y_ls = signal.welch(signal_array[start:end],sampling_rate,'flattop',seg_length,scaling='spectrum')


        if second == t_f - 1:

            x_psd, y_psd = signal.welch(signal_array[start:end],sampling_rate,nperseg=seg_length)
            x_ls, y_ls = signal.welch(signal_array[start:end],sampling_rate,'flattop',seg_length,scaling='spectrum')

        spectro_PSD[:,second] = y_psd/y_psd_max
        spectro_LS[:,second] = y_ls

    return spectro_PSD, spectro_LS

#define FFT maker

def quickFFTobjs(xn, sampling_rate):

    # normalized with / max val
    N = len(xn)
    T = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1/(2*T)), int(N/2))
    y_fft = (2.0/N)*np.abs(yf[:N//2]) / max ((2.0/N)*np.abs(yf[:N//2]))
    return x_fft, y_fft

# LogFFT Spectrograph

def logFFT(xn, sampling_rate,f_num):

    # normalized with / max val
    N = len(xn)
    T = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1/(2*T)), int(N/2))

    # normalized version
    #y_fft = (2.0/N)*np.abs(yf[:N//2]) / max ((2.0/N)*np.abs(yf[:N//2]))

    # non nomralized version
    y_fft = (2.0/N)*np.abs(yf[:N//2])

    # make log space of x output

    x_fft_loglog = np.logspace(1,4,f_num)
    y_fft_loglog = []

    # find corresponding x and y val from linear FFT

    for i in x_fft_loglog:

        idx = (np.abs(x_fft-i)).argmin()
        y_fft_loglog.append(y_fft[idx])

    np.asarray(y_fft_loglog,dtype=np.float32)
    np.asarray(x_fft_loglog,dtype=np.float32)

    # return normalized version
    y_fft_log_norm = (np.abs(y_fft_loglog)/max(np.abs(y_fft_loglog)))

    #test, return un nomalized
    #y_fft_log_norm = np.abs(y_fft_loglog)

    return x_fft_loglog, y_fft_log_norm

# LogFFT 2D matrix method

def logFFT2Dmatrix(signal_array, t_f,f_num,sampling_rate):

    spectroImage = np.zeros((f_num,t_f))

    for second in range(t_f):

        start = second*sampling_rate
        end = (second+1)*sampling_rate

        x_fft_log, y_fft_log = logFFT(signal_array[start:end],sampling_rate,f_num)

        if second == t_f - 1:

            x_fft_log, y_fft_log = logFFT(signal_array[start:end],sampling_rate,f_num)

        spectroImage[:,second] = y_fft_log

    return spectroImage

def fetch_sensor_db_data(sensor_id, start,end):
    assert end > start
    db.init()

    static_data = db.query_dataframe("""
    SELECT time,min,max,average,variance
    FROM monitoring.sensor_data
    WHERE sensor_id = %s
    AND time >= %s
    AND time <= %s
    ORDER BY time ASC
    """, (sensor_id, start,end))

    return static_data

def fetch_sensor_db_data_gapfill(sensor_id, start, end):
    assert end > start
    db.init()

    data = db.query_dataframe("""
        SELECT time_bucket_gapfill('1 second', time) as time,
            COALESCE(avg(min), 'NaN') as min,
            COALESCE(avg(max), 'NaN') as max,
            COALESCE(avg(average), 'NaN') as average,
            COALESCE(avg(variance), 'NaN') as variance
        FROM monitoring.sensor_data
        WHERE sensor_id = %s
        AND time > %s
        AND time < %s
        GROUP BY 1
        ORDER BY 1 ASC""", (sensor_id, start, end))

    return data


def download_well_dynamic_data(api, start, end, output_file, env):
    output_path = output_file
    process = subprocess.Popen(['lein', 'audio', api, start, end, output_path, env, 'well-dynamic-data'],
                              cwd=LEGEND_LIVE_DIR,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    #print("stdout:")
    #print(stdout)
    #print("stderr:")
    #print(stderr)

def pressureTimeObject(PATH):

    # Upload .npz, convert to .npy
    dynamic_npys = []
    npy = np.load(PATH)
    dynamic_npys.append(npy['arr_0'])

    dynamic_data = np.concatenate(dynamic_npys)

    # define data conditions

    start = 0
    end = len(dynamic_data)
    xn = dynamic_data[start:end]
    t = range(len(xn))

    return t, xn

def static_sensor_data(sensor_id, start_time, end_time):

    maxVal = db.query_dataframe(f"""SELECT max FROM sensor_data where sensor_id = '{sensor_id}' AND time BETWEEN '{start_time}' AND '{end_time}' ORDER BY time ASC""")
    #sampling_rate = 359.754
    sampling_rate = 1
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    diff = end_time - start_time
    num = diff.total_seconds()
    time_np = np.linspace(0,int(num),int(num*sampling_rate) +1)
    time = pd.DataFrame(data=time_np)

    return time, maxVal

def quickFFTobjs(xn, sampling_rate):

    # normalized with / max val
    N = len(xn)
    T = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1/(2*T)), int(N/2))
    y_fft = (2.0/N)*np.abs(yf[:N//2]) / max ((2.0/N)*np.abs(yf[:N//2]))
    return x_fft, y_fft

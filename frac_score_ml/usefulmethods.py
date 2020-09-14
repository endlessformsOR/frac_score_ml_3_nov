import datetime
import db
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import scipy.fftpack as sfp
import scipy.signal as signal
from dynamic_utils import interval_to_flat_array_resample

# change between live and master here based on what env you want to use

LEGEND_LIVE_DIR = '/home/ubuntu/legend-live'
DATA_RELATIVE_DIR = '../legend-analytics/'  # relative to legend-live-repo
S3_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


# Correct datetime formats

def parse_time_string_with_colon_offset(s):
    """parses timestamp string with colon in UTC offset ex: -6:00"""
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    utc = local.astimezone(pytz.utc)
    return utc


def mk_data_time_from_str(s):
    """same as parse_time_string, but includes utc offset"""
    if ":" == s[-3:-2]:
        s = s[:-3] + s[-2:]
    local = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    return local


def dt_obj_to_str(dt_obj):
    """
    takes date time object and returns formatted string with UTC offset
    """
    nst_formatted = dt_obj.strftime("%Y-%m-%dT%H:%M:%S%z")
    # add the ":" to the format (couldn't find an easier solution)
    str_time = nst_formatted[:-5]
    str_utc_app = nst_formatted[-4:-2] + ':' + nst_formatted[-2:]
    output_str = str_time + '-' + str_utc_app
    return output_str


def dynamic2dashboard(signal_array, time_in_seconds, time_array, sampling_rate):
    # determine sampling rate

    # sampling_rate = sampling_rate_int-1

    # for each second, find max

    x_list = []
    t_list = []

    for i in range(time_in_seconds):

        start = i * sampling_rate
        end = (i + 1) * sampling_rate
        max_value = 0

        if len(signal_array[start:end]) != 0:

            max_value = np.amax(signal_array[start:end], axis=0)

        elif len(signal_array[start:end]) == 0:

            max_value = 0

        if i == time_in_seconds - 1:

            if len(signal_array[-sampling_rate:-1]) != 0:

                max_value = np.amax(signal_array[-sampling_rate:-1], axis=0)

            elif len(signal_array[-sampling_rate:-1]) == 0:

                max_value = 0

        x_list.append(max_value)
        t_list.append(i + 1)

    x_out = np.asarray(x_list, dtype=np.float32)
    t_out = np.asarray(t_list, dtype=np.float32)

    return x_out, t_out


# Welch matrix method

def welch_matrix_methods(signal_array, t_f, f_num, sampling_rate, seg_length):
    # find size of array

    x_values, y_values = signal.welch(signal_array, sampling_rate, nperseg=seg_length)

    f_values = len(x_values)

    # f_values = int(seg_length/2 + 1)

    spectrogram_psd = np.zeros((f_values, t_f))
    spectrogram_ls = np.zeros((f_values, t_f))

    for second in range(t_f):

        start = second * sampling_rate
        end = (second + 1) * sampling_rate

        # Welch PSD

        x_psd, y_psd = signal.welch(signal_array[start:end], sampling_rate, nperseg=seg_length)

        y_psd_max = y_psd.max()

        # Welch LS

        x_ls, y_ls = signal.welch(signal_array[start:end], sampling_rate, 'flattop', seg_length, scaling='spectrum')

        if second == t_f - 1:
            x_psd, y_psd = signal.welch(signal_array[start:end], sampling_rate, nperseg=seg_length)
            x_ls, y_ls = signal.welch(signal_array[start:end], sampling_rate, 'flattop', seg_length, scaling='spectrum')

        spectrogram_psd[:, second] = y_psd / y_psd_max
        spectrogram_ls[:, second] = y_ls

    return spectrogram_psd, spectrogram_ls


# define FFT maker

def quick_fft_objs(xn, sampling_rate):
    # normalized with / max val
    n = len(xn)
    t = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1 / (2 * t)), int(n / 2))
    y_fft = (2.0 / n) * np.abs(yf[:n // 2]) / max((2.0 / n) * np.abs(yf[:n // 2]))
    return x_fft, y_fft


# LogFFT Spectrograph

def log_fft(xn, sampling_rate, f_num):
    # normalized with / max val
    n = len(xn)
    t = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1 / (2 * t)), int(n / 2))

    # normalized version
    # y_fft = (2.0/N)*np.abs(yf[:N//2]) / max ((2.0/N)*np.abs(yf[:N//2]))

    # non normalized version
    y_fft = (2.0 / n) * np.abs(yf[:n // 2])

    # make log space of x output

    x_fft_loglog = np.logspace(1, 4, f_num)
    y_fft_loglog = []

    # find corresponding x and y val from linear FFT

    for i in x_fft_loglog:
        idx = (np.abs(x_fft - i)).argmin()
        y_fft_loglog.append(y_fft[idx])

    np.asarray(y_fft_loglog, dtype=np.float32)
    np.asarray(x_fft_loglog, dtype=np.float32)

    # return normalized version
    y_fft_log_norm = (np.abs(y_fft_loglog) / max(np.abs(y_fft_loglog)))

    # test, return raw, not normalized
    # y_fft_log_norm = np.abs(y_fft_loglog)

    return x_fft_loglog, y_fft_log_norm


# LogFFT 2D matrix method

def log_fft_2d_matrix(signal_array, t_f, f_num, sampling_rate):
    spectrogram_image = np.zeros((f_num, t_f))

    for second in range(t_f):

        start = second * sampling_rate
        end = (second + 1) * sampling_rate

        x_fft_log, y_fft_log = log_fft(signal_array[start:end], sampling_rate, f_num)

        if second == t_f - 1:
            x_fft_log, y_fft_log = log_fft(signal_array[start:end], sampling_rate, f_num)

        spectrogram_image[:, second] = y_fft_log

    return spectrogram_image


def fetch_sensor_db_data(sensor_id, start, end):
    assert end > start
    db.init()

    static_data = db.query_dataframe("""
    SELECT time,min,max,average,variance
    FROM monitoring.sensor_data
    WHERE sensor_id = %s
    AND time >= %s
    AND time <= %s
    ORDER BY time ASC
    """, (sensor_id, start, end))

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


def static_sensor_data(sensor_id, start_time, end_time):
    max_value = db.query_dataframe(
        f"""SELECT max FROM sensor_data where sensor_id = '{sensor_id}' AND time BETWEEN '{start_time}' AND '{end_time}' ORDER BY time ASC""")
    # sampling_rate = 359.754
    sampling_rate = 1
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    diff = end_time - start_time
    num = diff.total_seconds()
    time_np = np.linspace(0, int(num), int(num * sampling_rate) + 1)
    time = pd.DataFrame(data=time_np)

    return time, max_value

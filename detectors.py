import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import scipy.signal as signal
import peakdetect

import db
import analysis

# ### Events to detect
#  * start of pumping down the perf guns and plugs
#  * end of pumping down the perf guns and plugs
#  * firing off perf guns
#  * start of frac stage
#  * detect steep pressurization step in the stage
#  * end of frac stage
#  * pressurizing to create fractures
#  * fractures occurring on active fracking well measuring on active fracking well
#  * fractures occuring on active fracking well measuring on offset well
#  * detecting signs of well communication
#  * detecting communication correlation
#  * Detect gear shift (false positive, looks alot like the "pops" when we are fracking, only occurs early in stage

# ### Goal
#  * produce a fracture report in near real-time
#      - log every detected fracture with a timestamp
#      - characterize their magnitude, count, detect a seam that cracks rapidly in succession
#  * detect early signs of communication using dynamic sensors before we see a static pressure increase in the offset well
#      - produce a communication event/alert so that the operator can decide whether they want to continue or not

def _merge_close_events(events, min_distance):
    '''Merge events that fall within min_distance space.'''
    merged = []
    already_added = False
    for i in range(len(events)-1):
        if already_added:
            already_added = False
            continue

        a = events[i]
        b = events[i+1]
        if (b['start'] - a['end']) < PUMPDOWN_MIN_DISTANCE:
            event = {'start': a['start'], 'end': b['end']}
            merged.append(event)
            already_added = True
        else:
            merged.append(a)
    return pd.DataFrame(merged)


STAGE_MIN_DISTANCE = datetime.timedelta(hours=1)

def detect_stages(static_data):
    smoothing_window_size = 71
    polynomial_order = 3
    x = signal.savgol_filter(static_data, smoothing_window_size, polynomial_order)
    peaks, peak_info = signal.find_peaks(x, prominence=2000, wlen=300)

    events = []
    for start, end in zip(peak_info['left_bases'], peak_info['right_bases']):
        events.append({
            'start': static_data.index[start],
            'end': static_data.index[end],
            'start_idx': start,
            'end_idx': end
            })

    merged = _merge_close_events(events, STAGE_MIN_DISTANCE)
    return pd.DataFrame(merged)


PUMPDOWN_MIN_DISTANCE = datetime.timedelta(hours=1)

def detect_pumpdowns(static_data):
    '''Detect pumpdowns in static pressure data.
    NOTE: expects data sampled at 60 second periods.'''
    peaks, peak_info = signal.find_peaks(static_data,
            prominence=(30, 1000),
            width=(5, 90),
            height=(2000, 6500),
            wlen=90)

    events = []
    for start, end in zip(peak_info['left_bases'], peak_info['right_bases']):
        events.append({
            'start': static_data.index[start],
            'end': static_data.index[end],
            'start_idx': start,
            'end_idx': end
            })

    merged = _merge_close_events(events, PUMPDOWN_MIN_DISTANCE)
    return pd.DataFrame(merged)


def detect_maintenance_breaks(static_data):
    peaks, peak_info = signal.find_peaks((-1*static_data),
                                         prominence=(1000,10000),
                                         width=(3,100),
                                         height=0,
                                         wlen=100
                                        )
    events = []
    for start, end in zip(peak_info['left_bases'], peak_info['right_bases']):
        events.append({
            'start': static_data.index[start],
            'end': static_data.index[end],
            'start_idx': start,
            'end_idx': end
            })
    return pd.DataFrame(events)


def fracture_rate(dynamic_sensor, start_time, end_time, env='live'):
    '''Compute the estimated fractures per second.'''
    n_secs = (end_time - start_time).total_seconds()
    chunk_size = 10 # in seconds

    # Iterate over chunks
    for i in range(0, n_secs-1, chunk_size):
        chunk_start = start_time + datetime.timedelta(seconds=i)
        chunk_end = start_time + datetime.timedelta(seconds=(i+chunk_size))
        chunk_data = db.dynamic_sensor_data(
                dynamic_sensor['id'],
                start_time=chunk_start,
                end_time=chunk_end,
                environment=env)
        sampling_rate = int(chunk_data.shape[0] / float(chunk_size))
        y = analysis.high_pass_filter(chunk_data, 150)
        pos_peaks, neg_peaks = peakdetect.peakdetect(y, lookahead=80, delta=18)
        peaks = pos_peaks + neg_peaks



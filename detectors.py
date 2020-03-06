import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import scipy.signal as signal

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

def setup():
    db.init()

    api = name_to_api("BRISCOE CATARINA", "33HU")[0][0]
    sensor_id = sensor_info(api)[0][0]
    start_time = datetime.datetime(year=2020, month=1, day=7, hour=14, minute=0, second=0).isoformat()
    end_time = datetime.datetime(year=2020, month=1, day=9, hour=24, minute=0, second=0).isoformat()
    static_data = static_sensor_data(sensor_id, start_time, end_time)

def static_sensor_data(name, number):
    api = db.well_name_to_api("BRISCOE CATARINA", "33HU")
    sensor_id = db.sensor_info(api)['id']
    start_time = datetime.datetime(year=2020, month=1, day=3, hour=0, minute=0, second=0).isoformat()
    end_time = datetime.datetime(year=2020, month=1, day=7, hour=0, minute=0, second=0).isoformat()
    static_data = db.static_sensor_data(sensor_id, start_time, end_time, period=60)


def plot_range_events(x, stages):
    plt.figure()
    plt.plot(x)
    for stage in stages:
        plt.axvline(stage['start'], color='green', linestyle=':')
        plt.axvline(stage['end'], color='red', linestyle=':')
    plt.show()


# TODO: convert start/stop to timestamps?
def detect_stages(x):
    smoothing_window_size = 71
    polynomial_order = 3
    x = signal.savgol_filter(x, smoothing_window_size, polynomial_order)
    peaks, peak_info = signal.find_peaks(x,
            prominence=2000,
            wlen=300)

    events = []
    for start, end in zip(peak_info['left_bases'], peak_info['right_bases']):
        events.append({
            'start': start,
            'end': end
            })
    return events


def detect_pumpdown(x):
    peaks, peak_info = signal.find_peaks(x,
            prominence=500,
            #width=(5, 100),
            height=(4000, 6500),
            wlen=50)

    events = []
    for start, end in zip(peak_info['left_bases'], peak_info['right_bases']):
        events.append({
            'start': start,
            'end': end
            })
    return events

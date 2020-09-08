#
#
#     Build a frac pops data set for ML models. Pulls from all pads, wells, during active pumping. Uses all eff. reports as of 9/3/20
#
#

from dynamic_utils import interval_to_flat_array_resample, parse_time_string_with_colon_offset
from usefulmethods import dynamic2dashboard, fetch_sensor_db_data, logFFT, logFFT2Dmatrix, Welch_matrix_methods, quickFFTobjs, parse_time_string_with_colon_offset, mkDataTimeFromStr,dtObj2str 
from impulse_utils import makeData, fracScoreDetector, butter_bandpass, butter_bandpass_filter, butter_lowpass_filter, butter_lowpass, mkFFTstuff, makePerfGunDataset, makeActiveFracsDataset, mkDF_FractureData, mkDF_PerfGunFiring
from fracScore_utils import makeData, fracScoreDetector_v2, fetchSensorIDsFromAPIs, mkUTCtime, mkRndmWindows, make_frac_score_data_set, butter_bandpass_filter
import os, sys, datetime, subprocess, time, db, statistics, ipywidgets
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import scipy.fftpack as sfp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import datetime as dt
from sklearn.preprocessing import normalize
from numpy.fft import fft, fftfreq, ifft
import IPython.display as ipd
from scipy.stats import linregress
from openpyxl import *
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
import pylab
from matplotlib import cm
import random
from datetime import datetime
%matplotlib inline

plt.rcParams.update({'figure.max_open_warning': 0})
base_folder = '/notebooks/'

# Set Working Directory

base_path = f'/home/ubuntu/legend-analytics/' + base_folder
os.chdir(base_path)
if base_path not in sys.path:
    sys.path.append(base_path)
os.chdir(base_path)
cwd = os.getcwd()

LEGEND_LIVE_DIR = '/home/ubuntu/legend-live'
DATA_RELATIVE_DIR = '../legend-analytics/' #relative to legend-live-repo

# create dataframes for all wells (to pull IDs) during all recorded pumping events

df_wellIDs = pd.read_csv('wells3Sept2020.csv')
df_effReport = pd.read_csv('fracPops_effReport_allPads_4Sept2020.csv')

# run through all events in combined efficency report, from all recorded wells

df_wellIDs = pd.read_csv('wells3Sept2020.csv')
df_effReport = pd.read_csv('fracPops_effReport_allPads_4Sept2020.csv')

num_windows_per_stage = 2
counting_window_secs = 15
#STD_M = 6.5
STD_M = 10
#distanceVal = 1000
distanceVal = 4000
timeBeforePop_secs = 2
timeAfterPop_secs = 3


# now make data

make_frac_score_data_set(df_wellIDs, df_effReport, num_windows_per_stage, counting_window_secs, STD_M, distanceVal, timeBeforePop_secs, timeAfterPop_secs)

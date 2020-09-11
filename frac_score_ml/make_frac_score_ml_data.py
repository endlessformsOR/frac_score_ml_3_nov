
## use this to build datasets for ML frac score.

import numpy as np
import pandas as pd
from frac_score_ml_utils import make_frac_score_data_non_events, make_frac_score_data_set

# run through all events in combined efficency report, from all recorded wells

df_wellIDs = pd.read_csv('wells3Sept2020.csv')
df_effReport = pd.read_csv('fracPops_effReport_allPads_4Sept2020.csv')

num_windows_per_stage = 4
counting_window_secs = 15
STD_M = 10
#distanceVal = 1000
distanceVal = 4000
timeBeforePop_secs = 2
timeAfterPop_secs = 3

make_frac_score_data_non_events(df_wellIDs, df_effReport, num_windows_per_stage, counting_window_secs, STD_M, distanceVal, timeBeforePop_secs, timeAfterPop_secs)

# run through all events in combined efficency report, from all recorded wells

num_windows_per_stage = 2
counting_window_secs = 15
STD_M = 10
#distanceVal = 1000
distanceVal = 4000
timeBeforePop_secs = 2
timeAfterPop_secs = 3


# now make data

make_frac_score_data_set(df_wellIDs, df_effReport, num_windows_per_stage, counting_window_secs, STD_M, distanceVal, timeBeforePop_secs, timeAfterPop_secs)

# use this to build datasets for ML frac score.

import pandas as pd
from ml_utils import make_frac_score_data_non_events, make_frac_score_data_set


# run through all events in combined efficiency report, for all recorded wells

df_well_ids = pd.read_csv('wells3Sept2020.csv')
df_eff_report = pd.read_csv('fracPops_effReport_allPads_4Sept2020.csv')

num_windows_per_stage = 2
counting_window_secs = 15
distance_val = 4000
std_m = 10
time_before_pop_secs = 2
time_after_pop_secs = 3

# now make data

make_frac_score_data_set(df_well_ids,
                         df_eff_report,
                         num_windows_per_stage,
                         counting_window_secs,
                         std_m, distance_val,
                         time_before_pop_secs,
                         time_after_pop_secs)


# make nonevents

num_windows_per_stage = 4
make_frac_score_data_non_events(df_well_ids,
                                df_eff_report,
                                num_windows_per_stage,
                                counting_window_secs,
                                std_m, distance_val,
                                time_before_pop_secs,
                                time_after_pop_secs)

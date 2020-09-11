import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from numpy import dstack
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from frac_score_ml_utils import load_ML_data_pops, dynamic_model, evaluate_model

# Look for these .npy fils. 5 second windows at 40k samples per sec.

POPS_FILENAME = 'fracScore_allPads_08_09_pop_events.npy'
NON_POPS_FILENAME = 'fracScore_allPads_08_09_not_pops.npy'
WINDOW_SIZE_SECS = 5
TRAIN_TEST_SPLIT_PERCENT = 70

# build train and test datasets

training_vals, testing_vals, training_labels, testing_labels = load_ML_data_pops(POPS_FILENAME,NON_POPS_FILENAME,WINDOW_SIZE_SECS,TRAIN_TEST_SPLIT_PERCENT)

print('number of training events: ' + str(len(training_vals)))
print('number of testing events: ' + str(len(testing_vals)))
print('shape of training vals: ' + str(np.shape(training_vals)))
print('shape of testing vals: ' + str(np.shape(testing_vals)))
print('shape of training labels: ' + str(np.shape(training_labels)))
print('shape of testing labels: ' + str(np.shape(testing_labels)))

# build 1D convo and evaluate

dyn_window_size = 5*40000

print('updated dyn window size: ' +  str(dyn_window_size))


n_outputs_dyn = 1

d_m = dynamic_model(n_outputs_dyn,dyn_window_size)

# print model summary

print(d_m.summary())

# evaluate model

d_acc = evaluate_model(d_m,training_vals,training_labels,testing_vals,testing_labels)

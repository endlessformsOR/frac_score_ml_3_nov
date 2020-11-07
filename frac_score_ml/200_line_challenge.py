# demonstrate resnet model on data in 200 lines

import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
import matplotlib.image as img
import tensorflow as tf
from ml_utils import *
from scipy import signal

# specify testing training sizes

number_pops = 5000
number_non_events = 5000
window_of_interest_ms = 20
test_train_ratio = 70

def grab_raw_data():

    x_test = []
    x_train = []
    y_test = []
    y_train = []

def make_testing_training_data(window_of_interest_ms, number_pops, number_non_events, test_train_ratio):

    x_test, x_train, y_test, y_train = grab_raw_data(window_of_interest_ms, number_pops, number_non_events)

    print('make testing/training data...')

    return x_test, x_train, y_test, y_train

def make_ml_model(model_type, input_size):

    print('making model: ' + str(model_type) + ' , for input size: ' + str(input_size))

def show_model_results(model_type):

    print('showing model results for: ' + str(model_type))

x_test, x_train, y_test, y_train = make_testing_training_data()

# make model

input_size = len(x_train)

mdl_resnet = make_ml_model('resnet', input_size)

# show model

#mdl_resnet.summary()

# show testing/training results

show_model_results(mdl_resnet)

# show test harness results

print('complete.')
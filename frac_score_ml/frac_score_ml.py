import numpy as np

from ml_utils import load_ml_data_pops, dynamic_model, evaluate_model

# Look for these .npy files. 5 second windows at 40k samples per sec.

pops_file_name = 'fracScore_allPads_08_09_pop_events.npy'
non_pops_file_name = 'fracScore_allPads_08_09_not_pops.npy'
window_size_seconds = 5
train_test_split_percent = 70

# build train and test datasets

training_values, testing_values, training_labels, testing_labels = load_ml_data_pops(pops_file_name,
                                                                                     non_pops_file_name,
                                                                                     window_size_seconds,
                                                                                     train_test_split_percent)

print('number of training events: ' + str(len(training_values)))
print('number of testing events: ' + str(len(testing_values)))
print('shape of training values: ' + str(np.shape(training_values)))
print('shape of testing values: ' + str(np.shape(testing_values)))
print('shape of training labels: ' + str(np.shape(training_labels)))
print('shape of testing labels: ' + str(np.shape(testing_labels)))

# build 1D convolution and evaluate

dyn_window_size = 5 * 40000

print('updated dyn window size: ' + str(dyn_window_size))

n_outputs_dyn = 1

d_m = dynamic_model(n_outputs_dyn, dyn_window_size)

# print model summary

print(d_m.summary())

# evaluate model

d_acc = evaluate_model(d_m,
                       training_values,
                       training_labels,
                       testing_values,
                       testing_labels)

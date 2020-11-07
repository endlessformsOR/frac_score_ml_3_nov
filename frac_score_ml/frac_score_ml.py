import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from tensorflow.keras import layers
from ml_utils import load_ml_data_pops, dynamic_model, evaluate_model, get_train_test_2d_cnn

import os
path = os.getcwd()
print('path is: ' + path)

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
# Look for these .npy files. 5 second windows at 40k samples per sec.

pops_file_name = 'raw_ml_data/fracScore_allPads_08_09_pop_events.npy'
non_pops_file_name = 'raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy'
window_size_seconds = 5
train_test_split_percent = 70

# # build train and test datasets
#
# training_values, testing_values, training_labels, testing_labels = load_ml_data_pops(pops_file_name,
#                                                                                      non_pops_file_name,
#                                                                                      window_size_seconds,
#                                                                                      train_test_split_percent)


# build 2d convolution testing training datasets

x_train, x_test, y_train, y_test = get_train_test_2d_cnn(pops_file_name,
                                                         non_pops_file_name,
                                                         window_size_seconds,
                                                         train_test_split_percent)


print('shape of training values: ' + str(np.shape(x_train)))
print('shape of testing values: ' + str(np.shape(x_test)))
print('shape of training labels: ' + str(np.shape(y_train)))
print('shape of testing labels: ' + str(np.shape(y_test)))

channels = 1
max_len = 11
buckets = 20
epoch = 50
batch_size = 3
num_classes = 2
callback = ''

x_window_size = 300 # 3 inch plot converted to 300 px
y_window_size = 150 # 1.5 inch converted to 150 px

# pseudocode - turn labels into one hots ?

def m_2dcnn():

    m = Sequential()
    m.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 input_shape=(y_window_size, x_window_size, channels),
                 activation='relu'))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())

    # add another 2d conv and pooling once this is working
    m.add(Dense(128, activation='relu'))
    m.add(Dense(num_classes, activation='softmax'))

    m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    m.summary()

    print('shape of x_train: ' + str(np.shape(x_train)))
    print('shape of x_test: ' + str(np.shape(x_test)))
    print('shape of y_train: ' + str(np.shape(y_train)))
    print('shape of y_test: ' + str(np.shape(y_test)))

    m.fit(x=x_train,
          y=y_train,
          batch_size=10,
          epochs=50,
          verbose=2,
          validation_data=(x_test, y_test),
          callbacks=['history'])

# see what happens...

m_2dcnn()







# # build 1D convolution and evaluate
#
# dyn_window_size = 5 * 40000
#
# print('updated dyn window size: ' + str(dyn_window_size))
#
# n_outputs_dyn = 1
#
# filters = [32, 16, 8, 4]
#
# def dynamic_model_2(n_outputs, window_size,filters):
#     # static model
#     m = Sequential()
#     m.add(layers.InputLayer(input_shape=(window_size, 1)))
#     m.add(layers.BatchNormalization(name='batch_norm_1'))
#     # new, added padding='same' instead of default
#     # m.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='conv_1', padding='same'))
#     m.add(layers.Conv1D(filters=filters[0], kernel_size=3, activation='relu', name='conv_1'))
#     m.add(layers.BatchNormalization(name='batch_norm_2'))
#     m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_1'))
#     m.add(layers.Conv1D(filters=filters[1], kernel_size=3, activation='relu', name='conv_2'))
#     m.add(layers.BatchNormalization(name='batch_norm_3'))
#     m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_2'))
#     m.add(layers.Conv1D(filters=filters[2], kernel_size=3, activation='relu', name='conv_3'))
#     m.add(layers.BatchNormalization(name='batch_norm_4'))
#     m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_3'))
#     m.add(layers.Conv1D(filters=filters[3], kernel_size=5, activation='relu', name='conv_4'))
#     m.add(layers.BatchNormalization(name='batch_norm_5'))
#     m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_4'))
#     m.add(layers.Flatten(name='flatten'))
#     m.add(layers.Dropout(0.4))
#     # m.add(layers.Dense(128, activation='relu', name='dense_1'))
#     m.add(layers.Dense(32, activation='relu', name='dense_1'))
#
#     m.add(layers.Dense(n_outputs, activation='softmax', name='output'))
#
#     return m
#
#
#
# d_m = dynamic_model_2(n_outputs_dyn, dyn_window_size,filters)
#
# # print model summary
#
# print(d_m.summary())
#
# # evaluate model
#
# d_acc = evaluate_model(d_m,
#                        training_values,
#                        training_labels,
#                        testing_values,
#                        testing_labels)

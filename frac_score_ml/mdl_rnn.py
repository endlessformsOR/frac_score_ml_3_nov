"""
Recurrent NN Model - Use this as a template. parameters not yet optimized.

"""
import numpy as np
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import keras
from ml_utils import load_ml_data_pops, dynamic_model, evaluate_model, get_train_test_2d_cnn
import os
import time
import matplotlib.pyplot as plt

# measure time with and without GPU

start_time = time.time()

path = os.getcwd()
print('path is: ' + path)

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
# Look for these .npy files. 5 second windows at 40k samples per sec.

# all params

## this is for a 20ms dynamic array

model_name = 'rnn_20msec' + '_v2'

# 9/24 - New dataset, higher std_m

# std_m = 12
# open file 5618 pops, 875 MB
# need to adjust, 4343 nonevents, filesize is 676 MB

pops_file_name = 'raw_ml_data/millisecond_data_pops_std12.npy'
non_pops_file_name = 'raw_ml_data/millisecond_data_not_pops_std12.npy'

window_size_seconds = 0.02
train_test_split_percent = 70
class_labels = ['pops', 'not_pops']
num_classes = len(class_labels)
epoch = 25
batch_size = 24
dyn_window_size = int(window_size_seconds * 40000)
n_outputs_dyn = 2

# build train and test datasets

x_train, x_test, y_train, y_test = load_ml_data_pops(pops_file_name,
                                                     non_pops_file_name,
                                                     window_size_seconds,
                                                     train_test_split_percent)

print('shape of training values: ' + str(np.shape(x_train)))
print('shape of testing values: ' + str(np.shape(x_test)))
print('shape of training labels: ' + str(np.shape(y_train)))
print('shape of testing labels: ' + str(np.shape(y_test)))

# build 1D RNN and evaluate

print('building recurrent neural network model...')


def rnn_v2(n_outputs, window_size):
    m = Sequential()

    m.add(layers.InputLayer(input_shape=(window_size, 1)))
    m.add(layers.LSTM(units=50,
                      return_sequences=True))
    m.add(layers.Dropout(0.2))
    m.add(layers.LSTM(units=50,
                      return_sequences=True))
    m.add(layers.Dropout(0.2))
    m.add(layers.LSTM(units=50))
    m.add(layers.Dropout(0.2))

    m.add(layers.Dense(n_outputs, activation='sigmoid', name='output'))

    return m


def rnn(n_outputs, window_size):
    # 1-layer, basic model, doesn't work very well...

    m = Sequential()
    m.add(layers.InputLayer(input_shape=(window_size, 1)))

    m.add(layers.SimpleRNN(units=64,
                           activation='tanh',
                           use_bias=True,
                           kernel_initializer='glorot_uniform',
                           recurrent_initializer='orthogonal',
                           bias_initializer='zeros',
                           kernel_regularizer=None,
                           recurrent_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           recurrent_constraint=None,
                           bias_constraint=None,
                           dropout=0.5,
                           recurrent_dropout=0.2,
                           return_sequences=False,
                           return_state=False,
                           go_backwards=False,
                           stateful=False,
                           unroll=False
                           ))
    m.add(layers.BatchNormalization(name='batch_norm_1'))

    # last layer

    m.add(layers.Dense(n_outputs, activation='sigmoid', name='output'))

    return m


model = rnn_v2(n_outputs_dyn, dyn_window_size)

# look at model

model.summary()

# compile and train the model

print('compiling model...')


# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# def root_mean_squared_error(y_true, y_pred):
#     return tf.math.sqrt(tf.keras.backend.mean(tf.math.square(y_pred - y_true)))
#
# model.compile(optimizer='Adam',
#               loss=root_mean_squared_error,
#               metrics=['accuracy'])

model.compile(optimizer='Adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print('compilation complete, displaying history and plotting accuracy per epoch...')

history = model.fit(x_train,
                    y_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))

# plot acc per epoch

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Frac Score: ' + model_name + ' Adam optimizer ')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('2d cnn model successfully built. Acc: ' + str(test_acc))

model.save(model_name + '.h5')

print('model saved')

print('total exec. time: ' + str(time.time() - start_time))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from ml_utils import get_train_test_2d_cnn, m_2d_cnn_tf
import time
import os
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

"""
Use this to build a 2d CNN model from dynamic pressure data
"""

# check to see if GPU is working

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# clock time

start_time = time.time()

# all params
model_name = '2dcnn_20msec'
class_labels = ['pops', 'not_pops']
num_classes = len(class_labels)
channels = 1
f_min = 1
f_max = 5000
dpi = 100
fig_size_x_in = 6
fig_size_y_in = 3
epoch = 50
batch_size = 6
x_window_size = dpi * fig_size_x_in  # to pixels
y_window_size = dpi * fig_size_y_in  # to pixels

# Look for these .npy files. 5 second windows at 40k samples per sec.
#pops_file_name = 'raw_ml_data/fracScore_allPads_08_09_pop_events.npy'
#non_pops_file_name = 'raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy'

## this is for a 20ms dynamic array

model_name = '2dcnn_20msec'
pops_file_name = 'raw_ml_data/milisecond_data_pops.npy'
non_pops_file_name = 'raw_ml_data/milisecond_data_not_pops.npy'

window_size_seconds = 0.02
train_test_split_percent = 70

# build 2d convolution testing training datasets

x_train, x_test, y_train, y_test = get_train_test_2d_cnn(pops_file_name,
                                                         non_pops_file_name,
                                                         window_size_seconds,
                                                         train_test_split_percent,
                                                         fig_size_x_in,
                                                         fig_size_y_in,
                                                         f_min,
                                                         f_max)

# print shapes of testing and training datasets

print('shape of training values: ' + str(np.shape(x_train)))
print('shape of testing values: ' + str(np.shape(x_test)))
print('shape of training labels: ' + str(np.shape(y_train)))
print('shape of testing labels: ' + str(np.shape(y_test)))

# make model

model = m_2d_cnn_tf(x_window_size, y_window_size, channels)

# look at model

model.summary()

# compile and train the model

print('compiling model...')

model.compile(optimizer='Adagrad',
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
plt.title('Frac Score: ' + model_name + ' Adagrad optimizer ')

#plt.plot(history.history['val_accuracy'], label='val_accuracy')
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

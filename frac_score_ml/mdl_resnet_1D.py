"""
1D ResNet ML - Use this as a template. parameters not yet optimized.
# see https://arxiv.org/pdf/1611.06455.pdf for model design

"""
import numpy as np
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import keras
from ml_utils import load_1d_data, dynamic_model, evaluate_model, resnet_1d_model
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

model_name = '1dresNet_20msec_nov6'
data_file_name = 'raw_ml_data/frac_score_ml_dataset.npy'

window_size_msecs = 20
train_test_split_percent = 70
class_labels = ['pops', 'not_pops']
num_classes = len(class_labels)
epoch = 50
batch_size = 50
dyn_window_size = int((window_size_msecs/1000) * 40000)
n_outputs_dyn = 2

# build train and test datasets

x_train, x_test, y_train, y_test = load_1d_data(data_file_name, train_test_split_percent, window_size_msecs)

print('shape of training values: ' + str(np.shape(x_train)))
print('shape of testing values: ' + str(np.shape(x_test)))
print('shape of training labels: ' + str(np.shape(y_train)))
print('shape of testing labels: ' + str(np.shape(y_test)))

# build 1D convolution and evaluate

print('building resnet model...')

model = resnet_1d_model(n_outputs_dyn, dyn_window_size)

# look at model

model.summary()

# compile and train the model

print('compiling model...')

optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# Adagrad is best for 1d convo
# Maaaybe Adamax for 1D resnet?

for opt in optimizers:

    model.compile(optimizer=opt,
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
    plt.title('Frac Score: ' + model_name + ' , optimizer: ' + opt)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    print('2d cnn model successfully built. Acc: ' + str(test_acc))

# model.save(model_name + '.h5')
#
# print('model saved')

print('total exec. time: ' + str(time.time() - start_time))

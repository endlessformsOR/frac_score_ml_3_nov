"""
ResNet Model, K.He, X.Zhang, S.Ren - Adapted from Jeff Heaton's GWU course

9/27 - Initial build. Use this as a template. parameters not yet optimized to dynamic pressure signal.
10/8 - Revised spectro image functions used to train/test model. got 85% val_acc and 95% acc
10/11 - Revised testing/training dataset. Better results
10/17 - used gradient, standard_scalar to pick much better pops
10/28 - Updated spectro, outputs as matrix, looks a lot better
11/5 - Updated dataset, nonevents collected during pumping, offset pops in window, new data loader
"""
import numpy as np
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import keras
from ml_utils import load_ml_data_pops, dynamic_model, evaluate_model, load_2d_data
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint

path = os.getcwd()
print('path is: ' + path)

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
# Look for these .npy files. 5 second windows at 40k samples per sec.

# all params

## this is for a 20ms dynamic array

model_name = 'resNet_20_msec_5Nov2020'


window_size_seconds = 0.02
train_test_split_percent = 70
class_labels = ['pops', 'not_pops']
num_classes = len(class_labels)
dyn_window_size = int(window_size_seconds * 40000)
n_outputs_dyn = 2

channels = 1
f_min = 1
f_max = 1000
dpi = 100
fig_size_x_in = 8
fig_size_y_in = 3
x_window_size = dpi * fig_size_x_in  # to pixels
y_window_size = dpi * fig_size_y_in  # to pixels

# ResNet params

version = 1
epoch = 25
batch_size = 25
USE_AUGMENTATION = False
colors = 1

# resnet model version

if version == 1:
    depth = colors * 6 + 2
elif version == 2:
    depth = colors * 9 + 2

# measure time

start_time = time.time()

# build 2d testing training datasets
data_file_name = 'raw_ml_data/frac_score_ml_dataset.npy'
window_size_msecs = 20
x_train, x_test, y_train, y_test = load_2d_data(data_file_name, train_test_split_percent, window_size_msecs)

# # limit number of events, uncomment to see all 7691+ events
#
# number_of_events = 1000
#
# x_train = x_train[0:number_of_events]
# x_test = x_test[0:number_of_events]
# y_train = y_train[0:number_of_events]
# y_test = y_test[0:number_of_events]

# print shapes of testing and training datasets

print('shape of training values: ' + str(np.shape(x_train)))
print('shape of testing values: ' + str(np.shape(x_test)))
print('shape of training labels: ' + str(np.shape(y_train)))
print('shape of testing labels: ' + str(np.shape(y_test)))

input_shape = x_train.shape[1:]


# Following are defs from Heaton

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = layers.Conv2D(num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature
    map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of
    filters is
    doubled. Within each stage, the layers have the same number
    filters and the same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = layers.Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            # first layer but not first stack
            if stack > 0 and res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])
            x = layers.Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is
    halved (downsampled) by a convolutional layer with
    strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same
    number filters and the same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = layers.Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.AveragePooling2D(pool_size=8)(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# run ResNet

# Create the neural network
if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

# look at model

model.summary()

# let it rip

print('compiling model...')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks)

model.save(model_name + '.h5')

print('model saved')

print('total exec. time: ' + str(time.time() - start_time))

# plot acc per epoch


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Frac Score: ' + model_name + ' Adam optimizer ')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

#
#
#
# # compile and train the model
#
# print('compiling model...')
#
#
# # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#
# # def root_mean_squared_error(y_true, y_pred):
# #     return tf.math.sqrt(tf.keras.backend.mean(tf.math.square(y_pred - y_true)))
# #
# # model.compile(optimizer='Adam',
# #               loss=root_mean_squared_error,
# #               metrics=['accuracy'])
#
# model.compile(optimizer='Adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# print('compilation complete, displaying history and plotting accuracy per epoch...')
#
# history = model.fit(x_train,
#                     y_train,
#                     epochs=epoch,
#                     batch_size=batch_size,
#                     validation_data=(x_test, y_test))
#
# # plot acc per epoch
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.title('Frac Score: ' + model_name + ' Adam optimizer ')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()
#
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
#
# print('2d cnn model successfully built. Acc: ' + str(test_acc))
#
# model.save(model_name + '.h5')
#
# print('model saved')
#
# print('total exec. time: ' + str(time.time() - start_time))

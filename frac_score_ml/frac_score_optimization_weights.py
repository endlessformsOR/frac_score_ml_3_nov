import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from ml_utils import get_train_test_2d_cnn, m_2d_cnn_tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import gc

"""
Scikit learn optimization: use accuracy to explore optimal number of epochs and batch_sizes via grid search


"""

# address out of memory issue
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# load datasets

pops_file_name = 'raw_ml_data/fracScore_allPads_08_09_pop_events.npy'
non_pops_file_name = 'raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy'
window_size_seconds = 5
train_test_split_percent = 90
f_min = 1
f_max = 1000
dpi = 100
fig_size_x_in = 4
fig_size_y_in = 2
x_window_size = dpi * fig_size_x_in  # to pixels
y_window_size = dpi * fig_size_y_in  # to pixels
channels = 1

# build 2d convolution testing training datasets

x_train, x_test, y_train, y_test = get_train_test_2d_cnn(pops_file_name,
                                                         non_pops_file_name,
                                                         window_size_seconds,
                                                         train_test_split_percent,
                                                         fig_size_x_in,
                                                         fig_size_y_in,
                                                         f_min,
                                                         f_max)

# NOTE, WE ARE UPDATING THIS MODEL TO HAVE 2 CONVOLUTIONS

def create_model(init_mode='uniform'):
    """
    This is the model in the Tensorflow docs. Use this as a template.

    Args:
        x_window_size: window size in elements
        y_window_size: window size in elements
        channels:

    Returns: 2D CNN model

    """
    print('building model...')

    mdl = models.Sequential()
    mdl.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_window_size, y_window_size, channels)))
    mdl.add(layers.MaxPooling2D((2, 2)))
    mdl.add(layers.Conv2D(64, (3, 3), activation='relu'))
    mdl.add(layers.MaxPooling2D((2, 2)))
    mdl.add(layers.Conv2D(64, (3, 3), activation='relu'))
    mdl.add(layers.Flatten())
    mdl.add(layers.Dense(64,
                           kernel_initializer=init_mode,
                           activation='relu'))
    mdl.add(layers.Dense(2,
                           kernel_initializer=init_mode))  # modify output classes to 2

    mdl.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    return mdl

#

X = x_train
Y = y_train
seed = 7
np.random.seed(seed)
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=5, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
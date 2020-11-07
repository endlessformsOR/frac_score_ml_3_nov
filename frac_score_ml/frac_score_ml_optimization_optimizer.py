import numpy as np
import tensorflow as tf
from ml_utils import get_train_test_2d_cnn, m_2d_cnn_tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
import gc

"""
Scikit learn optimization: use acc to find the best optimizer via grid search
"""
# load datasets

pops_file_name = 'raw_ml_data/fracScore_allPads_08_09_pop_events.npy'
non_pops_file_name = 'raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy'
window_size_seconds = 5
train_test_split_percent = 90
f_min = 1
f_max = 1000
dpi = 100
fig_size_x_in = 6
fig_size_y_in = 3
x_window_size = dpi * fig_size_x_in  # to pixels
y_window_size = dpi * fig_size_y_in  # to pixels
channels = 1


# build model and compiling structure

def make_2d_cnn_model(x=x_window_size, y=y_window_size, optimizer='adam'):
    channels = 1

    m = m_2d_cnn_tf(x, y, channels)
    # Compile model
    m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return m


# fix random seed for reproducibility

seed = 7
np.random.seed(seed)

# build 2d convolution testing training datasets

x_train, x_test, y_train, y_test = get_train_test_2d_cnn(pops_file_name,
                                                         non_pops_file_name,
                                                         window_size_seconds,
                                                         train_test_split_percent,
                                                         fig_size_x_in,
                                                         fig_size_y_in,
                                                         f_min,
                                                         f_max)

# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
epoch = 10  # determined by optimization
batch_size = 5  # determined by optimization
output_list = []

for i in range(len(optimizer)):

    m = m_2d_cnn_tf(x_window_size, y_window_size, channels)

    # having memory issues, see if clearing memory helps

    gc.collect()

    m.compile(optimizer=optimizer[i],
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = m.fit(x_train,
                    y_train,
                    epochs=epoch,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test))

    loss, acc = m.evaluate(x_test, y_test, verbose=2)

    txt_out = 'optimizer: ' + optimizer[i] + ', acc: ' + str(acc) + ' loss: ' + str(loss)
    print(txt_out)
    output_list.append(txt_out)

# look at results

print('summary of grid search:')

for z in range(len(output_list)):
    print(output_list[z])


# SCIKIT LEARN CODE BELOW
#
# # feeding data into scikit learn format
#
# X = x_train
# Y = y_train
#
# # create model
#
# model = KerasClassifier(build_fn=make_2d_cnn_model, epochs=20, batch_size=10, verbose=0)
#
# # define the grid search parameters
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#
# param_grid = dict(optimizer=optimizer)
#
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#
# grid_result = grid.fit(X, Y)
#
# # summarize results
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

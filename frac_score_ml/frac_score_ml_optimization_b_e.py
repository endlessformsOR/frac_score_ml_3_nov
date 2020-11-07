import numpy as np
import tensorflow as tf
from ml_utils import get_train_test_2d_cnn, m_2d_cnn_tf
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import gc
import matplotlib.pyplot as plt

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
train_test_split_percent = 70
channels = 1

# f_min = 1
# f_max = 1000
# dpi = 100
# fig_size_x_in = 4
# fig_size_y_in = 2
# max acc: .8086 with batch 6, epoch 9

f_min = 1
f_max = 1200
dpi = 100
fig_size_x_in = 5
fig_size_y_in = 3

x_window_size = int(dpi * fig_size_x_in)  # to pixels
y_window_size = int(dpi * fig_size_y_in)  # to pixels

# build 2d convolution testing training datasets

x_train, x_test, y_train, y_test = get_train_test_2d_cnn(pops_file_name,
                                                         non_pops_file_name,
                                                         window_size_seconds,
                                                         train_test_split_percent,
                                                         fig_size_x_in,
                                                         fig_size_y_in,
                                                         f_min,
                                                         f_max)

# look at model

# m.summary()

# nested loop with epoch and batch grid search
print('start manual grid search...')
# batch_size = [5, 10, 25] # max batch for my memory is 150

#batch_size = [4, 6, 10]  # max batch for my memory is 150
#epochs = [5, 10, 15]

batch_size = [2,3,4,5,6,7,8,9,10]  # max batch for my memory is 150
epochs = [2,3,4,5,6,7,8,9,10]

output_list = []
plot_object = []

for i in range(len(epochs)):

    for j in range(len(batch_size)):
        print('starting batch: ' + str(batch_size[j]) + ' , epoch: ' + str(epochs[i]))

        # build model

        m = m_2d_cnn_tf(x_window_size, y_window_size, channels)

        # having memory issues, see if clearing memory helps
        gc.collect()

        m.compile(optimizer='Adagrad',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        history = m.fit(x_train,
                        y_train,
                        epochs=epochs[i],
                        batch_size=batch_size[j],
                        validation_data=(x_test, y_test))

        loss, acc = m.evaluate(x_test, y_test, verbose=2)

        txt_out = 'batch: ' + str(batch_size[j]) + ' , epoch: ' + str(epochs[i]) + ', acc: ' + str(
            acc) + ' , loss: ' + str(loss)
        print(txt_out)
        output_list.append(txt_out)
        plot_object.append([batch_size[j], epochs[i], acc, loss])

# look at results

print('summary of grid search:')

for z in range(len(output_list)):
    print(output_list[z])

#save .py file to plot

np.save('batch_epoch_space.npy', plot_object)

print('object saved as .npy file. ')

# def surface_plot(X, Y, Z, **kwargs):
#     """ Take x,y,z lists and make surface plot
#     """
#     xlabel, ylabel, zlabel, title = kwargs.get('xlabel', ""), kwargs.get('ylabel', ""), kwargs.get('zlabel',
#                                                                                                    ""), kwargs.get(
#         'title', "")
#     fig = plt.figure()
#     fig.patch.set_facecolor('white')
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X, Y, Z)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#     ax.set_title(title)
#     plt.show()

# plot output

# SCIKIT LEARN METHOD

# # build model and compiling structure
#
# def make_2d_cnn_model(x=x_window_size, y=y_window_size):
#     channels = 1
#
#     m = m_2d_cnn_tf(x, y, channels)
#     # Compile model
#     m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return m
#
#
# # fix random seed for reproducibility
#
# seed = 7
# np.random.seed(seed)
#
# # feeding data into scikit learn format
#
# X = x_train
# Y = y_train
#
# # create model
# model = KerasClassifier(build_fn=make_2d_cnn_model, verbose=0)
#
# # define the grid search parameters
# batch_size = [5, 10, 15, 20, 25, 30]
# epochs = [5, 10, 20]
#
# param_grid = dict(batch_size=batch_size, epochs=epochs)
#
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#
# grid_result = grid.fit(X, Y)
#
# # summarize results
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

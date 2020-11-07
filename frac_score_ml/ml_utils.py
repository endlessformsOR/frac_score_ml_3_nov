import os
import numpy as np
from tensorflow.keras import datasets, layers, models
#from tensorflow.models import Sequential
from datetime import datetime
from dynamic_utils import interval_to_flat_array_resample, parse_time_string_with_colon_offset
from usefulmethods import fetch_sensor_db_data, parse_time_string_with_colon_offset, \
    mk_data_time_from_str, dt_obj_to_str
from datetime import datetime
import random
import scipy.signal as signal
from scipy.signal import butter, sosfilt, lfilter
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.image as img
import pylab
import scipy.fftpack as sfp
from skimage.transform import rescale

"""
Use this file to build ml training and testing datasets from all pads and wells with efficiency reports. 
"""


def fft_arrays(xn, sampling_rate):
    """
    Convert a raw signal into an fft array
    Args:
        xn: Input signal
        sampling_rate: Sampling Frequency, 40kHz for us

    Returns:

        x_fft = An array of equally spaced frequencies
        y_fft = An array of equally spaced, normalized powers for associated frequencies from fft

    """
    # normalized with / max val

    N = len(xn)
    T = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1 / (2 * T)), int(N / 2))

    # normalize fft

    y_fft = (2.0 / N) * np.abs(yf[:N // 2]) / max((2.0 / N) * np.abs(yf[:N // 2]))

    # do not normalize

    # y_fft = (2.0/N)*np.abs(yf[:N//2])

    return x_fft, y_fft


def make_fft_matrix(dynamic_values, sampling_rate, num_partitions):
    """
    Create a FFT matrix from a window of dynamic data
    This is used mostly for 20ms dynamic pressure data

    Args:
        dynamic_values: input window
        sampling_rate: 40 kHz in our case
        num_partitions: we are choosing 40, since we're dealing with 20 ms

    Returns:

        fft_matrix: spectrogram-esque matrix of signal

    """
    window_size = int(len(dynamic_values) / num_partitions)
    fft_array = []

    for i in range(num_partitions):

        dyn_window = dynamic_values[i * window_size:(i + 1) * window_size]
        x_fft, y_fft = fft_arrays(dyn_window, sampling_rate)
        z_fft = x_fft * y_fft

        fft_array.append(z_fft)

    # make list of arrays into 2D matrix

    A = np.array(fft_array)
    fft_matrix = np.rot90(A, 3)

    fft_matrix = np.flip(fft_matrix, axis=1)

    # upscale to larger array by a factor of 50. This is for ResNet, may be a bad idea...
    fft_matrix = rescale(fft_matrix, 10, anti_aliasing=False)

    return fft_matrix


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def standard_scalar(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std


def fracScoreDetector_gss(dynamic_values, sampling_rate):
    """
    New method to detect frac pops. Since basis pops look like sudden jumps in the dynamic signal,
    we are essentially counting the number of non-smooth events during pumping.
    We use the standard scalar method as a way to quickly normalize the range of the dynamic signal

    """

    standard_scalar_raw = standard_scalar(dynamic_values)
    gradient_standard_scalar = np.gradient(standard_scalar_raw)

    abs_gss = np.absolute(gradient_standard_scalar)

    snr_abs_gss = signaltonoise(abs_gss)
    mean_gss = np.mean(abs_gss)
    max_gss = np.nanmax(abs_gss)
    std_gss = np.std(abs_gss)

    # default cutoff if things go bad

    gss_cutoff = 150 * std_gss

    if max_gss < 0.6:

        # this is probably an execellent timeseries, should be easy to pick vals

        gss_cutoff = 28 * std_gss

        if gss_cutoff > max_gss:
            gss_cutoff = 0.95 * max_gss

    if max_gss > 0.6 and max_gss < 1.00:

        # noisy data, still has relevant pops, but need to adjust cutoff higher

        gss_cutoff = 55 * std_gss

        if gss_cutoff > max_gss:
            gss_cutoff = 0.95 * max_gss

    if max_gss > 1.0 and max_gss < 1.35:

        # very noisy data, still has relevant pops, but need to adjust cutoff higher

        gss_cutoff = 75 * std_gss

        if gss_cutoff > max_gss:
            gss_cutoff = 0.95 * max_gss

    # if pops occur within a 2 - 5 ms window, we can assume they do not overlap at this scale..

    distance_pops_elements = sampling_rate * (5 * 0.001)

    # now run peak detect to look for these large jumps in the velocity space

    peaks_gss, _dict_gss = signal.find_peaks(abs_gss,
                                             height=gss_cutoff,
                                             distance=distance_pops_elements)

    return peaks_gss


def m_2d_cnn_tf(x_window_size, y_window_size, channels):
    """
    This is the model in the Tensorflow docs. Use this as a template.

    Args:
        x_window_size: window size in elements
        y_window_size: window size in elements
        channels:

    Returns: 2D CNN model

    """
    print('building model...')

    model = models.Sequential()
    # new, batch normalization, since output prediction vals are whacked
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(x_window_size, y_window_size, channels)))
    model.add(layers.BatchNormalization(name='batch norm_1'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization(name='batch norm_2'))

    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization(name='batch norm_3'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax', name='output'))  # modify output classes to 2

    return model


# build training and testing data with this function


def load_1d_data(data_filename, train_test_split_percent, window_size_msecs):
    """
    This is for a 1D ResNet Classifier for the frac score ML
    I will also use this on the 1D CNN we made earlier this year


    Args:
        pops_filename: .npy file with conditioned events
        train_test_split_percent: for acc and validation acc
        window_size_msecs: should be 20 ms

    Returns: training_arrays, testing_arrays, training_labels, testing_labels

    """
    # load npy file WN,Api,stage,dt_obj_to_str(window_start),dt_obj_to_str(window_stop),x_d_win

    print('loading data, this may take some time...')

    events = np.load(data_filename, allow_pickle=True)

    # well_name = events[i][0]
    # api = events[i][1]
    # reason = events[i][2]
    # start_date = events[i][3]
    # start_time = events[i][4]
    # end_date = events[i][5]
    # end_time = events[i][6]
    # stage = events[i][7]
    # utc_start = events[i][8]
    # comments = events[i][9]
    # dynamic_values = events[i][10]
    # label = events[i][11]

    print('number of events: ', len(events))

    # split pops to non_pops

    pops_list = []
    not_pops_list = []

    for i in range(len(events)):

        dynamic_values = events[i][10]
        label = events[i][11]

        if label == 0:
            not_pops_list.append(dynamic_values)
        if label == 1:
            pops_list.append(dynamic_values)

    # force size of array, looks like some array sizes are less than expected

    forced_size_of_array = 40000 * (window_size_msecs / 1000)

    # setup output lists

    training_labels = []
    testing_labels = []
    training_values = []
    testing_values = []

    print('data loaded, now building training and testing datasets')

    # ratio is testing, training split

    ratio = int(100 / (100 - train_test_split_percent))

    for j in range(len(pops_list)):

        if len(pops_list[j]) == forced_size_of_array:

            if j % ratio == 0:

                training_labels.append([1])
                training_values.append(pops_list[j])

            else:

                testing_labels.append([1])
                testing_values.append(pops_list[j])

    for k in range(len(not_pops_list)):

        if len(not_pops_list[k]) == forced_size_of_array:

            if k % ratio == 0:

                training_labels.append([0])
                training_values.append(not_pops_list[k])

            else:

                testing_labels.append([0])
                testing_values.append(not_pops_list[k])

    # convert output arrays to numpy arrays with appropriate dimensions

    print('shape of training_values: ' + str(np.shape(training_values)))

    training_arrays = np.expand_dims(np.asarray(training_values), axis=2)
    testing_arrays = np.expand_dims(np.asarray(testing_values), axis=2)

    training_labels = np.array(training_labels, dtype=np.uint8)
    testing_labels = np.array(testing_labels, dtype=np.uint8)

    print('testing and training data building complete.' + '\n')

    return training_arrays, testing_arrays, training_labels, testing_labels


# old 1d data loader

def load_ml_data_pops(pops_filename, non_pops_filename, window_size_secs, train_test_split_percent):
    # load npy file WN,Api,stage,dt_obj_to_str(window_start),dt_obj_to_str(window_stop),x_d_win

    print('loading data, this may take some time...')

    input_pop_event = np.load(pops_filename, allow_pickle=True)
    input_not_pops = np.load(non_pops_filename, allow_pickle=True)

    # just look at the first 1000 events for each, faster debugging. remove these lines when you're ready

    input_pop_event = input_pop_event[0:1500]
    input_not_pops = input_not_pops[0:1500]

    # force size of array, looks like some array sizes are less than expected

    forced_size_of_array = 40000 * window_size_secs

    # setup output lists

    training_labels = []
    testing_labels = []
    training_values = []
    testing_values = []

    print('data loaded, now building training and testing datasets')

    # cycle through events, use ratio to split between training and testing

    ratio = int(100 / (100 - train_test_split_percent))

    for i in range(len(input_pop_event)):

        array_in_question = input_pop_event[i][5]

        if len(array_in_question) == forced_size_of_array:

            if i % ratio == 0:

                training_labels.append([1])
                training_values.append(array_in_question)

            else:

                testing_labels.append([1])
                testing_values.append(array_in_question)

    for j in range(len(input_not_pops)):

        array_in_question = input_not_pops[j][5]

        if len(array_in_question) == forced_size_of_array:

            # reshape

            if j % ratio == 0:

                training_labels.append([0])
                training_values.append(array_in_question)

            else:

                testing_labels.append([0])
                testing_values.append(array_in_question)

    # convert output arrays to numpy arrays with appropriate dimensions

    print('shape of training_values: ' + str(np.shape(training_values)))

    training_arrays = np.expand_dims(np.asarray(training_values), axis=2)
    testing_arrays = np.expand_dims(np.asarray(testing_values), axis=2)

    training_labels = np.array(training_labels, dtype=np.uint8)
    testing_labels = np.array(testing_labels, dtype=np.uint8)

    print('testing and training data building complete.' + '\n')

    return training_arrays, testing_arrays, training_labels, testing_labels


def make_spectrogram_image_matrix(array_in_question, f_min, f_max):
    """

    Args:

        array_in_question: Take a dynamic signal array and create a spectrogram image.

    Returns: A spectrogram 2D matrix (with rgb) of the spectrogram image

    """

    # plt.figure(figsize=(fig_size_x_in, fig_size_y_in))
    plt.figure()

    color_map = 'jet_r'
    sampling_rate = 40000

    #  for 5 second
    # time_sample_window = .1
    # overlap_factor = 10

    # new parameters for 1-second samples
    time_sample_window = .02
    # overlap_factor = .25
    # n_fft = int(sampling_rate * time_sample_window)  # 300 divisions

    n_fft = 256  # 300 divisions
    # n_fft = 8
    # n_overlap = 16
    n_overlap = 8
    overlap_factor = 0.25
    # n_overlap = int(sampling_rate * (time_sample_window / overlap_factor))
    # NFFT = int(n_fft * 0.05),
    measured_time = len(array_in_question) / sampling_rate

    # plt.specgram(array_in_question,
    #              NFFT=int(n_fft * 1.0),
    #              Fs=sampling_rate,
    #              noverlap=n_overlap,
    #              cmap=pylab.get_cmap(color_map))

    TIME_SAMPLE_WINDOW = .06
    OVERLAP_FACTOR = .25
    NFFT = int((sampling_rate * TIME_SAMPLE_WINDOW) / 60)
    color_map = 'jet_r'
    NOVERLAP = int(measured_time * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

    plt.specgram(array_in_question,
                 NFFT=NFFT,
                 Fs=sampling_rate,
                 noverlap=NOVERLAP,
                 cmap=pylab.get_cmap(color_map))

    # f, t, Sxx = signal.spectrogram(array_in_question, sampling_rate)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylim([0,5000])

    # plt.axis([0, measured_time, f_min, f_max])
    # plt.show()

    # plt.axis([0, measured_time, f_min, f_max])
    plt.axis('off')

    # plt.show()
    # spectrogram info is saved as image
    save_file_name = 'temp_' + '.jpg'
    plt.savefig(save_file_name)
    # plt.savefig(save_file_name, bbox_inches='tight')

    # now output as 2d array and flatten
    # spectrogram_matrix = img.imread(save_file_name).flatten()
    spectrogram_matrix = img.imread(save_file_name)
    # print('shape of new matrix: ' + str(np.shape(spectrogram_matrix)))
    # print('shape of new matrix, height: '+ str(spectrogram_matrix.shape[0]))
    # print('shape of new matrix, width: '+ str(spectrogram_matrix.shape[1]))
    # print('shape of new matrix, chan: '+ str(spectrogram_matrix.shape[2]))

    x_dims = spectrogram_matrix.shape[1]
    y_dims = spectrogram_matrix.shape[0]
    channel_dims = spectrogram_matrix.shape[2]

    # x_window_size = 800
    # y_window_size = 300
    # channels = 3

    # matrix_reshaped = spectrogram_matrix.reshape(x_window_size, y_window_size, channels)

    # print('shape of matrix_reshaped: ' + str(np.shape(matrix_reshaped)))

    out_data = np.asarray(spectrogram_matrix)

    # out_reshaped = np.array(out_data, dtype=np.uint8).reshape(1, x_window_size, y_window_size, channels)

    # try averaging all values together, we can leave RGB channels here by commenting out the avg

    # img_avg = np.mean(spectrogram_matrix, axis=2)

    # now remove image file
    os.remove(save_file_name)
    # print('file deleted, your image shape', spectrogram_matrix.shape)

    # print('shape of spectrogram:', np.shape(spectrogram_matrix))
    # print('shape of spectrogram:', np.shape(img_avg))

    plt.close()

    # return img_avg
    return spectrogram_matrix, x_dims, y_dims, channel_dims


def get_train_test_2d_cnn(pops_filename,
                          non_pops_filename,
                          window_size_secs,
                          train_test_split_percent,
                          fig_size_x_in,
                          fig_size_y_in,
                          f_min,
                          f_max):
    # load npy file WN,Api,stage,dt_obj_to_str(window_start),dt_obj_to_str(window_stop),x_d_win

    print('loading data, this may take some time...')

    input_pop_event = np.load(pops_filename, allow_pickle=True)
    input_not_pops = np.load(non_pops_filename, allow_pickle=True)

    # just look at the first 2000 events for each, faster debugging. remove these lines when you're ready

    number_of_events = 2000

    input_pop_event = input_pop_event[0:number_of_events]
    input_not_pops = input_not_pops[0:number_of_events]

    # force size of array, looks like some array sizes are less than expected

    forced_size_of_array = 40000 * window_size_secs
    # channels = 1
    channels = 3
    dpi = 100
    x_window_size = int(dpi * fig_size_x_in)  # to pixels
    y_window_size = int(dpi * fig_size_y_in)  # to pixels

    # setup output lists

    training_labels = []
    testing_labels = []
    training_values = []
    testing_values = []

    # class labels

    class_labels = ['pops', 'not_pops']

    class_label_matrix = np.identity(len(class_labels))

    print('data loaded, now building training and testing datasets')

    # cycle through events, use ratio to split between training and testing

    ratio = int(100 / (100 - train_test_split_percent))

    for i in range(len(input_pop_event)):

        array_in_question = input_pop_event[i][5]

        if len(array_in_question) == forced_size_of_array:

            # spectrogram_matrix = make_spectrogram_image_matrix(array_in_question,
            #                                                    fig_size_x_in,
            #                                                    fig_size_y_in,
            #                                                    f_min,
            #                                                    f_max)

            spectrogram_matrix, x_dims, y_dims, channel_dims = make_spectrogram_image_matrix(array_in_question,
                                                                                             f_min,
                                                                                             f_max)
            if i % ratio == 0:

                training_labels.append([1])
                training_values.append(spectrogram_matrix)

            else:

                testing_labels.append([1])
                testing_values.append(spectrogram_matrix)

        print('event number: ' + str(i) + ' , percent complete: ' + str(round((i / len(input_pop_event)) * 100, 2)))

    for j in range(len(input_not_pops)):

        array_in_question = input_not_pops[j][5]

        if len(array_in_question) == forced_size_of_array:

            # spectrogram_matrix = make_spectrogram_image_matrix(array_in_question,
            #                                                    fig_size_x_in,
            #                                                    fig_size_y_in,
            #                                                    f_min,
            #                                                    f_max)

            spectrogram_matrix, x_dims, y_dims, channel_dims = make_spectrogram_image_matrix(array_in_question,
                                                                                             f_min,
                                                                                             f_max)
            if j % ratio == 0:

                training_labels.append([0])
                training_values.append(spectrogram_matrix)

            else:

                testing_labels.append([0])
                testing_values.append(spectrogram_matrix)

        print('event number: ' + str(j) + ' , percent complete: ' + str(round((j / len(input_not_pops)) * 100, 2)))

    # convert output arrays to numpy arrays with appropriate dimensions

    # x_train = training_values
    # x_test = testing_values
    # y_train = np.asarray(training_labels)
    # y_test = np.asarray(testing_labels)

    print('testing and training data building complete.' + '\n')

    # reshape data objects before putting into model
    #
    # x_train = np.array(training_values, dtype=np.uint8).reshape(len(training_values), x_window_size, y_window_size,
    #                                                             channels)
    # x_test = np.array(testing_values, dtype=np.uint8).reshape(len(testing_values), x_window_size, y_window_size,
    #                                                           channels)

    x_train = np.array(training_values, dtype=np.uint8).reshape(len(training_values), y_dims, x_dims,
                                                                channel_dims)
    x_test = np.array(testing_values, dtype=np.uint8).reshape(len(testing_values), y_dims, x_dims,
                                                              channel_dims)

    y_train = np.array(training_labels, dtype=np.uint8)
    y_test = np.array(testing_labels, dtype=np.uint8)

    return x_train, x_test, y_train, y_test


def load_2d_data(data_filename, train_test_split_percent, window_size_msecs):
    # load npy file WN,Api,stage,dt_obj_to_str(window_start),dt_obj_to_str(window_stop),x_d_win

    sampling_rate = 40000

    print('loading data, this may take some time...')

    events = np.load(data_filename, allow_pickle=True)

    # well_name = events[i][0]
    # api = events[i][1]
    # reason = events[i][2]
    # start_date = events[i][3]
    # start_time = events[i][4]
    # end_date = events[i][5]
    # end_time = events[i][6]
    # stage = events[i][7]
    # utc_start = events[i][8]
    # comments = events[i][9]
    # dynamic_values = events[i][10]
    # label = events[i][11]

    # force size of array, looks like some array sizes are less than expected

    forced_size_of_array = sampling_rate * (window_size_msecs / 1000)

    # split pops to non_pops

    pops_list = []
    not_pops_list = []

    for i in range(len(events)):

        dynamic_values = events[i][10]

        if len(dynamic_values) == forced_size_of_array:

            label = events[i][11]

            if label == 0:
                not_pops_list.append(dynamic_values)
            if label == 1:
                pops_list.append(dynamic_values)

    print('number of events: ', len(events))
    print('number of pops: ', len(pops_list))
    print('number of non pops: ', len(not_pops_list))

    # setup output lists

    training_labels = []
    testing_labels = []
    training_values = []
    testing_values = []

    y_dims = 0
    x_dims = 0

    # class labels

    class_labels = ['pops', 'not_pops']

    class_label_matrix = np.identity(len(class_labels))

    print('data loaded, now building training and testing datasets')

    # ratio is testing, training split

    ratio = int(100 / (100 - train_test_split_percent))

    for j in range(len(pops_list)):

        window_in_question = pops_list[j]
        spectrogram_matrix = make_fft_matrix(window_in_question, sampling_rate, 50)
        y_dims = spectrogram_matrix.shape[0]
        x_dims = spectrogram_matrix.shape[1]

        if len(window_in_question) == forced_size_of_array:

            if j % ratio == 0:

                training_labels.append([1])
                training_values.append(spectrogram_matrix)

            else:

                testing_labels.append([1])
                testing_values.append(spectrogram_matrix)

    for k in range(len(not_pops_list)):

        window_in_question = not_pops_list[k]
        spectrogram_matrix = make_fft_matrix(window_in_question, sampling_rate, 50)
        y_dims = spectrogram_matrix.shape[0]
        x_dims = spectrogram_matrix.shape[1]

        if len(window_in_question) == forced_size_of_array:

            if k % ratio == 0:

                training_labels.append([0])
                training_values.append(spectrogram_matrix)

            else:

                testing_labels.append([0])
                testing_values.append(spectrogram_matrix)

    # convert output arrays to numpy arrays with appropriate dimensions

    print('testing and training data building complete.' + '\n')

    channel_dims = 1

    x_train = np.array(training_values, dtype=np.uint8).reshape(len(training_values), y_dims, x_dims, channel_dims)
    x_test = np.array(testing_values, dtype=np.uint8).reshape(len(testing_values), y_dims, x_dims, channel_dims)
    y_train = np.array(training_labels, dtype=np.uint8)
    y_test = np.array(testing_labels, dtype=np.uint8)

    return x_train, x_test, y_train, y_test


def get_train_test_2d_cnn_new(pops_filename,
                              non_pops_filename,
                              window_size_secs,
                              train_test_split_percent
                              ):
    # load npy file WN,Api,stage,dt_obj_to_str(window_start),dt_obj_to_str(window_stop),x_d_win

    print('loading data, this may take some time...')

    input_pop_event = np.load(pops_filename, allow_pickle=True)
    input_not_pops = np.load(non_pops_filename, allow_pickle=True)

    # just look at the first 2000 events for each, faster debugging. remove these lines when you're ready

    number_of_events = 2000

    input_pop_event = input_pop_event[0:number_of_events]
    input_not_pops = input_not_pops[0:number_of_events]

    # # force size of array, looks like some array sizes are less than expected
    #
    forced_size_of_array = 40000 * window_size_secs
    # #channels = 1
    # channels = 3
    # dpi = 100
    # x_window_size = int(dpi * fig_size_x_in)  # to pixels
    # y_window_size = int(dpi * fig_size_y_in)  # to pixels

    # setup output lists

    training_labels = []
    testing_labels = []
    training_values = []
    testing_values = []

    y_dims = 0
    x_dims = 0

    # class labels

    class_labels = ['pops', 'not_pops']

    class_label_matrix = np.identity(len(class_labels))

    print('data loaded, now building training and testing datasets')

    # cycle through events, use ratio to split between training and testing

    ratio = int(100 / (100 - train_test_split_percent))

    for i in range(len(input_pop_event)):

        array_in_question = input_pop_event[i][5]

        if len(array_in_question) == forced_size_of_array:

            spectrogram_matrix = make_fft_matrix(array_in_question, 40000, 50)

            y_dims = spectrogram_matrix.shape[0]
            x_dims = spectrogram_matrix.shape[1]

            if i % ratio == 0:

                training_labels.append([1])
                training_values.append(spectrogram_matrix)

            else:

                testing_labels.append([1])
                testing_values.append(spectrogram_matrix)

        print('event number: ' + str(i) + ' , percent complete: ' + str(round((i / len(input_pop_event)) * 100, 2)))

    for j in range(len(input_not_pops)):

        array_in_question = input_not_pops[j][5]

        if len(array_in_question) == forced_size_of_array:

            spectrogram_matrix = make_fft_matrix(array_in_question, 40000, 50)

            if j % ratio == 0:

                training_labels.append([0])
                training_values.append(spectrogram_matrix)

            else:

                testing_labels.append([0])
                testing_values.append(spectrogram_matrix)

        print('event number: ' + str(j) + ' , percent complete: ' + str(round((j / len(input_not_pops)) * 100, 2)))

    # convert output arrays to numpy arrays with appropriate dimensions

    print('testing and training data building complete.' + '\n')

    channel_dims = 1

    x_train = np.array(training_values, dtype=np.uint8).reshape(len(training_values), y_dims, x_dims,
                                                                channel_dims)
    x_test = np.array(testing_values, dtype=np.uint8).reshape(len(testing_values), y_dims, x_dims,
                                                              channel_dims)

    y_train = np.array(training_labels, dtype=np.uint8)
    y_test = np.array(testing_labels, dtype=np.uint8)

    return x_train, x_test, y_train, y_test

def resnet_1d_model(n_outputs, window_size):

    n_feature_maps = 64

    #input_layer = layers.Input(input_shape=(window_size, 1))
    input_layer = layers.Input((window_size, 1))

    # BLOCK 1

    conv_x = layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation('relu')(conv_x)

    conv_y = layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = layers.BatchNormalization()(conv_y)
    conv_y = layers.Activation('relu')(conv_y)

    conv_z = layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = layers.BatchNormalization()(conv_z)

    # expand channels for the sum

    shortcut_y = layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = layers.BatchNormalization()(shortcut_y)

    output_block_1 = layers.add([shortcut_y, conv_z])
    output_block_1 = layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation('relu')(conv_x)

    conv_y = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = layers.BatchNormalization()(conv_y)
    conv_y = layers.Activation('relu')(conv_y)

    conv_z = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = layers.BatchNormalization()(shortcut_y)

    output_block_2 = layers.add([shortcut_y, conv_z])
    output_block_2 = layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = layers.BatchNormalization()(conv_x)
    conv_x = layers.Activation('relu')(conv_x)

    conv_y = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = layers.BatchNormalization()(conv_y)
    conv_y = layers.Activation('relu')(conv_y)

    conv_z = layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = layers.BatchNormalization()(output_block_2)

    output_block_3 = layers.add([shortcut_y, conv_z])
    output_block_3 = layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = layers.Dense(n_outputs, activation='softmax')(gap_layer)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

    #
    # m = Sequential()
    #
    # m.add(layers.InputLayer(input_shape=(window_size, 1)))
    #
    # # block 1
    #
    # m.add(layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same', name='conv block x 1'))
    # m.add(layers.BatchNormalization(name='batch norm block x 1'))
    # m.add(layers.Activation(activation='relu'))
    #
    # m.add(layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same', name='conv block y 1'))
    # m.add(layers.BatchNormalization(name='batch norm block y 1'))
    # m.add(layers.Activation(activation='relu'))
    #
    # m.add(layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same', name='conv block z 1'))
    # m.add(layers.BatchNormalization(name='batch norm block z 1'))
    #
    # # expand channels for sum?
    #
    # input_layer = layers.Input(input_shape=(window_size, 1))
    # shortcut_y = layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    # shortcut_y = layers.BatchNormalization()(shortcut_y)
    #
    # output_block1 = layers.a
    # return m


def dynamic_model(n_outputs, window_size):
    # dynamic 1d convo model
    m = Sequential()
    m.add(layers.InputLayer(input_shape=(window_size, 1)))
    m.add(layers.BatchNormalization(name='batch_norm_1'))
    m.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', name='conv_1'))
    m.add(layers.BatchNormalization(name='batch_norm_2'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_1'))
    m.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='conv_2'))
    m.add(layers.BatchNormalization(name='batch_norm_3'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_2'))
    m.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', name='conv_3'))
    m.add(layers.BatchNormalization(name='batch_norm_4'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_3'))
    m.add(layers.Conv1D(filters=8, kernel_size=5, activation='relu', name='conv_4'))
    m.add(layers.BatchNormalization(name='batch_norm_5'))
    m.add(layers.MaxPooling1D(pool_size=3, name='max_pool_4'))
    m.add(layers.Flatten(name='flatten'))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(128, activation='relu', name='dense_1'))
    m.add(layers.Dense(n_outputs, activation='softmax', name='output'))

    return m


def evaluate_model(input_model, values_train, labels_train, values_test, labels_test):
    input_model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'],
                        )
    # fit network
    input_model.fit(x=values_train,
                    y=labels_train,
                    validation_data=(values_test, labels_test),
                    # callbacks=[monitor],
                    epochs=20,
                    verbose=2)

    # evaluate model
    loss_values, accuracy = input_model.evaluate(x=values_test,
                                                 y=labels_test,
                                                 verbose=2)
    return accuracy


def make_frac_score_data_non_events(df_well_ids, df_eff_report, num_windows_per_stage, counting_window_secs,
                                    time_before_pop_secs, time_after_pop_secs):
    now_str = datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    save_file_name = 'fracScore_allPads_non_events' + now_str + '.npy'

    print('saving data under the filename: ' + save_file_name)

    non_events_obj = []

    total_events = len(df_eff_report)

    for i in range(total_events):

        print('analyzing record: ' + str(i) + ' of: ' + str(total_events) + '\n')

        well_name = str(df_eff_report.loc[i, "Well"])
        api = str(df_eff_report.loc[i, "API #"])
        stage = str(df_eff_report.loc[i, "Stage"])
        start_date = str(df_eff_report.loc[i, "Start Date"])
        start_time = str(df_eff_report.loc[i, "Start Time"])
        end_date = str(df_eff_report.loc[i, "End Date"])
        end_time = str(df_eff_report.loc[i, "End Time"])
        reason = str(df_eff_report.loc[i, "Reason"])
        comments = str(df_eff_report.loc[i, "Comments"])
        stamped_time = str(df_eff_report.loc[i, "TimeInMin"])

        if reason != 'Pumping':

            # Gather Well info for output text

            # print out well and job info

            print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage,
                                    stamped_time, comments)

            utc_start = mk_utc_time(start_date, start_time)
            utc_end = mk_utc_time(end_date, end_time)

            # make sure time is greater than 10 minutes

            if float(stamped_time) > 10:

                # now pull IDs

                static_id, dynamic_id = fetch_sensor_ids_from_apis(df_well_ids, api)

                if dynamic_id != '-1' and static_id != '-1':

                    # we wish to draw X number of samples, or num_windows_per_stage

                    for j in range(num_windows_per_stage):

                        print(' j - ', j)
                        print('num windows per stage: ', num_windows_per_stage)
                        print('counting windows secs: ', counting_window_secs)
                        print('float of stamped time : ' + stamped_time)
                        print('utc start: ' + utc_start)
                        print('utc end: ' + utc_end)

                        rdm_window_utc_start, rdm_window_utc_stop = mk_random_windows(counting_window_secs,
                                                                                      int(float(stamped_time)),
                                                                                      utc_start)

                        # now make data

                        x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dynamic_id,
                                                                               static_id,
                                                                               rdm_window_utc_start,
                                                                               rdm_window_utc_stop)

                        if len(x_d) > 0:

                            # just pull a window

                            activation_time = mk_data_time_from_str(rdm_window_utc_start)
                            window_start = activation_time - dt.timedelta(seconds=time_before_pop_secs)
                            window_stop = activation_time + dt.timedelta(seconds=time_after_pop_secs)
                            # pull new data

                            x_d_win = interval_to_flat_array_resample(dynamic_id, window_start, window_stop).values()

                            # now add all the good stuff

                            non_events_obj.append([well_name,
                                                   api,
                                                   stage,
                                                   dt_obj_to_str(window_start),
                                                   dt_obj_to_str(window_stop),
                                                   x_d_win])

                        else:

                            print('unable to evaluate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100 * (i / total_events), 2)) + '\n\n')

    # save the data while you can!

    print('frac data build complete')

    np.save(save_file_name, non_events_obj)


def make_data(dyn_id, stat_id, start_string, stop_string):
    # pull dyn pressure data. Does not include accelerometer data

    start_converted = parse_time_string_with_colon_offset(start_string)
    stop_converted = parse_time_string_with_colon_offset(stop_string)
    delta = stop_converted - start_converted
    seconds = int(delta.total_seconds())
    x_d = interval_to_flat_array_resample(dyn_id, start_converted, stop_converted).values()
    # x_d = interval_to_flat_array_resample(dynID, start_converted, stop_converted)

    sampling_rate = int(len(x_d) / seconds)

    t_d = np.linspace(0, seconds, len(x_d))

    # make static data

    stat_obj = fetch_sensor_db_data(stat_id, start_string, stop_string)

    x_s = stat_obj['max'].to_numpy()
    t_s = np.linspace(0, len(x_s), len(x_s))

    return x_d, t_d, x_s, t_s, sampling_rate, seconds


def frac_score_detector_gradient(dynamic_window, sampling_rate):
    # New method, with gradient method to weed out false signals
    # Frac score method, updated version with bandpass filter

    dist = 0.02 * sampling_rate
    f_low = 50
    f_high = 500
    height = 0.2

    filtered_xd = butter_bandpass_filter(dynamic_window, f_low, f_high, sampling_rate, order=9)
    max_dyn = np.nanmax(np.absolute(filtered_xd))
    dynamic_normalized = np.absolute(filtered_xd) / np.absolute(max_dyn)
    dynamic_gradient = np.gradient(dynamic_normalized)

    peaks_dynamic_g, _dict = signal.find_peaks(np.absolute(dynamic_gradient), height=height, distance=dist)

    return peaks_dynamic_g


def frac_score_detector(dynamic_window, sampling_rate, f_low, f_high, std_m, distance_val):
    # Frac score method, updated version with bandpass filter

    filtered_xd = butter_bandpass_filter(dynamic_window, f_low, f_high, sampling_rate, order=9)

    # Std_dynamic_window is the standard deviation value of the filtered dynamic pressure response

    std_dynamic_window = np.std(np.absolute(filtered_xd))

    # see the scipy.signal docs for more info.
    # STD_MULTIPLIER = 14.5
    # distance = 1000

    # 7. The height threshold is the acceptable signal to noise ration, or relative magnitude of the filtered signal

    height_threshold = std_m * std_dynamic_window

    # 8. Using the signal find peaks method, see the scipy.signal docs for more info.

    peaks, _ = signal.find_peaks(np.absolute(filtered_xd), height=height_threshold, distance=distance_val)

    return peaks


def fetch_sensor_ids_from_apis(df_well_ids, api_to_match):
    stat_id = ''
    dyn_id = ''

    for i in range(len(df_well_ids)):

        api_raw = str(df_well_ids.loc[i, "api"])
        api_mod = api_raw.replace('-', '')
        api_mod = api_mod[:-4]
        api_to_match = api_to_match.replace('-', '')

        if api_to_match == api_mod:
            stat_id = str(df_well_ids.loc[i, "static_id"])
            dyn_id = str(df_well_ids.loc[i, "dynamic_id"])

    return stat_id, dyn_id


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    sos = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def butter_low_pass_filter(data, cutoff, fs, order=5):
    b, a = butter_low_pass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_low_pass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def mk_utc_time(str_date, str_time):
    # combine strings:

    date = str_date.split('/')
    time = str_time.split(':')

    mth = str(date[0])
    day = str(date[1])
    yr = str(date[2])
    hr = str(time[0])
    mn = str(time[1])
    # sec = '00.000000Z'
    # sec = '00.000000000Z'
    sec = '00'
    fixed_utc = '05:00'

    if len(date[0]) != 2:
        mth = '0' + mth

    if len(date[1]) != 2:
        day = '0' + day

    if len(time[0]) != 2:
        hr = '0' + hr

    if len(time[1]) != 2:
        mn = '0' + mn

    out_str = '20' + yr + '-' + mth + '-' + day + 'T' + hr + ':' + mn + ':' + sec + '-' + fixed_utc

    # '2020-07-29T14:45:00-05:00'
    # print('new utc time: ' + out_str)

    return out_str


def mk_random_windows(window_size, stamped_time, utc_start):
    new_start = int(random.uniform(0, int(stamped_time)))

    # print('new start: ', new_start)

    rdm_window_utc_start = mk_data_time_from_str(utc_start) + dt.timedelta(seconds=60 * new_start)
    rdm_window_utc_stop = rdm_window_utc_start + dt.timedelta(seconds=window_size)

    return dt_obj_to_str(rdm_window_utc_start), dt_obj_to_str(rdm_window_utc_stop)


def print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage, stamped_time,
                            comments):
    out_text = 'Well Name: ' + well_name + '\n'
    out_text += 'API: ' + api + '\n'
    out_text += 'Event Type: ' + reason + '\n'
    out_text += 'Start Date: ' + start_date + '\n'
    out_text += 'Start Time: ' + start_time + '\n'
    out_text += 'End Date: ' + end_date + '\n'
    out_text += 'End Time: ' + end_time + '\n'
    out_text += 'Stage Number: ' + stage + '\n'
    out_text += 'Time ' + '(' + 'minutes' + ')' + ':' + stamped_time + '\n'
    out_text += 'Comments: ' + comments + '\n'

    # make UTC start stop time

    utc_start = mk_utc_time(start_date, start_time)
    utc_end = mk_utc_time(end_date, end_time)

    out_text += 'UTC start dateTime: ' + utc_start + '\n'
    out_text += 'UTC stop dateTime: ' + utc_end

    print(out_text)


def make_frac_score_data_set(df_well_ids, df_eff_report,
                             num_windows_per_stage,
                             counting_window_secs,
                             std_m, distance_val,
                             time_before_pop_secs,
                             time_after_pop_secs):
    now_str = datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    save_file_name = 'fracScore_allPads_' + now_str + '.npy'

    print('saving data under the filename: ' + save_file_name)

    frac_score_pops_all_pads = []

    total_events = len(df_eff_report)

    for i in range(total_events):

        print('analyzing record: ' + str(i) + ' of: ' + str(total_events) + '\n')

        well_name = str(df_eff_report.loc[i, "Well"])
        api = str(df_eff_report.loc[i, "API #"])
        stage = str(df_eff_report.loc[i, "Stage"])
        # crew = str(df_eff_report.loc[i, "Crew"])
        start_date = str(df_eff_report.loc[i, "Start Date"])
        start_time = str(df_eff_report.loc[i, "Start Time"])
        end_date = str(df_eff_report.loc[i, "End Date"])
        end_time = str(df_eff_report.loc[i, "End Time"])
        reason = str(df_eff_report.loc[i, "Reason"])
        comments = str(df_eff_report.loc[i, "Comments"])
        stamped_time = str(df_eff_report.loc[i, "TimeInMin"])

        if reason == 'Pumping':

            # print out well and job info

            print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage,
                                    stamped_time, comments)

            utc_start = mk_utc_time(start_date, start_time)
            utc_end = mk_utc_time(end_date, end_time)

            # make sure time is greater than 10 minutes

            if float(stamped_time) > 10:

                # pull IDs

                stat_id, dyn_id = fetch_sensor_ids_from_apis(df_well_ids, api)

                # draw X number of samples, or num_windows_per_stage

                for j in range(num_windows_per_stage):

                    print(' j - ', j)
                    print('num windows per stage: ', num_windows_per_stage)
                    print('counting windows secs: ', counting_window_secs)
                    print('float of stamped time : ' + stamped_time)
                    print('utc start: ' + utc_start)
                    print('utc end: ' + utc_end)

                    rdm_window_utc_start, rdm_window_utc_stop = mk_random_windows(counting_window_secs,
                                                                                  int(float(stamped_time)),
                                                                                  utc_start)
                    # now make data

                    x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dyn_id,
                                                                           stat_id,
                                                                           rdm_window_utc_start,
                                                                           rdm_window_utc_stop)
                    if len(x_d) > 0:

                        # We assume that the frac score is pretty decent at finding BASIS pops...

                        peaks_bp = frac_score_detector(x_d, sampling_rate, 150, 500, std_m, distance_val)

                        for pk in peaks_bp:

                            # ID activation second, build data before and after peak, then draw new values

                            if pk > 0.01 * sampling_rate:
                                activation_second = t_d[pk]

                                activation_time = mk_data_time_from_str(rdm_window_utc_start) + dt.timedelta(
                                    seconds=activation_second)

                                # write time before and after

                                window_start = activation_time - dt.timedelta(seconds=time_before_pop_secs)
                                window_stop = activation_time + dt.timedelta(seconds=time_after_pop_secs)

                                # pull new data

                                x_d_win = interval_to_flat_array_resample(dyn_id, window_start, window_stop).values()

                                # now add all the good stuff

                                frac_score_pops_all_pads.append([well_name,
                                                                 api,
                                                                 stage,
                                                                 dt_obj_to_str(window_start),
                                                                 dt_obj_to_str(window_stop),
                                                                 x_d_win])

                    else:

                        print('unable to evaluate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100 * (i / total_events), 2)) + '\n\n')

    # save the data

    print('frac data build complete')

    np.save(save_file_name, frac_score_pops_all_pads)


def make_frac_score_data_set_with_gradients(df_well_ids, df_eff_report,
                                            num_windows_per_stage,
                                            counting_window_secs,
                                            std_m, distance_val,
                                            time_before_pop_secs,
                                            time_after_pop_secs):
    # note, gradient method works better for long time spans, like 10mins

    now_str = datetime.now().strftime("%d_%m_%Y_T%H_%M_%S")
    save_file_name = 'fracScore_allPads_' + now_str + '.npy'

    print('saving data under the filename: ' + save_file_name)

    frac_score_pops_all_pads = []

    total_events = len(df_eff_report)

    for i in range(total_events):

        print('analyzing record: ' + str(i) + ' of: ' + str(total_events) + '\n')

        well_name = str(df_eff_report.loc[i, "Well"])
        api = str(df_eff_report.loc[i, "API #"])
        stage = str(df_eff_report.loc[i, "Stage"])
        # crew = str(df_eff_report.loc[i, "Crew"])
        start_date = str(df_eff_report.loc[i, "Start Date"])
        start_time = str(df_eff_report.loc[i, "Start Time"])
        end_date = str(df_eff_report.loc[i, "End Date"])
        end_time = str(df_eff_report.loc[i, "End Time"])
        reason = str(df_eff_report.loc[i, "Reason"])
        comments = str(df_eff_report.loc[i, "Comments"])
        stamped_time = str(df_eff_report.loc[i, "TimeInMin"])

        if reason == 'Pumping':

            # print out well and job info

            print_out_well_job_info(well_name, api, reason, start_date, start_time, end_date, end_time, stage,
                                    stamped_time, comments)

            utc_start = mk_utc_time(start_date, start_time)
            utc_end = mk_utc_time(end_date, end_time)

            # make sure time is greater than 10 minutes

            if float(stamped_time) > 10:

                # pull IDs

                stat_id, dyn_id = fetch_sensor_ids_from_apis(df_well_ids, api)

                # draw X number of samples, or num_windows_per_stage

                for j in range(num_windows_per_stage):

                    print(' j - ', j)
                    print('num windows per stage: ', num_windows_per_stage)
                    print('counting windows secs: ', counting_window_secs)
                    print('float of stamped time : ' + stamped_time)
                    print('utc start: ' + utc_start)
                    print('utc end: ' + utc_end)

                    rdm_window_utc_start, rdm_window_utc_stop = mk_random_windows(counting_window_secs,
                                                                                  int(float(stamped_time)),
                                                                                  utc_start)
                    # now make data

                    x_d, t_d, x_s, t_s, sampling_rate, seconds = make_data(dyn_id,
                                                                           stat_id,
                                                                           rdm_window_utc_start,
                                                                           rdm_window_utc_stop)
                    if len(x_d) > 0:

                        # We assume that the frac score is pretty decent at finding BASIS pops...

                        # peaks_bp = frac_score_detector(x_d, sampling_rate, 150, 500, std_m, distance_val)
                        peaks_grad = frac_score_detector_gradient(x_d, sampling_rate)

                        for pk in peaks_grad:

                            # ID activation second, build data before and after peak, then draw new values

                            if pk > 0.01 * sampling_rate:
                                activation_second = t_d[pk]

                                activation_time = mk_data_time_from_str(rdm_window_utc_start) + dt.timedelta(
                                    seconds=activation_second)

                                # write time before and after

                                window_start = activation_time - dt.timedelta(seconds=time_before_pop_secs)
                                window_stop = activation_time + dt.timedelta(seconds=time_after_pop_secs)

                                # pull new data

                                x_d_win = interval_to_flat_array_resample(dyn_id, window_start, window_stop).values()

                                # now add all the good stuff

                                frac_score_pops_all_pads.append([well_name,
                                                                 api,
                                                                 stage,
                                                                 dt_obj_to_str(window_start),
                                                                 dt_obj_to_str(window_stop),
                                                                 x_d_win])

                    else:

                        print('unable to evaluate frac since length of dynamic array is: ' + str(len(x_d)) + '\n\n')

        print('percent complete: ' + str(round(100 * (i / total_events), 2)) + '\n\n')

    # save the data

    print('frac data build complete')

    np.save(save_file_name, frac_score_pops_all_pads)

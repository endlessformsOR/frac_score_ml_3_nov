import os
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
import matplotlib.image as img
import tensorflow as tf
from ml_utils import frac_score_detector, butter_bandpass_filter, make_spectrogram_image_matrix, fracScoreDetector_gss, make_fft_matrix
from scipy import signal

print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

"""
Use this as a test harness for measuring complimentary classification between models

9/23 - Looking at 1 sec windows, NNs will classify on millisecond timescale 
9/24 - Looking at 50 20ms windows, applying 1d and 2d convo models 
9/25 - Added RNN model with LSTMs  
9/28 - Added ResNet model v1
10/8 - New data with gradient method, retrained resnet
10/11 - updated gradient data, better model 
10/17 - conditioned pops dataset 
10/29 - Updated ResNet dataset and model with new spectro method for 20ms data
"""

# # open file 7000 pops, 5.2 GB
# detectedPops = np.load('raw_ml_data/fracScore_allPads_08_09_pop_events.npy', allow_pickle=True)
# # need to adjust, 5788 nonevents, filesize is 4.3 GB
# notPops = np.load('raw_ml_data/fracScore_allPads_non_events10_09_2020_T02_10_22.npy', allow_pickle=True)

# 1 - second windows
#
# # open file 4245 pops, 661.7 MB
# detectedPops = np.load('raw_ml_data/fracScore_allPads_23_09_2020_T01_05_44.npy', allow_pickle=True)
# # need to adjust, 4348 nonevents, filesize is 677.3 MB
# notPops = np.load('raw_ml_data/fracScore_allPads_non_events23_09_2020_T01_06_36.npy', allow_pickle=True)

# 9/24 - New dataset, higher std_m

# std_m = 12
# open file 5618 pops, 875 MB

# 10/11/20

# new data set, vals collected by taking the gradient of raw dynamic signal:

#detectedPops = np.load('raw_ml_data/fracScore_allPads_11_10_2020_T00_04_07.npy', allow_pickle=True)

# 10/17 clean, conditioned 20ms pops dataset.
detectedPops = np.load('raw_ml_data/conditioned_pops_1_sec_updated.npy', allow_pickle=True)

# need to adjust, 4343 nonevents, filesize is 676 MB
notPops = np.load('raw_ml_data/fracScore_allPads_non_events24_09_2020_T13_49_41.npy', allow_pickle=True)

print('number of pops detected: ' + str(len(detectedPops)))
print('number of nonevents detected: ' + str(len(notPops)))

# set axis dims
class_labels = ['pops', 'not_pops']

fig_size_x_in = 6
fig_size_y_in = 3
fig_x = 25
fig_y = 10
f_min = 1
f_max = 5000
COLORMAP = 'jet_r'
TIME_SAMPLE_WINDOW = .0075  # 0.01 is good for 1 sec
OVERLAP_FACTOR = 50  # 10 is good for 1 sec
sampling_rate = 40000
n_FFT = int(sampling_rate * TIME_SAMPLE_WINDOW)  # 5ms window
n_overlap = int(sampling_rate * (TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

# view detected pop events

print('number of pops detected: ' + str(len(detectedPops)))

def plot_event(index, data_object, event_type):
    well_name = data_object[index][0]
    api = data_object[index][1]
    start = data_object[index][3]
    stop = data_object[index][4]
    values_array = data_object[index][5]

    out_text = 'pop event: ' + str(index) + '\n'
    out_text += 'WN: ' + well_name + '\n'
    out_text += 'API: ' + api + '\n'
    out_text += 'start: ' + start + '\n'
    out_text += 'stop: ' + stop + '\n'

    if event_type == 'pop':
        stage = data_object[i][2]
        out_text += 'stage: ' + str(stage) + '\n'

    out_text += "std_m = 10, red lines" + '\n'
    out_text += 'std_m = 6, green lines' + '\n'
    out_text += 'gss, brown lines' + '\n'
    out_text += 'res3_oct11 = red' + '\n'
    out_text += 'res4_oct17 = gold' + '\n'
    out_text += 'res5_oct29 = purple'



    measured_time = len(values_array) / sampling_rate
    time_array = np.linspace(0, measured_time, len(values_array))

    # subplot stuff

    fig, axs = plt.subplots(3, 1, figsize=(fig_x, fig_y))

    # 1st plot, dynamic signal

    axs[0].set_title('Dynamic Pressure Signal: ' + str(event_type), fontsize=18, color='black')
    axs[0].plot(time_array, values_array, color='black', label='dyn, raw', zorder=1)

    # filtered signal for frac score

    filtered_xd = butter_bandpass_filter(values_array, 150, 500, sampling_rate, order=9)

    axs[0].plot(time_array, filtered_xd, color='blue', label='bp filtered', zorder=1)

    axs[0].axis([0, measured_time, np.nanmin(values_array), np.nanmax(values_array)])
    axs[0].legend(loc='upper right')
    axs[0].text(1, 0.6 * min(values_array), out_text, fontsize=11, verticalalignment='center')
    axs[0].set_ylabel('Rel. Mag.')

    # spectrogram. Specgram method

    axs[1].set_title('Spectrogram: ' + str(event_type), fontsize=18, color='black')

    axs[1].specgram(values_array,
                    NFFT=n_FFT,
                    Fs=sampling_rate,
                    noverlap=n_overlap,
                    cmap=pylab.get_cmap(COLORMAP))

    axs[1].set_ylabel('Frequency in Hz')
    axs[1].axis([0, measured_time, f_min, f_max])

    # show 0.02 portion

    ten_millisecs = int(0.01 * sampling_rate)
    mid_point = int(len(values_array) / 2)

    ten_milisecond_window = values_array[mid_point - ten_millisecs:mid_point + ten_millisecs]

    # plot it
    axs[2].set_title('new spectro', fontsize=18, color='black')
    # axs[2].plot(time_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
    #             values_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
    #             color='black',
    #             label='dyn,',
    #             zorder=1)

    new_matrix = make_fft_matrix(values_array, 40000, 1000)

    axs[2].imshow(new_matrix)
    axs[2].axis('off')


    # run heuristic frac score detector

    # std_m = 10, red lines
    # std_m = 6, green lines
    # gss, brown lines
    # res3_oct11 = red
    # res4_oct17 = gold
    # res5_oct29 = purple

    peaks_10 = frac_score_detector(values_array,
                                   sampling_rate,
                                   f_low=150,
                                   f_high=500,
                                   std_m=10,
                                   distance_val=40)

    for pk in peaks_10:

        pop_time = time_array[pk]

        if pop_time > 0.01:
            axs[0].axvline(x=pop_time,
                           ls='-.',
                           linewidth=3,
                           color='red',
                           label='heuristic')

    peaks_6 = frac_score_detector(values_array,
                                  sampling_rate,
                                  f_low=150,
                                  f_high=500,
                                  std_m=6,
                                  distance_val=40)

    for pk in peaks_6:

        pop_time = time_array[pk]

        if pop_time > 0.01:
            axs[0].axvline(x=pop_time,
                           ls='--',
                           linewidth=3,
                           color='green',
                           label='heuristic')

    gss_peaks = fracScoreDetector_gss(values_array, sampling_rate)

    for pk in gss_peaks:

        pop_time = time_array[pk]

        if pop_time > 0.01:
            axs[0].axvline(x=pop_time,
                           ls='--',
                           linewidth=3,
                           color='brown',
                           label='gss')

    # # look at pop spectro
    # axs[3].set_title('Spectrogram: 0.5s +/- 0.01s', fontsize=18, color='black')
    # axs[3].specgram(values_array[mid_point - ten_millisecs:mid_point + ten_millisecs],
    #                 NFFT=int(n_FFT * 0.05),
    #                 Fs=sampling_rate,
    #                 noverlap=int(n_overlap * 0.1),
    #                 cmap=pylab.get_cmap(COLORMAP))
    #
    # axs[3].set_ylabel('Frequency in Hz')
    # axs[3].axis([, measured_time, F_MIN, F_MAX])

    ### ML stuff

    # load models

    model_1d_cnn = tf.keras.models.load_model('ml_models/1dcnn_20msec.h5')
    model_1d_rnn = tf.keras.models.load_model('ml_models/rnn_20msec.h5')
    model_2d_cnn = tf.keras.models.load_model('ml_models/2dcnn_20msec.h5')
    model_resNet_v1 = tf.keras.models.load_model('ml_models/resNet_20_msec.h5')
    model_resNet_grad = tf.keras.models.load_model('ml_models/resNet_20_msec_7oct2020.h5')
    model_resNet_grad_2 = tf.keras.models.load_model('ml_models/resNet_20_msec_8oct2020.h5')

    model_resNet_grad_3 = tf.keras.models.load_model('ml_models/resNet_20_msec_11oct2020.h5')
    model_resNet_grad_4 = tf.keras.models.load_model('ml_models/resNet_20_msec_17oct2020.h5')

    model_resNet_grad_5 = tf.keras.models.load_model('ml_models/resNet_20_msec_28oct2020.h5')

    #model_1d_cnn.summary()
    #model_2d_cnn.summary()
    #model_resNet_grad_2.summary()

    # split 1 second into 50, 20ms windows, and feed them into the models to see if there is a pop or now

    split_value = int(len(values_array) / (0.02 * sampling_rate))

    #print('split value is: ', split_value)

    for j in range(split_value):

        chunk = values_array[int(j * (0.02 * sampling_rate)):int((j + 1) * (0.02 * sampling_rate))]

        #print('chunk')
        #print(np.shape(chunk))

        if len(chunk) == 800:

            print ('making predictions...')

            # input_1d_cnn = np.expand_dims(np.asarray(chunk), axis=2)
            # input_1d_cnn = np.asarray(chunk)
            # input_1d_cnn = input_1d_cnn.reshape(-1, chunk.shape[0], 1)
            # input_1d_rnn = input_1d_cnn
            # input_2d_cnn = make_spectrogram_image_matrix(chunk,
            #                                              fig_size_x_in,
            #                                              fig_size_y_in,
            #                                              f_min,
            #                                              f_max)

            input_2d_cnn, x_dims, y_dims, channel_dims = make_spectrogram_image_matrix(chunk,
                                                         f_min,
                                                         f_max)

            #input_2d_cnn = np.asarray(input_2d_cnn)
            input_2d_cnn = input_2d_cnn.reshape(-1, y_dims, x_dims, channel_dims)
            #input_resNet = input_2d_cnn.reshape(-1, input_2d_cnn.shape[1], input_2d_cnn.shape[0], 1)


            input_resNet = input_2d_cnn

            # prediction_1d_cnn = model_1d_cnn.predict(input_1d_cnn)
            # prediction_2d_cnn = model_2d_cnn.predict(input_2d_cnn)
            # prediction_1d_rnn = model_1d_rnn.predict(input_1d_rnn)
            # prediction_resNet = model_resNet_v1.predict(input_resNet)
            #prediction_resNet_g = model_resNet_grad.predict(input_resNet)
            prediction_resNet_g_2 = model_resNet_grad_2.predict(input_resNet)
            prediction_resNet_g_3 = model_resNet_grad_3.predict(input_resNet)
            prediction_resNet_g_4 = model_resNet_grad_4.predict(input_resNet)

            # print('1d convo prediction:', prediction_1d_cnn)
            # print('2d convo prediction:', prediction_2d_cnn / -128)
            # print('1d rnn prediction:', prediction_1d_rnn)
            # print('resNet prediction:', prediction_resNet)
            #print('resNet, grad pred: ', prediction_resNet_g)
            #print('resNet, grad pred2: ', prediction_resNet_g_2)
            #print('resNet, grad pred3: ', prediction_resNet_g_3)
            #print('resNet, grad pred3: ', prediction_resNet_g_4)

            # stuff for new resNet,reshape input fft matrix

            spectrogram_matrix = make_fft_matrix(chunk, 40000, 50)
            y_dims = spectrogram_matrix.shape[0]
            x_dims = spectrogram_matrix.shape[1]

            input_resnet_new = np.array(spectrogram_matrix, dtype=np.uint8).reshape(-1,y_dims,x_dims,1)

            prediction_resNet_g_5 = model_resNet_grad_5.predict(input_resnet_new)


            #
            # if prediction_1d_cnn[0][0] > 0.9975:
            #     print('you got a ml peak 1d cnn')
            #
            #     # shade area
            #
            #     axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
            #                    time_array[int((j + 1) * (0.02 * sampling_rate))-1],
            #                    alpha=0.5,
            #                    color='red')
            #
            # prediction_2d_cnn_rescaled = prediction_2d_cnn[0][0] / -128
            #
            # print('2d convo prediction is: ' + str(prediction_2d_cnn_rescaled))
            #
            # if prediction_2d_cnn_rescaled > 0.45:
            #     print('you got a ml peak 2d cnn')
            #
            #     # shade area
            #
            #     axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
            #                    time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
            #                    alpha=0.5,
            #                    color='blue')
            #
            # if prediction_1d_rnn[0][0] > 0.985:
            #     print('you got a ml peak 1d rnn')
            #
            #     # shade area
            #
            #     axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
            #                    time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
            #                    alpha=0.5,
            #                    color='green')
            #
            # if prediction_resNet[0][0] > 0.98:
            #     print('you got a resnet peak')
            #
            #     # shade area
            #
            #     axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
            #                    time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
            #                    alpha=0.5,
            #                    color='yellow')

            # if prediction_resNet_g[0][0] > 0.997:
            #     print('you got a resnet_g peak')
            #
            #     # shade area
            #
            #     axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
            #                    time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
            #                    alpha=0.5,
            #                    color='pink')

            # if prediction_resNet_g_2[0][0] > 0.98:
            #     print('you got a resnet_g peak')
            #
            #     # shade area
            #
            #     axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
            #                    time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
            #                    alpha=0.5,
            #                    color='blue')
            #
            #     axs[2].plot(chunk, color='blue')

            if prediction_resNet_g_3[0][0] > 0.98:
                #print('you got a resnet_g V3 peak')

                # shade area

                axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
                               time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
                               alpha=0.5,
                               color='red')

                #axs[2].plot(chunk, color='red')

            if prediction_resNet_g_4[0][0] > 0.98:
                # print('you got a resnet_g V3 peak')

                # shade area

                axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
                                time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
                                alpha=0.5,
                                color='gold')

                #axs[2].plot(chunk, color='gold')

            if prediction_resNet_g_5[0][0] > 0.90:

                # shade area

                axs[0].axvspan(time_array[int(j * (0.02 * sampling_rate))],
                                time_array[int((j + 1) * (0.02 * sampling_rate) - 1)],
                                alpha=0.5,
                                color='purple')



        # run model on these chunk

    # prediction_1d_cnn = model_1d_cnn.predic

    plt.show()

    # spectrogram - scikit,
    # consider the window arg: window=signal.get_window(time_array,len(values_array),fftbins=True),
    # nperseg = int(20 * (sampling_rate / measured_time)),

    # try_window = signal.get_window('hann',n_FFT,fftbins=True)
    # print(try_window)
    #
    # print('using nFFT: ' + str(n_FFT) + ' , n_overlap: ' + str(n_overlap))

    # f_new, t_new, Sxx = signal.spectrogram(x=values_array,
    #                                        fs=sampling_rate,
    #                                        window=try_window,
    #                                        nperseg=n_FFT,
    #                                        noverlap=n_overlap,
    #                                        nfft=n_FFT,
    #                                        scaling='density',
    #                                        mode='psd')

    # f_new, t_new, Sxx = signal.spectrogram(x=values_array,
    #                                        fs=sampling_rate,
    #                                        nperseg=n_FFT,
    #                                        noverlap=n_overlap,
    #                                        nfft=n_FFT,
    #                                        )
    #
    #
    # axs[2].set_title('Spectrogram: Scipy function', fontsize=18, color='black')
    # axs[2].pcolormesh(t_new,f_new,Sxx, shading='gouraud')
    # axs[2].axis([0, measured_time, F_MIN, F_MAX])

    # plt.show()


# see if output plt.image file works:

# for z in range(10):
#     i = random.randrange(len(detectedPops))
#     make_spectrogram_image_matrix(detectedPops[i][5])
#     plt.imshow(make_spectrogram_image_matrix)

for z in range(20):
    i = random.randrange(len(detectedPops))

    print('look at: ' + str(i))

    plot_event(i, detectedPops, 'pop')

# now view non events

print('number of non pops detected: ' + str(len(notPops)))

# look at 20 arrays, pick randomly

for r in range(20):
    i = random.randrange(len(notPops))
    print('look at nonevent: ' + str(i))

    plot_event(i, notPops, 'non')

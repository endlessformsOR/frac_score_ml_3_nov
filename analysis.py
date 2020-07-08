import os
import sys
import subprocess
import datetime
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import matplotlib.transforms

import librosa
import db

plt.rcParams["figure.figsize"] = (20,6)
db.init()

def exp_avg(sample, avg, w):
    return w*sample + (1-w)*avg


def edge_detect(data, threshold):
    '''Detect edges that pass through a threshold value.'''
    fast_avg = data[0]
    slow_avg = data[0]
    prev_diff = 0
    fa = np.empty_like(data)
    sa = np.empty_like(data)
    edges = np.empty_like(data, dtype=bool)

    for i, sample in enumerate(data):
        fast_avg = exp_avg(sample, fast_avg, 0.25)
        fa[i] = fast_avg
        slow_avg = exp_avg(sample, slow_avg, 0.0625)
        sa[i] = slow_avg
        diff = abs(fast_avg - slow_avg)
        is_edge = prev_diff < threshold and diff >= threshold
        edges[i] = is_edge
        prev_diff = diff

    return fa, sa, np.flatnonzero(edges)


def windowed_average(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def windowed_max(arr, n):
    end =  n * int(len(arr)/n)
    return np.max(arr[:end].reshape(-1, n), 1)


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)


def plot_spectrogram(signal, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.seismic,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    fig.colorbar(cax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_statistics_and_filter(
    mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
    plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
    plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
    plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
    ax[0].set_title("Threshold for mask")
    ax[0].legend()
    cax = ax[1].matshow(smoothing_filter, origin="lower")
    fig.colorbar(cax)
    ax[1].set_title("Filter for smoothing Mask")
    plt.show()


def removeNoise(
    audio_clip,
    noise_clip,
    n_grad_freq=2,
    n_grad_time=4,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    n_std_thresh=1.5,
    prop_decrease=1.0,
    verbose=False,
    visual=False,
):
    """Remove noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    if verbose:
        start = time.time()
    # STFT over noise
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # convert to dB
    # Calculate statistics over noise
    mean_freq_noise = np.mean(noise_stft_db, axis=1)
    std_freq_noise = np.std(noise_stft_db, axis=1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    if verbose:
        print("STFT on noise:", td(seconds=time.time() - start))
        start = time.time()
    # STFT over signal
    if verbose:
        start = time.time()
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    if verbose:
        print("STFT on signal:", td(seconds=time.time() - start))
        start = time.time()
    # Calculate value to mask dB to
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    print(noise_thresh, mask_gain_dB)
    # Create a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    # calculate the threshold for each frequency/time bin
    db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
    # mask if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
    # convolve the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
    # mask the signal
    sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
    )  # mask real
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
        1j * sig_imag_masked
    )
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
    # recover the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
    if visual:
        plot_spectrogram(noise_stft_db, title="Noise")
    if visual:
        plot_statistics_and_filter(
            mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        )
    if visual:
        plot_spectrogram(sig_stft_db, title="Signal")
    if visual:
        plot_spectrogram(sig_mask, title="Mask applied")
    if visual:
        plot_spectrogram(sig_stft_db_masked, title="Masked signal")
    if visual:
        plot_spectrogram(recovered_spec, title="Recovered spectrogram")
    return recovered_signal


def low_pass_filter(xn, cutoff_freq, sampling_rate=40000, order=4):
    sos = signal.butter(order, cutoff_freq, btype='lowpass', fs=sampling_rate, output='sos')
    y = signal.sosfilt(sos, xn)
    return y


def high_pass_filter(xn, cutoff_freq, sampling_rate=40000, order=4):
    sos = signal.butter(order, cutoff_freq, btype='highpass', fs=sampling_rate, output='sos')
    y = signal.sosfilt(sos, xn)
    return y


def plot_spectrogram(times, freqs, spectrums, gamma=0.3):
    min_v = spectrums.min()
    max_v = spectrums.max()
    vmin = 0.0
    vmax = 2000.0
    fig, ax = plt.subplots(figsize=(24,4))
    #normalizer = colors.Normalize(vmin=vmin,vmax=vmax)
    #normalizer = colors.LogNorm(vmin=vmin, vmax=vmax)
    normalizer = colors.PowerNorm(gamma=gamma)
    pcm = ax.pcolormesh(times, freqs, spectrums,
                        norm=normalizer,
                        cmap='Spectral')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    fig.colorbar(pcm)
    plt.show()


def plot_spectrogram_b(signal, grid=True):
    fig, ax = plt.subplots(figsize=(20, 4))
    cax = ax.matshow(
        signal,
        origin="lower",
        aspect="auto",
        cmap=plt.cm.spectral,
        vmin=-1 * np.max(np.abs(signal)),
        vmax=np.max(np.abs(signal)),
    )
    plt.grid(grid)
    fig.colorbar(cax)


def spectrogram(xn, sampling_rate=40000):
    window_size = 1024
    overlap = window_size / 4
    scaling = 'spectrum' # 'density' or 'spectrum'
    mode = 'magnitude'
    freqs, times, spectrums = signal.spectrogram(xn, sampling_rate,
                                                 nperseg=window_size, noverlap=overlap,
                                                 return_onesided=True, scaling=scaling, mode=mode)
    return freqs, times, spectrums


def find_peaks(xn, min_prominence=30, min_spacing=2):
    peaks, props = signal.find_peaks(y, prominence=min_prominence, distance=min_spacing)
    return peaks


def plot_peaks(peaks, y):
    plt.plot(peaks, y[peaks], '.')


def plot_range_events(x, stages, title=None, edge_labels=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x)
    for i, stage in stages.iterrows():
        ax.axvline(stage['start'], color='green', linestyle=':')
        ax.axvline(stage['end'], color='red', linestyle=':')

        if edge_labels:
            trans = ax.get_xaxis_transform()
            plt.text(stage['start'], .2, stage['start'].strftime('%H:%M'),
                    transform=trans, rotation=-30,
                    bbox=dict(boxstyle="round", ec=(1., 1., 1.), fc=(1,1,1)))
            plt.text(stage['end'], .2, stage['end'].strftime('%H:%M'),
                    transform=trans, rotation=-30,
                    bbox=dict(boxstyle="round", ec=(1., 1., 1.), fc=(1,1,1)))

    # format, rotate, and align the tick labels so they look better
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    fig.autofmt_xdate()
    plt.grid()
    if title:
        plt.title(title)
    plt.show()


def pumpdown_length_histogram(pumpdowns):
    '''Plot a histogram of pumpdown lengths.'''
    pumpdown_lens = []
    for index, event in pumpdowns.iterrows():
        pumpdown_lens.append(event['end'] - event['start'])
    plt.figure(figsize=(8,4))
    plt.hist(pumpdown_lens)


def dynamic2dashboard(signal_array, time_in_seconds, time_array,sampling_rate):

    # determine sampling rate

    signal_array_list = signal_array.tolist()

    #sampling_rate = sampling_rate_int-1

    # for each second, find max

    x_list = []
    t_list = []

    for i in range(time_in_seconds):

        start = i*sampling_rate
        end = (i+1)*sampling_rate

        maxVal = np.amax(signal_array[start:end],axis=0)

        if i == time_in_seconds-1:
            maxVal = np.amax(signal_array[-sampling_rate:-1],axis=0)

        x_list.append(maxVal)
        t_list.append(i+1)

    x_out = np.asarray(x_list, dtype=np.float32)
    t_out = np.asarray(t_list, dtype=np.float32)

    return x_out, t_out


#Welch matrix method

def Welch_matrix_methods(signal_array, t_f,f_num,sampling_rate, seg_length):

    # find size of array

    x_vals, y_vals = signal.welch(signal_array,sampling_rate,nperseg=seg_length)

    f_vals = len(x_vals)

    #f_vals = int(seg_length/2 + 1)

    spectro_PSD = np.zeros((f_vals,t_f))
    spectro_LS = np.zeros((f_vals,t_f))

    for second in range(t_f):

        start = second*sampling_rate
        end = (second+1)*sampling_rate

        # Welch PSD

        x_psd, y_psd = signal.welch(signal_array[start:end],sampling_rate,nperseg=seg_length)

        y_psd_max = y_psd.max()

        # Welch LS

        x_ls, y_ls = signal.welch(signal_array[start:end],sampling_rate,'flattop',seg_length,scaling='spectrum')


        if second == t_f - 1:

            x_psd, y_psd = signal.welch(signal_array[start:end],sampling_rate,nperseg=seg_length)
            x_ls, y_ls = signal.welch(signal_array[start:end],sampling_rate,'flattop',seg_length,scaling='spectrum')

        spectro_PSD[:,second] = y_psd/y_psd_max
        spectro_LS[:,second] = y_ls

    return spectro_PSD, spectro_LS

#define FFT maker

def quickFFTobjs(xn, sampling_rate):

    # normalized with / max val
    N = len(xn)
    T = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1/(2*T)), int(N/2))
    y_fft = (2.0/N)*np.abs(yf[:N//2]) / max ((2.0/N)*np.abs(yf[:N//2]))
    return x_fft, y_fft

# LogFFT Spectrograph

def logFFT(xn, sampling_rate,f_num):

    # normalized with / max val
    N = len(xn)
    T = 1 / sampling_rate
    yf = sfp.fft(xn)
    x_fft = np.linspace(int(0), int(1/(2*T)), int(N/2))

    # normalized version
    #y_fft = (2.0/N)*np.abs(yf[:N//2]) / max ((2.0/N)*np.abs(yf[:N//2]))

    # non nomralized version
    y_fft = (2.0/N)*np.abs(yf[:N//2])

    # make log space of x output

    x_fft_loglog = np.logspace(1,4,f_num)
    y_fft_loglog = []

    # find corresponding x and y val from linear FFT

    for i in x_fft_loglog:

        idx = (np.abs(x_fft-i)).argmin()
        y_fft_loglog.append(y_fft[idx])

    np.asarray(y_fft_loglog,dtype=np.float32)
    np.asarray(x_fft_loglog,dtype=np.float32)

    # return normalized version
    y_fft_log_norm = (np.abs(y_fft_loglog)/max(np.abs(y_fft_loglog)))

    #test, return un nomalized
    #y_fft_log_norm = np.abs(y_fft_loglog)

    return x_fft_loglog, y_fft_log_norm

# LogFFT 2D matrix method

def logFFT2Dmatrix(signal_array, t_f,f_num,sampling_rate):

    spectroImage = np.zeros((f_num,t_f))

    for second in range(t_f):

        start = second*sampling_rate
        end = (second+1)*sampling_rate

        x_fft_log, y_fft_log = logFFT(signal_array[start:end],sampling_rate,f_num)

        if second == t_f - 1:

            x_fft_log, y_fft_log = logFFT(signal_array[start:end],sampling_rate,f_num)

        spectroImage[:,second] = y_fft_log

    return spectroImage

def fetch_sensor_db_data(sensor_id, start,end):
    assert end > start
    db.init()

    static_data = db.query_dataframe("""
    SELECT time,min,max,average,variance
    FROM monitoring.sensor_data
    WHERE sensor_id = %s
    AND time >= %s
    AND time <= %s
    ORDER BY time ASC
    """, (sensor_id, start,end))

    return static_data


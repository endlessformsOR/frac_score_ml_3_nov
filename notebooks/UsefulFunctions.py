## useful functions built by Jeff and Jon Origin Rose

## REFERNCES

import datetime
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.fft as fft
import scipy.fftpack as sfp

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook

import librosa
#import peakdetect
#import databases as db

## FUNCTIONS FOR API AND DB QUERY

def table_cols(table):
    return db.query(f"""SELECT *
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name   = '{table}';""")

def sensor_info(api):
    return db.query(f"""SELECT sensors.id
FROM wells
JOIN sensors ON sensors.well_id = wells.id
JOIN sensor_models ON sensors.sensor_model_id = sensor_models.id
WHERE api = '{api}' AND pressure_type = 'static'""")

def static_sensor_data(sensor_id, start_time, end_time):
    return db.query_dataframe(f"""SELECT max FROM sensor_data where sensor_id = '{sensor_id}' AND time BETWEEN '{start_time}' AND '{end_time}' ORDER BY time ASC""")

def name_to_api(name, number):
    return db.query(f"""select api from wells where wells.name like '%{name}%' AND number = '{number}'""")

## WindowAVG Methods

def windowed_average(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def windowed_max(arr, n):
    end =  n * int(len(arr)/n)
    return np.max(arr[:end].reshape(-1, n), 1)

##METHODS FOR LIBROSA AND STFFT

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


## FILTERING FUNCTIONS 


def high_pass_filter(xn, cutoff_freq, sampling_rate=40000):
    # Create a highpass butterworth filter at 150 Hz
    filter_order = 10
    sos = signal.butter(filter_order, cutoff_freq, btype='highpass', fs=sampling_rate, output='sos')
    y = signal.sosfilt(sos, xn)
    return y

#     size = len(times)
#     offset = 0.0
#     portion = 0.1
#     a = int(offset * size)
#     b = int((offset + portion) * size)
#     n = b - a
    
def plot_spectrogram_other(times, freqs, spectrums, gamma=0.3):
    min_v = spectrums.min()
    max_v = spectrums.max()
    vmin = 0.0
    vmax = 2000.0
    fig, ax = plt.subplots(figsize=(24,4))
    #normalizer = colors.Normalize(vmin=vmin,vmax=vmax)
    #normalizer = colors.LogNorm(vmin=vmin, vmax=vmax)
    normalizer = colors.PowerNorm(gamma=gamma)
    #temp fix to error
    a = 0
    b = len(times)
    n = len(spectrums)
    pcm = ax.pcolormesh(times[a:b], freqs, spectrums[:,:n],
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

### Saucedo methods - Using this to create a p(t) object from PATH 

def pressureTimeObject(PATH):
    
    # Upload .npz, convert to .npy
    dynamic_npys = []
    npy = np.load(PATH)
    dynamic_npys.append(npy['arr_0'])
    dynamic_data = np.concatenate(dynamic_npys)
    
    # define data conditions
    
    #sampling_rate = int(len(dynamic_data) / (60 * 60))
    sampling_rate = 43000
    #start = int(32.02 * 60 * sampling_rate)
    start = 0
    end = start + int(0.02 * 60 * sampling_rate)
    xn = dynamic_data[start:end]
    t = range(len(xn))
    return t, xn

def fftObject(PATH):

	# Upload .npz, convert to .npy
	dynamic_npys = []
	npy = np.load(PATH)
	dynamic_npys.append(npy['arr_0'])
	dynamic_data = np.concatenate(dynamic_npys)

	# define data conditions

	sampling_rate = int(len(dynamic_data) / (60 * 60))
	start = int(32.02 * 60 * sampling_rate)
	end = start + int(0.02 * 60 * sampling_rate)
	xn = dynamic_data[start:end]
	y = high_pass_filter(xn, 150)
	N = len(xn)
	T = N / sampling_rate
	yf = sfp.fft(y)
	x_fft = np.linspace(0.0, 1.0/(2.0*T), (N//2))
	y_fft = 2.0/N * np.abs(yf[:N//2])
	return x_fft, y_fft

def concatenateDynamicSignalsIntoArray(PATH_main,DATE,EXT,hours):

	dynamic_npys = []
	for hour in hours:
		path = PATH_main + "_" + DATE + "T" + str(hour) + EXT
		npy = np.load(path)
		dynamic_npys.append(npy['arr_0'])
	dynamic_data = np.concatenate(dynamic_npys)
	return dynamic_data

def concatArray2pressureTimeObj(PATH,DATE,EXT,hours):

	dynamic_data = concatenateDynamicSignalsIntoArray(PATH,DATE,EXT,hours)

	# define data conditions

	#sampling_rate = int(len(dynamic_data) / (60 * 60))

	sampling_rate = 43,000.0

	#start = int(32.02 * 60 * sampling_rate)
	#end = start + int(0.02 * 60 * sampling_rate)
	#xn = dynamic_data[start:end]
	#t = range(len(xn))
	#return t, xn

	xn = dynamic_data
	t = range(len(xn))
	return t, xn

def quickFFTobjects(sampling_rate, input_vals):

	N = len(input_vals)
	T = 1 / sampling_rate

	y = sfp.fft(input_vals)

	x_fft = np.linspace(int(0), int(1/(2*T)), int(N/2))

	# normalized output

	y_fft = (2.0/N)*np.abs(y[:N//2]) / max ((2.0/N)*np.abs(y[:N//2]))

	return x_fft, y_fft








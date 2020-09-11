import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm
import random

# open file 7000 pops, 5.2 GB

detectedPops = np.load('fracScore_allPads_08_09_pop_events.npy', allow_pickle=True)

# need to adjust, 5788 nonevents, filesize is 4.3 GB

notPops = np.load('fracScore_allPads_non_events10_09_2020_T02_10_22.npy', allow_pickle=True)

# set axis dims

fig_x = 25
fig_y = 5

# view detected pop events

ignoreEvents = []

sampling_rate = 40000

print('number of pops detected: ' + str(len(detectedPops)))

# look at 200 random events in list
for i in range(100):

    i = random.randrange(len(detectedPops))
    print('look at: ' + str(i))

    if any(i == j for j in ignoreEvents) == False: # move past any index elements in the ignoreEvents list

        WN = detectedPops[i][0]
        api = detectedPops[i][1]
        stage = detectedPops[i][2]
        start = detectedPops[i][3]
        stop = detectedPops[i][4]
        vals = detectedPops[i][5]

        outText = 'pop event: ' + str(i) + '\n'
        outText += 'WN: ' + WN + '\n'
        outText += 'API: ' + api + '\n'
        outText += 'start: ' + start + '\n'
        outText += 'stop: ' + stop + '\n'
        outText += 'stage: ' + str(stage) + '\n'

        measuredTime = len(vals)/sampling_rate

        timeVals = np.linspace(0,measuredTime,len(vals))

        # spectro details

        #
        # Optimal vals so far
        #
        #TIME_SAMPLE_WINDOW = .5
        #OVERLAP_FACTOR = 500


        # try to find better ones

        TIME_SAMPLE_WINDOW = .05
        OVERLAP_FACTOR = 50

        NFFT = int(sampling_rate*TIME_SAMPLE_WINDOW)  # 5ms window
        noverlap = int(sampling_rate*(TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

        CUTOFF_FREQ = 10
        # sepctro figure stuff

        F_MIN = 10
        F_MAX = 400

        COLORMAP = 'jet_r'



        # subplots stuff

        fig, axs = plt.subplots(2,1,figsize=(fig_x,fig_y))
        #fig.suptitle(outText)



        # Spectro, filtered
        axs[0].set_title('Spectrogram ', fontsize = 18, color='black')
        axs[0].specgram(vals, NFFT=NFFT, Fs=sampling_rate, noverlap=noverlap, cmap=pylab.get_cmap(COLORMAP))
        #axs[1].specgram(vals, NFFT=None, Fs=sampling_rate, noverlap=128, cmap=pylab.get_cmap(COLORMAP))
        #axs[0].set_title('Filtered, f < 1000 Hz')
        #axs[0].set_xlabel('Elapsed time, Seconds')
        axs[0].set_ylabel('Frequency in Hz')
        axs[0].axis([0,measuredTime,F_MIN,F_MAX])


        # 1st plot, dynamic signal

        axs[1].set_title('Dynamic Pressure Signal ', fontsize = 18, color='black')
        axs[1].plot(timeVals, vals, color='black', label = 'dyn, raw',zorder=1)
        #axs[1].plot(timeVals, dyn_fitrd_bp, color='red', label = 'dyn, bandpass filtered',zorder=1, linewidth=4)
        axs[1].axis([0 ,measuredTime, np.nanmin(vals), np.nanmax(vals)])
        axs[1].legend(loc='upper right')
        axs[1].text(1,0.6*min(vals),outText,fontsize=11,verticalalignment='center')
        #axs[1].text(int(measuredTime/2),0.6*min(vals),outText,fontsize=11,verticalalignment='center')
        #axs[1].set_xlabel('Elapsed time, Seconds')
        axs[1].set_ylabel('Rel. Mag.')

        plt.show()

# now view non events

print('number of non pops detected: ' + str(len(notPops)))

# look at first 20
for i in range(100):

    i = random.randrange(len(notPops))
    print('look at nonevent: ' + str(i))

    if any(i == j for j in ignoreEvents) == False: # move past any index elements in the ignoreEvents list

        WN = notPops[i][0]
        api = notPops[i][1]
        stage = notPops[i][2]
        start = notPops[i][3]
        stop = notPops[i][4]
        vals = notPops[i][5]

        outText = 'pop event: ' + str(i) + '\n'
        outText += 'WN: ' + WN + '\n'
        outText += 'API: ' + api + '\n'
        outText += 'start: ' + start + '\n'
        outText += 'stop: ' + stop + '\n'
        #outText += 'stage: ' + str(stage) + '\n'

        measuredTime = len(vals)/sampling_rate

        timeVals = np.linspace(0,measuredTime,len(vals))

        # spectro stuff

        TIME_SAMPLE_WINDOW = .05
        OVERLAP_FACTOR = 50

        NFFT = int(sampling_rate*TIME_SAMPLE_WINDOW)  # 5ms window
        noverlap = int(sampling_rate*(TIME_SAMPLE_WINDOW / OVERLAP_FACTOR))

        CUTOFF_FREQ = 10

        # subplots stuff

        fig, axs = plt.subplots(2,1,figsize=(fig_x,fig_y))
        #fig.suptitle(outText)



        # Spectro, filtered
        axs[0].set_title('Spectrogram ', fontsize = 18, color='black')
        axs[0].specgram(vals, NFFT=NFFT, Fs=sampling_rate, noverlap=noverlap, cmap=pylab.get_cmap(COLORMAP))
        #axs[1].specgram(vals, NFFT=None, Fs=sampling_rate, noverlap=128, cmap=pylab.get_cmap(COLORMAP))
        #axs[0].set_title('Filtered, f < 1000 Hz')
        #axs[0].set_xlabel('Elapsed time, Seconds')
        axs[0].set_ylabel('Frequency in Hz')
        axs[0].axis([0,measuredTime,F_MIN,F_MAX])


        # 1st plot, dynamic signal

        axs[1].set_title('Dynamic Pressure Signal ', fontsize = 18, color='black')
        axs[1].plot(timeVals, vals, color='black', label = 'dyn, raw',zorder=1)
        #axs[1].plot(timeVals, dyn_fitrd_bp, color='red', label = 'dyn, bandpass filtered',zorder=1, linewidth=4)
        axs[1].axis([0 ,measuredTime, np.nanmin(vals), np.nanmax(vals)])
        axs[1].legend(loc='upper right')
        axs[1].text(1,0.6*min(vals),outText,fontsize=11,verticalalignment='center')
        #axs[1].text(int(measuredTime/2),0.6*min(vals),outText,fontsize=11,verticalalignment='center')
        #axs[1].set_xlabel('Elapsed time, Seconds')
        axs[1].set_ylabel('Rel. Mag.')

        plt.show()

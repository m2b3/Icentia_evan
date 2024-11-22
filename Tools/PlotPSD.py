import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def plotRRIAndSpectrum(start, signal, annotations, fs, windowDuration, fig, axSignal, axSpectrum):
    
    ###########################
    t = np.arange(0, len(signal)//fs, 1/fs)
    
    end = start + windowDuration
    
    windowData = signal[int(start*fs):int(end*fs)]

    windowTime = t[int(start*fs):int(end*fs)]
    
    axSignal.clear()
    axSpectrum.clear()
    
    beatTotal = annotations["btype"]
    rythmTotal = annotations["rtype"]
    
    QBeatsIndex = np.array([beat for beat in beatTotal[0] if (beat/fs >= start and beat/fs < end)], dtype = int)
    NBeatsIndex = np.array([beat for beat in beatTotal[1] if (beat/fs >= start and beat/fs < end)], dtype = int)
    SBeatsIndex = np.array([beat for beat in beatTotal[2] if (beat/fs >= start and beat/fs < end)], dtype = int)
    aBeatsIndex = np.array([beat for beat in beatTotal[3] if (beat/fs >= start and beat/fs < end)], dtype = int)
    VBeatsIndex = np.array([beat for beat in beatTotal[4] if (beat/fs >= start and beat/fs < end)], dtype = int)
    
    axSignal.plot(windowTime, windowData, color='lightgray', alpha=0.75)
    
    ################


    nullRythmIndex = np.array([rythm for rythm in rythmTotal[0] if (rythm/fs >= start and rythm/fs < end)], dtype = int)
    endRythmIndex = np.array([rythm for rythm in rythmTotal[1] if (rythm/fs >= start and rythm/fs < end)], dtype = int)
    noiseRythmIndex = np.array([rythm for rythm in rythmTotal[2] if (rythm/fs >= start and rythm/fs < end)], dtype = int)
    NRythmIndex = np.array([rythm for rythm in rythmTotal[3] if (rythm/fs >= start and rythm/fs < end)], dtype = int)
    AFIBRythmIndex = np.array([rythm for rythm in rythmTotal[4] if (rythm/fs >= start and rythm/fs < end)], dtype = int)
    AFLRythmIndex = np.array([rythm for rythm in rythmTotal[5] if (rythm/fs >= start and rythm/fs < end)], dtype = int)

    

    #don't incude the split rythm

    axSignal.scatter(nullRythmIndex/fs, windowData[nullRythmIndex%(windowDuration*fs)], color='blue', label = "null", marker='x', s = 200)
    axSignal.scatter(endRythmIndex/fs, windowData[endRythmIndex%(windowDuration*fs)], color='orange', label = "end", marker='x', s = 200)
    axSignal.scatter(noiseRythmIndex/fs, windowData[noiseRythmIndex%(windowDuration*fs)], color='red', label = "noise", marker='x', s = 200)
    axSignal.scatter(NRythmIndex/fs, windowData[NRythmIndex%(windowDuration*fs)], color='green', label = "N", marker='x', s = 200)
    axSignal.scatter(AFIBRythmIndex/fs, windowData[AFIBRythmIndex%(windowDuration*fs)], color='purple', label = "AFIB", marker='x', s = 200)
    axSignal.scatter(AFLRythmIndex/fs, windowData[AFLRythmIndex%(windowDuration*fs)], color='black', label = "AFL", marker='x', s = 200)



    axSignal.plot(QBeatsIndex/fs, windowData[QBeatsIndex%(windowDuration*fs)], color='blue', label = "Q", marker='o', linestyle='')
    axSignal.plot(NBeatsIndex/fs, windowData[NBeatsIndex%(windowDuration*fs)], color='yellow', label = "N", marker='o', linestyle='')
    axSignal.plot(SBeatsIndex/fs, windowData[SBeatsIndex%(windowDuration*fs)], color='purple', label = "S", marker='o', linestyle='')
    axSignal.plot(aBeatsIndex/fs, windowData[aBeatsIndex%(windowDuration*fs)], color='red', label = "a", marker='o', linestyle='')
    axSignal.plot(VBeatsIndex/fs, windowData[VBeatsIndex%(windowDuration*fs)], color='green', label = "V", marker='o', linestyle='')




    ################
    
    axSignal.set_title(f'in {windowDuration}s Window')
    axSignal.set_xlabel('Time [s]')
    axSignal.set_ylabel('Amplitude (bpm)')
    axSignal.legend()
    
    n = len(windowData)
    freqs = np.fft.fftfreq(n, 1/fs)
    
    fft_values = np.fft.fft(windowData - np.mean(windowData), n = None, axis = -1, norm = None)    
    freq_mask = (freqs >= 0.04) & (freqs <= 0.15)
    filtered_freqs = freqs[freq_mask]
    filtered_fft_values = np.abs(fft_values[freq_mask])
        
    axSpectrum.plot(filtered_freqs, filtered_fft_values, color='black')
    axSpectrum.set_title('Power Spectrum (0.04-0.15 Hz)')
    axSpectrum.set_xlabel('Frequency [Hz]')
    axSpectrum.set_ylabel('Magnitude')
    
    fig.canvas.draw_idle()

def InteractiveWindowAndPSD(ecg, annotations, fs = 250, windowDuration = 300, initialStart = 0):
    t = np.arange(0, len(ecg)//fs, 1/fs)

    fig, (axSignal, axSpectrum) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, hspace=0.4)

    initial_start = 0
    plotRRIAndSpectrum(initialStart, ecg, annotations, fs, windowDuration, fig, axSignal, axSpectrum)

    ax_slider = plt.axes([0.1, 0.2, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Window \n start(s)', 0, t[-1] - windowDuration, valinit=initialStart)

    slider.on_changed(lambda val: plotRRIAndSpectrum(val, ecg, annotations, fs, windowDuration, fig, axSignal, axSpectrum))

    plt.show()

import os 
from scipy.signal import lombscargle, butter, filtfilt
from neurokit2.hrv.hrv_utils import _hrv_format_input
from neurokit2.hrv.intervals_process import intervals_process
import numpy as np
import neurokit2 as nk
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch


def findPeaks(annotations, N, beatTypesToExclude = []): 
    
    beats = annotations["btype"]
    beatTypes = ["Q", "N", "S", "a", "V"]
    
    if (not any((beatType in beatTypes) for beatType in beatTypesToExclude)) and len(beatTypesToExclude): 
        raise Exception("The types of beats to exclude are invalid, should be among Q, N, S, a, V")
        
    peaks = np.zeros(N, dtype = int)
    
    for beatType in list(set(beatTypes) - set(beatTypesToExclude)): 
        peaks[beats[beatTypes.index(beatType)]] = 1
    
    return peaks

def find_noise(annotations, N, beatTypesToExclude = ["N", "S", "a","V"]): 
    beats = annotations["btype"]
    beatTypes = ["Q", "N", "S", "a", "V"]
    
    # Validate excluded beat types
    if (not any((beatType in beatTypes) for beatType in beatTypesToExclude)) and len(beatTypesToExclude): 
        raise Exception("The types of beats to exclude are invalid, should be among Q, N, S, a, V")
        
    peaks = np.zeros(N, dtype = int)
    
    # Mark peaks for all beat types except those in beatTypesToExclude
    for beatType in list(set(beatTypes) - set(beatTypesToExclude)): 
        peaks[beats[beatTypes.index(beatType)]] = 1
    
    return peaks


def peakStartAndFinish(peaks): 
    return np.where(peaks)[0][0], np.where(peaks)[0][-1]
        
def RRI(peaks, fs):
    rri, rriTime, _ = _hrv_format_input(peaks, sampling_rate=fs)
    
    return rri, rriTime

def interpolateRRI(rri, rriTime, interpolationRate): 
    rriInterpolated, rriTimeInterpolated, samplingRate = intervals_process(
        rri, intervals_time=rriTime, interpolate=True, interpolation_rate=interpolationRate)
    return rriInterpolated, rriTimeInterpolated, samplingRate

def cutToWindows(rri, rriTime, interpolated, peakStart, peakEnd, annotations, fsRythm, interpolatedRateRRI = None, windowTime = 300, rythmTypesToExclude = ["End", "Noise", "AFIB", "AFL"]): 
       
    if interpolatedRateRRI is None: 
        interpolatedRateRRI = fsRythm
        
    rythmTypeExcludeInRRI = rythmExclusionRRI(len(rri), annotations, fsRythm, interpolatedRateRRI, rythmTypesToExclude)
    
    #new Interpolation rate is interpolatedRateRRI
        
    windows = []
    startTimes = []
    windowLength = interpolatedRateRRI * windowTime

    windowStart = peakStart

    while windowStart < peakEnd + 1 - windowLength: 
        if accept(rythmTypeExcludeInRRI, windowStart, windowLength): 
            if not(interpolated is None): 
                window = rri[windowStart:windowStart + windowLength], rriTime[windowStart:windowStart + windowLength]
            else: 
                rriWindow = [(time, beat) for (time, beat) in zip(rriTime, rri) if ((time*interpolatedRateRRI) >= windowStart and (windowStart+windowLength) >(time*interpolatedRateRRI))]
                windows = list(zip(*rriWindow))

            windows.append(window)
            startTimes.append(windowStart / interpolatedRateRRI)
            windowStart += windowLength
        else: 
            windowStart += 1

    return windows, startTimes

def rythmExclusionRRI(N, annotations, fsRythm, interpolatedRateRRI = None, rythmTypesToExclude = ["End", "Noise", "AFIB", "AFL"]): 
    if interpolatedRateRRI is None: 
        interpolatedRateRRI = fsRythm
    
    rythms = annotations["rtype"]
    rythmTypes = ["undefined", "end", "noise", "n", "afib", "afl"] #Don't include None
    
    if any((rythmType.lower() not in rythmTypes) for rythmType in rythmTypesToExclude): 
        raise Exception("The types of rythms to exclude are invalid, should be among Undefined, End, Noise, N, AFIB, AFL")
        
    rythmTypeExcludeInRRI = np.zeros(int(N*interpolatedRateRRI / fsRythm), dtype = int)
    
    for rythmType in rythmTypesToExclude: 
        
        #turn the time in the rythm array, sampled at fsRythm, to the same time as rri, sampled at 
        #interpolatedRateRRI
        
        rythmTypeExcludeInRRI[np.array(rythms[rythmTypes.index(rythmType.lower())]*interpolatedRateRRI/fsRythm).astype(int)] = 1
        
    return rythmTypeExcludeInRRI

def accept(rythmTypeExcludeInRRI, windowStart, windowLength): 
    return not any(rythmTypeExcludeInRRI[windowStart:windowStart + windowLength])


def PSD(rri, rriTimes, minFreq = 0.01, maxFreq = 0.5, numFreqPoints = 500, interpolationRate = 100, method = "fourier"): 
    if method.lower() == "lomb": 
        return lomb(rri, rriTimes, minFreq, maxFreq, numFreqPoints)
    elif method.lower() == "fourier" or method.lower() == "fft": 

        #NEED TO FILTER FIRST
        return FFT(rri, rriTimes, interpolationRate, minFreq, maxFreq)

    else: 
        raise Exception("Invalid PSD method")


def lomb(rri, rriTimes, minFreq = 0.01, maxFreq = 0.5, numFreqPoints = 500): 
    frequencies = np.linspace(minFreq, maxFreq, numFreqPoints)
    angularFrequencies = 2*np.pi * frequencies
    periodogram = lombscargle(rriTimes, rri, angularFrequencies)
    
    return frequencies, periodogram

def bandpass_filter(signal, fmin, fmax, fs, order=4):
    nyquist = 0.5 * fs
    low = fmin / nyquist
    high = fmax / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def FFT(rriInterpolated, interpolationRate = 100, minFreq = 0.04, maxFreq = 0.16): 
        
    n = len(rriInterpolated)
    freqs = np.fft.rfftfreq(len(rriInterpolated), 1/interpolationRate)
    
    fftValues = np.fft.rfft(rriInterpolated - np.mean(rriInterpolated))    
    freqMask = (freqs >= minFreq) & (freqs <= maxFreq)
    filteredFreqs = freqs[freqMask]
    filteredFFTValues = np.abs(fftValues[freqMask])

    
    return filteredFreqs, filteredFFTValues

def welchPSD(rriInterpolated, interpolationRate = 100, minFreq = 0.04, maxFreq = 0.16, windowTime = 200, overlapRatio = 0.5):
    nperseg = windowTime*interpolationRate  # Length of each segment
    noverlap = nperseg*overlapRatio  # Overlap between segments
    window = 'hann' 
    freqs, periodogram = welch(rriInterpolated, interpolationRate, window=window, nperseg=nperseg, noverlap=noverlap)
    fftValues = np.fft.rfft(rriInterpolated - np.mean(rriInterpolated))    
    freqMask = (freqs >= minFreq) & (freqs <= maxFreq)
    filteredFreqs = freqs[freqMask]
    filteredFFTValues = fftValues[freqMask]

    return filteredFreqs, filteredFFTValues
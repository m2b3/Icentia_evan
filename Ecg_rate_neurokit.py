import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import neurokit2 as nk
import pandas as pd
import gzip
import pickle
import multiprocessing as mp


def readSubject(subjectID, dataFolder): 
    """
    Reads the data from file

    parameters:
    - subjectID: id of subject
    - dataFolder: folder path

    output:
    - ECGTotal: an array of length 50, where each array contains the ecg_raw data
    """
    #neurokit only requires ecg data, ignoring the annotations
    #with gzip.open(f"{dataFolder}/{subjectID:05d}_batched.pkl.gz", "rb") as file: 
        #ECGTotal = (pickle.load(file))
        #ECGTotal = np.array(pickle.load(file))
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched_lbls.pkl.gz", "rb") as file: 
        peakDictTotal = np.array(pickle.load(file))

    return peakDictTotal

def ecg_rate(data_folder, subject, fs):
    """
    Generates ECG_rate for all sessions within a subject
    
    parameters:
    - data_folder: where to read the data
    - subject: subject_id
    - fs: sampling frequency

    output:
    - rate_total: dictionary with keys being the session_id, and values being the ecg_rate
                  also contains an array of removed session id
    - session_removed: an array of removed session id
    _ number_session_removed: # of sessions removed because of their quality
    """
    ecg_total= readSubject(subject, data_folder)
    # initialization for outputs
    rate_total = {}
    session_removed = []
    number_session_removed = 0

    #looping over all 50 sessions of subjects, computing its ecg_rate
    for session in range(len(ecg_total)):
        ecg_raw = np.float32(ecg_total[session])
        ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=250, method="neurokit")
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs)
        ecg_quality = nk.ecg_quality(ecg_clean,rpeaks=rpeaks["ECG_R_Peaks"], sampling_rate=fs, method = "zhao2018")
        
        # if the quality of ecg_clean is "unacceptable", then update the parameters and skip this session
        if ecg_quality == "Unacceptable":
            session_removed.append(session)
            number_session_removed += 1
            continue
        
        ecg_rate = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg_clean))
        rate_total[f"session_{session:02d}"]=ecg_rate
    
    rate_total["Removed sessions"] = session_removed
    rate_total["# sessions removed"] = number_session_removed
        
    return rate_total

def process_subject(subject_id):
    """
    takes the subject, apply ect_rate function and save the resulting result into a file
    """
    # here adjust data_folder to the correct directory
    # processes 50 sessions of one subject and save the ecg_rate in pkl.gzip files
    data_folder = '100data'
    fs = 250
    output={}
    try:
        subject_data = ecg_rate(data_folder, subject_id, fs)
        output[f"{subject_id:05d}_subject"] = subject_data
        dataframe = pd.DataFrame(output)
        dataframe.to_pickle(f'Test/{subject_id:05d}_ecg_rate.pkl.gz', compression='gzip') 
    
    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")

    


def process_all_subjects():
    """
    processes all subjects using parallel running with cpu assigned = 10
    """
    #looping over all 11000 files using parallel running
    subject_ids = range(100)

    with mp.Pool(processes=10) as pool:  # Adjust number of processes as needed
        pool.map(process_subject, subject_ids)


if __name__ == "__main__":
    #data_folder = os.getcwd()+"/icentia11k"
    #data_folder = os.getcwd()+"/Data"
    final_dataframe = process_all_subjects()
    
    

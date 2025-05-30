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
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched.pkl.gz", "rb") as file: 
        ECGTotal = np.array(pickle.load(file))
 
    return ECGTotal

def ecg_rate(data_folder, subject, fs):
    ecg_total= readSubject(subject, data_folder)
    rate_total = len(ecg_total), len(ecg_total)

    return rate_total

def process_subject(subject_id):
    """
    takes the subject, apply ect_rate function and save the resulting result into a file
    """
    
    data_folder = '100data'
    fs = 250
    output={}
    try:
        subject_data = ecg_rate(data_folder, subject_id, fs)
        output[f"{subject_id:05d}_subject"] = subject_data
        dataframe = pd.DataFrame(output)
        dataframe.to_pickle(f'local_test/{subject_id:05d}_ecg_rate.pkl.gz', compression='gzip') 

    except Exception as e:
        print(f"Error processing subject {subject_id}:")




def process_all_subjects():
    """
    processes all subjects using parallel running with cpu assigned = 10
    """
    #looping over all 11000 files using parallel running
    subject_ids = range(100)

    with mp.Pool(processes=10) as pool:  # Adjust number of processes as needed
        pool.map(process_subject, subject_ids)


if __name__ == "__main__":

    final_dataframe = process_all_subjects()
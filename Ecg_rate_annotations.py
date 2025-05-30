import numpy as np
import neurokit2 as nk
import pandas as pd
import gzip
import pickle
import multiprocessing as mp
from Tools.AnalysisTools import *
import warnings


def findPeaks(annotations, N, beatTypesToExclude = []): 
    
    beats = annotations["btype"]
    beatTypes = ["Q", "N", "S", "a", "V"]
    
    if (not any((beatType in beatTypes) for beatType in beatTypesToExclude)) and len(beatTypesToExclude): 
        raise Exception("The types of beats to exclude are invalid, should be among Q, N, S, a, V")
        
    peaks = np.zeros(N, dtype = int)
    
    for beatType in list(set(beatTypes) - set(beatTypesToExclude)): 
        peaks[beats[beatTypes.index(beatType)]] = 1
    
    return peaks

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
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched.pkl.gz", "rb") as file: 
        ECGTotal = np.array(pickle.load(file))
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched_lbls.pkl.gz", "rb") as file: 
        peakDictTotal = np.array(pickle.load(file))

    return ECGTotal, peakDictTotal

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
    ecg_total, annotations_total= readSubject(subject, data_folder)
    # initialization for outputs
    rate_total = {}
    session_removed = []
    number_session_removed = 0

    #looping over all 50 sessions of subjects, computing its ecg_rate
    for session in range(len(ecg_total)):
        try:
            ecg_raw = np.float32(ecg_total[session])
            annotation = (annotations_total)[session]
            ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=250, method="neurokit")

            peaks= findPeaks(annotation,len(ecg_clean))
            rpeaks = np.where(peaks == 1)[0]

            ecg_quality = nk.ecg_quality(ecg_clean,rpeaks=rpeaks, sampling_rate=fs, method = "zhao2018")
            
            # if the quality of ecg_clean is "unacceptable", then update the parameters and skip this session
            if ecg_quality == "Unacceptable":
                session_removed.append(session)
                number_session_removed += 1
                continue
            
            ecg_rate = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg_clean))
            rate_total[f"session_{session:02d}"]=ecg_rate

        except Warning as w:
            print(f'Warning processing subject_{subject:05d}_session_{session:02d}: {w}')

        except Exception as e:
            print(f'Error processing subject_{subject:05d}_session_{session:02d}: {e}')
    
    rate_total["Removed sessions"] = session_removed
    rate_total["# sessions removed"] = number_session_removed
        
    return rate_total

def process_subject(subject_id):
    """
    takes the subject, apply ect_rate function and save the resulting result into a file
    """
    # here adjust data_folder to the correct directory
    # processes 50 sessions of one subject and save the ecg_rate in pkl.gzip files
    data_folder = '/home/evan1/projects/rrg-skrishna/evan1/icentia11k'
    fs = 250
    output={}
    try:
        subject_data = ecg_rate(data_folder, subject_id, fs)
        output[f"{subject_id:05d}_subject"] = subject_data
        dataframe = pd.DataFrame(output)
        dataframe.to_pickle(f'/home/evan1/projects/rrg-skrishna/evan1/heart_rate_annotation/{subject_id:05d}_ecg_rate.pkl.gz', compression='gzip') 
    
    except Exception as a:
        print(f"Error processing subject {subject_id}: {a}")

    


def process_all_subjects():
    """
    processes all subjects using parallel running with cpu assigned = 10
    """
    #looping over all 11000 files using parallel running
    subject_ids = range(11000)

    with mp.Pool(processes=60) as pool:  # Adjust number of processes as needed
        pool.map(process_subject, subject_ids)


if __name__ == "__main__":
    #data_folder = os.getcwd()+"/icentia11k"
    #data_folder = os.getcwd()+"/Data"
    final_dataframe = process_all_subjects()
    

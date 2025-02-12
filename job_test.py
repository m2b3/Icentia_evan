import numpy as np
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
    
    data_folder = '/home/evan1/projects/rrg-skrishna/evan1/icentia11k'
    fs = 250
    output={}

    try:
        subject_data = ecg_rate(data_folder, subject_id, fs)
        output[f"{subject_id:05d}_subject"] = subject_data
        dataframe = pd.DataFrame(output)
        dataframe.to_pickle(f'scratch/evan_test/{subject_id:05d}_ecg_rate.pkl.gz', compression='gzip') 

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")

    


def process_all_subjects():
    """
    processes all subjects using parallel running with cpu assigned = 10
    """
    #looping over all 11000 files using parallel running
    subject_ids = range(11000)

    with mp.Pool(processes=40) as pool:  # Adjust number of processes as needed
        pool.map(process_subject, subject_ids)

if __name__ == "__main__":
<<<<<<< HEAD
    final_dataframe = process_all_subjects()
=======
    process_all_subjects()
>>>>>>> 46f89d50f8a11333452c8410219a5417b7312cdd
    

    
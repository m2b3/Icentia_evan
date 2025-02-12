import numpy as np
import neurokit2 as nk
import pandas as pd
import gzip
import pickle
import multiprocessing as mp
import warnings


with open("missing.csv", "r") as f:
    missing_subjects = [line.strip() for line in f.readlines()]
    missing = [int(i) for i in missing_subjects]

def readSubject(subjectID, dataFolder): 
    """ Reads ECG data from a file """
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched.pkl.gz", "rb") as file: 
        ECGTotal = np.array(pickle.load(file))
    return ECGTotal


def ecg_rate(data_folder, subject, fs):
    """ Generates ECG_rate for all sessions within a subject """
    
    warnings.filterwarnings("error")

    ecg_total = readSubject(subject, data_folder)
    
    # Initialize output
    rate_total = {}
    session_removed = []
    number_session_removed = 0

    for session in range(len(ecg_total)):
        try:
            ecg_raw = np.float32(ecg_total[session])
            ecg_clean = nk.ecg_clean(ecg_raw, sampling_rate=250, method="neurokit")
            _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs)

            if "ECG_R_Peaks" not in rpeaks or len(rpeaks["ECG_R_Peaks"]) == 0:
                raise ValueError(f"No R-peaks detected in subject_{subject:05d}_session_{session:02d}")
            
            ecg_quality = nk.ecg_quality(ecg_clean, rpeaks=rpeaks["ECG_R_Peaks"], sampling_rate=250, method="zhao2018")
            
            # Skip sessions with "Unacceptable" ECG quality
            if ecg_quality == "Unacceptable":
                session_removed.append(session)
                number_session_removed += 1
                continue
            
            ecg_rate = nk.signal_rate(rpeaks, sampling_rate=fs, desired_length=len(ecg_clean))
            rate_total[f"session_{session:02d}"] = ecg_rate

        except Warning as w:
            print(f' Warning: subject_{subject:05d}_session_{session:02d}: {w}')
        except Exception as e:
            print(f'Error: subject_{subject:05d}_session_{session:02d}: {e}')

    rate_total["Removed sessions"] = session_removed
    rate_total["# sessions removed"] = number_session_removed
        
    return rate_total


def process_subject(subject_id):
    """ Processes a subject and saves ECG rate """
    data_folder = '/home/evan1/projects/rrg-skrishna/evan1/icentia11k'
    fs = 250

    try:
        subject_data = ecg_rate(data_folder, subject_id, fs)
        output = {f"{subject_id:05d}_subject": subject_data}
        dataframe = pd.DataFrame(output)

        output_path = f'/home/evan1/projects/rrg-skrishna/evan1/heart_rate_neurokit/{subject_id:05d}_heart_rate.pkl.gz'
        dataframe.to_pickle(output_path, compression='gzip')

    except Exception as e:
        print(f" Error processing subject {subject_id}: {e}")


def process_all_missing_subjects():
    """ Runs only on missing files using multiprocessing """
    with mp.Pool(processes=60) as pool:  # Adjust based on system capabilities
        pool.map(process_subject, missing)


if __name__ == "__main__":
    process_all_missing_subjects()

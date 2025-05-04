import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from fooof import FOOOF
from sklearn.metrics import r2_score
from scipy import signal
from scipy.fft import rfft, rfftfreq
import re
import neurokit2 as nk
import gzip
import pickle


fs = 250

def readSubject(subjectID, dataFolder): 
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched.pkl.gz", "rb") as file: 
        ECGTotal = np.array(pickle.load(file))
    with gzip.open(f"{dataFolder}/{subjectID:05d}_batched_lbls.pkl.gz", "rb") as file: 
        peakDictTotal = np.array(pickle.load(file))

    return ECGTotal, peakDictTotal

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

def process_ecg_rate(ecg_rate, detrend, filter_enabled, highcut, fs):
    if detrend:
        ecg_rate = signal.detrend(ecg_rate)

    if filter_enabled:
        ecg_rate = nk.signal_filter(
            ecg_rate,
            highcut=highcut,
            sampling_rate=fs,
        )
    return ecg_rate

def sliding_window(data, window_time, overlap, noises, sampling_frequency=250):
    window_length = int(window_time * 60 * sampling_frequency)
    step_size = int(window_length * (1 - overlap))
    
    # Initialize dictionaries to hold windows with and without noise
    windows = {}
    removed_windows = {}
    segment = 0

    # Iterate over the data using the calculated step size and window length
    for start in range(0, len(data), step_size):
        window = data[start:start + window_length]
        noise_window = noises[start:start + window_length]
        
        if np.any(noise_window): 
            removed_windows[segment] = window
        else:
            windows[f'segment_{segment}'] = window
        
        segment += 1
    
    return windows, removed_windows

def calculate_differences(a_PSD,n_PSD):
    
    abs_differences = []
    for i in range (len(n_PSD)):
         _, a_fft = a_PSD[i]
         _, n_fft = n_PSD[i]
         a_fft = np.array(a_fft)
         n_fft = np.array(n_fft)

         differences = a_fft - n_fft
    
         abs_difference = np.sum(np.abs(differences))
         abs_differences.append(abs_difference)

    return(abs_differences)

def psd_and_hr(annotations_folder, neurokit_folder, subject, overlap, length, filter_threshold):
    annotation_rates = pd.read_pickle(f'{annotations_folder}/{subject:05d}_ecg_rate.pkl.gz', compression='gzip')
    neurokit_rates = pd.read_pickle(f'{neurokit_folder}/{subject:05d}_ecg_rate.pkl.gz', compression='gzip')

    neurokit_rows = neurokit_rates[f"{subject:05d}_subject"][:-2]
    annotation_rows = annotation_rates[f"{subject:05d}_subject"][:-2]

    a_session_names = annotation_rows.keys()
    n_session_names = neurokit_rows.keys()
    common_session_names = list(set(a_session_names) & set(n_session_names))

    neurokit_rows = {key: value for key, value in neurokit_rows.items() if key in common_session_names}
    annotation_rows = {key: value for key, value in annotation_rows.items() if key in common_session_names}

    session_numbers = sorted([int(re.search(r'\d+$', s).group()) for s in common_session_names if re.search(r'\d+$', s)])

    neurokit_rates = list(neurokit_rows.values())
    annotation_rates = list(annotation_rows.values())

    _, annotations = readSubject(subject, "100data")
    number_of_sessions = len(common_session_names)

    session_results_psd = {}  # Stores PSD data
    session_results_ecg = {}  # Stores ECG rate data

    for session in range(number_of_sessions):
        neurokit_rate = neurokit_rates[session]
        annotation_rate = annotation_rates[session]
        annotation = annotations[session_numbers[session]]

        noises = find_noise(annotation, len(neurokit_rate))
        a_windows, _ = sliding_window(annotation_rate, window_time=length, overlap=overlap, noises=noises)
        n_windows, _ = sliding_window(neurokit_rate, window_time=length, overlap=overlap, noises=noises)

        a_segmented_rates = list(a_windows.values())
        segment_ids = list(a_windows.keys())
        n_segmented_rates = list(n_windows.values())

        results_psd = {}  # PSD results
        results_ecg = {}  # ECG rate results

        for i in range(len(n_segmented_rates)):
            a_rate = a_segmented_rates[i]
            n_rate = n_segmented_rates[i]
            segment = segment_ids[i]

            if np.isnan(n_rate).any() or np.isnan(a_rate).any():
                continue
            try:
                a_rate_d = process_ecg_rate(a_rate, True, True, highcut=0.35, fs=fs)
                n_rate_d = process_ecg_rate(n_rate, True, True, highcut=0.35, fs=fs)

                a_freqs, a_fft = compute_fft(a_rate_d, fs)
                n_freqs, n_fft = compute_fft(n_rate_d, fs)

                valid_idx = (a_freqs > 0) & (a_freqs <= 20)
                freqs = a_freqs[valid_idx]
                a_fft = np.array(a_fft[valid_idx])
                n_fft = np.array(n_fft[valid_idx])

                differences = a_fft - n_fft
                abs_difference = np.sum(np.abs(differences))

                if abs_difference <= filter_threshold:
                    # Store PSD data
                    results_psd[segment] = {
                        "freqs": freqs,
                        "fft": a_fft
                    }

                    # Store corresponding ECG rate data
                    results_ecg[segment] = {
                        "ecg_rate_annotation": a_rate,
                        "ecg_rate_neurokit": n_rate
                    }

                    session_results_psd[f'Session_{session_numbers[session]:02d}'] = results_psd
                    session_results_ecg[f'Session_{session_numbers[session]:02d}'] = results_ecg

            except:
                print(f'Error processing (Subject_{subject:05d}_Session_{session_numbers[session]:02d}_Segment_{segment})')

    # Convert to DataFrames
    results_df_psd = pd.DataFrame(session_results_psd)
    results_df_ecg = pd.DataFrame(session_results_ecg)

    return results_df_psd, results_df_ecg  # Return both PSD and ECG rate data

def compute_fft(ecg_rate, fs):
    N = len(ecg_rate)
    fft_values = np.abs(rfft(ecg_rate))
    freqs = rfftfreq(N, 1 / fs)
    
   
    window_size = 8
    smooth_fft = np.convolve(fft_values, np.ones(window_size) / window_size, mode='same')
    return freqs, smooth_fft


def process_subject(subject_id, annotations_folder, neurokit_folder, overlap, length, filter_threshold, fs):
    """Process a single subject and return results as a DataFrame."""
    try:
        results, hr_results = psd_and_hr(annotations_folder=annotations_folder, 
                                          neurokit_folder=neurokit_folder, 
                                          subject=subject_id, 
                                          overlap=overlap, 
                                          length=length, 
                                          filter_threshold=filter_threshold)
        data = []

        for i in range(len(results.columns)):
            session = results.columns[i]
            hr_session = hr_results.columns[i]

            for j in range(len(results.index)):
                segment = results.index[j]
                hr_segment = hr_results.index[j]
                
                entry = results.at[segment, session]
                if isinstance(entry, dict) and 'freqs' in entry and 'fft' in entry:
                    hr_rate = hr_results.at[segment, session]['ecg_rate_annotation']
                    hr_mean = np.mean(hr_rate)
                    hr_var = np.var(hr_rate)

                    freqs = np.array(entry['freqs'])
                    power_spectrum = np.array(entry['fft'])

                    # FOOOF fit
                    fm = FOOOF(peak_width_limits=[0.02, 0.06], max_n_peaks=6, 
                                min_peak_height=0.19, peak_threshold=2.0, 
                                aperiodic_mode='knee')
                    fm.fit(freqs, power_spectrum, [0.01, 0.35])

                    # Extract R^2 values
                    r_squared_total = fm.r_squared_
                    r_squared_aperiodic = r2_score(fm.power_spectrum, fm._ap_fit)

                    # Extract aperiodic parameters: offset, knee, phi
                    offset, knee, phi = fm.aperiodic_params_

                    # Extract max peaks safely
                    peaks = fm.get_params('peak_params')
                    if isinstance(peaks, np.ndarray):
                        peaks = peaks.tolist()
                    if len(peaks) > 0:
                        max_peak_0_15 = max(
                            [peak for peak in peaks if isinstance(peak, (list, tuple)) and len(peak) >= 2 and 0.0 <= peak[0] <= 0.15], 
                            key=lambda x: x[1], 
                            default=None
                        )
                        max_peak_15_35 = max(
                            [peak for peak in peaks if isinstance(peak, (list, tuple)) and len(peak) >= 2 and 0.15 <= peak[0] <= 0.35], 
                            key=lambda x: x[1], 
                            default=None
                        )
                    else:
                        max_peak_0_15, max_peak_15_35 = None, None

                    data.append({
                        'id': f'{session}-{segment}',
                        'heart_rate': hr_rate,
                        'heart_rate_mean': hr_mean,
                        'heart_rate_var': hr_var,
                        'power_spectrum': power_spectrum.tolist(),  # Convert array to list for easier storage
                        'fooof_fit': {
                            'periodic': fm.get_params('peak_params'),
                            'aperiodic': fm.get_params('aperiodic_params')
                        },
                        'r_squared_values': {'r_squared_total': r_squared_total, 'r_squared_aperiodic': r_squared_aperiodic},
                        'phi': phi,
                        'offset': offset,
                        'max_peak 0 to 0.15': max_peak_0_15,
                        'max_peak 0.15 to 0.35': max_peak_15_35
                    })
        
        # Convert to DataFrame and save to .pkl file
        df = pd.DataFrame(data)
        result_path = os.path.join('fooof_results', f'subject_{subject_id:05d}_results.pkl')
        df.to_pickle(result_path)  # Save the DataFrame to a .pkl file
        print(f"Saved results for subject {subject_id} to {result_path}")

        return df  # Return the DataFrame for further processing if needed

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# --- Main script execution ---
if __name__ == "__main__":
    os.makedirs('fooof_results', exist_ok=True)

    #annotations_folder = '/home/evan1/projects/rrg-skrishna/evan1/heart_rate_annotation'
    #neurokit_folder = '/home/evan1/projects/rrg-skrishna/evan1/heart_rate_neurokit'

    annotations_folder = 'heart_rate_annotation'
    neurokit_folder = 'heart_rate_neurokit'

    subject_ids = range(20)  # Adjust the range as needed

    # Process subjects in parallel.
    with ProcessPoolExecutor(max_workers=5) as executor:  # Adjust the number of workers as needed
        futures = {executor.submit(process_subject, subj, annotations_folder, neurokit_folder, 0, 5, 2e5, fs): subj for subj in subject_ids}
        for future in as_completed(futures):
            subj = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Subject {subj} generated an exception: {exc}")

    print("Processing complete. All subject data saved in 'fooof_results' directory.")
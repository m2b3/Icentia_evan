import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from Tools.ReadPatientData import readSubject
from Tools.AnalysisTools import *
from numpy.fft import rfft, rfftfreq
from scipy.fft import rfft, rfftfreq
import scipy.signal as signal

from concurrent.futures import ProcessPoolExecutor

from fooof import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model

fs = 250

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

def compute_fft(ecg_rate, fs):
    N = len(ecg_rate)
    fft_values = np.abs(rfft(ecg_rate))
    freqs = rfftfreq(N, 1 / fs)
    
   
    window_size = 8
    smooth_fft = np.convolve(fft_values, np.ones(window_size) / window_size, mode='same')
    return freqs, smooth_fft

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

def save_psd(annotations_folder, neurokit_folder, subject, overlap,length, filter_threshold, plot=False):
    annotation_rates=pd.read_pickle(f'{annotations_folder}/{subject:05d}_ecg_rate.pkl.gz', compression='gzip')
    neurokit_rates=pd.read_pickle(f'{neurokit_folder}/{subject:05d}_ecg_rate.pkl.gz', compression='gzip')

    neurokit_rows = neurokit_rates[f"{subject:05d}_subject"][:-2]
    annotation_rows = annotation_rates[f"{subject:05d}_subject"][:-2]

    a_session_names = (annotation_rows.keys())
    n_session_names = (neurokit_rows.keys())
    common_session_names = list(set(a_session_names) & set(n_session_names))

    neurokit_rows = {key: value for key, value in neurokit_rows.items() if key in common_session_names}
    annotation_rows = {key: value for key, value in annotation_rows.items() if key in common_session_names}
   
    session_numbers = sorted([int(re.search(r'\d+$', s).group()) for s in common_session_names if re.search(r'\d+$', s)])

    neurokit_rates = list(neurokit_rows.values())
    annotation_rates = list(annotation_rows.values())

    _, annotations = readSubject(subject,"100data")
    number_of_sessions = len(common_session_names)

    session_results = {}
    for session in range(number_of_sessions):
        neurokit_rate = neurokit_rates[session]
        annotation_rate = annotation_rates[session]
        annotation = annotations[session_numbers[session]]

        noises = find_noise(annotation, len(neurokit_rate))
        a_windows, _ = sliding_window(annotation_rate,window_time=length, overlap=overlap, noises=noises)
        n_windows, n_removed_windows = sliding_window(neurokit_rate,window_time=length, overlap=overlap, noises=noises)

        a_segmented_rates = list(a_windows.values())
        segment_ids = list(a_windows.keys())

        n_segmented_rates = list(n_windows.values())

        results = {}
        for i in range(len(n_segmented_rates)):
            a_rate = a_segmented_rates[i]
            n_rate = n_segmented_rates[i]
            segment = segment_ids[i]

            if np.isnan(n_rate).any() or np.isnan(a_rate).any():
                continue
            try:
                a_rate_d = process_ecg_rate(a_rate, True, True, highcut = 0.35, fs=fs)
                n_rate_d = process_ecg_rate(n_rate, True, True, highcut = 0.35, fs=fs)

                a_freqs, a_fft = compute_fft(a_rate_d, fs)
                n_freqs, n_fft = compute_fft(n_rate_d, fs)

                a_fft = np.array(a_fft)
                n_fft = np.array(n_fft)
                differences = a_fft - n_fft
                abs_difference = np.sum(np.abs(differences))
                
                if abs_difference <= filter_threshold:
                    results[segment] = {
                        "freqs": a_freqs,
                        "fft": a_fft
                        }
                    
                    session_results[f'Session_{session_numbers[session]:02d}'] = results
        
                    if plot:
                        time = (np.arange(len(a_rate))/fs)/60
                        # Create a single figure with 4 subplots
                        plt.figure(figsize=(20, 10))

                        # Top-left subplot: (annotations) ECG Rate
                        plt.subplot(2, 2, 1)
                        plt.plot(time, a_rate, color='blue')
                        plt.xlabel('Time (minutes)')
                        plt.ylabel('ECG Rate')
                        plt.title(f'(annotations) ECG Rate (Subject_{subject:05d}_Session_{session_numbers[session]:02d}_Segment_{segment})')
                        plt.grid(True)

                        # Top-right subplot: (annotations) Power Spectral Density
                        plt.subplot(2, 2, 2)
                        plt.plot(a_freqs, a_fft, color='red')
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Magnitude')
                        plt.title(f'(annotations) Power Spectral Density (Subject_{subject:05d}_Session_{session_numbers[session]:02d}_Segment_{segment})')
                        plt.xlim(0, 0.35)
                        plt.grid(True)

                        # Bottom-left subplot: (neurokit) ECG Rate
                        plt.subplot(2, 2, 3)
                        plt.plot(time, n_rate, color='blue')
                        plt.xlabel('Time (minutes)')
                        plt.ylabel('ECG Rate')
                        plt.title(f'(neurokit) ECG Rate (Subject_{subject:05d}_Session_{session_numbers[session]:02d}_Segment_{segment})')
                        plt.grid(True)

                        # Bottom-right subplot: (neurokit) Power Spectral Density
                        plt.subplot(2, 2, 4)
                        plt.plot(n_freqs, n_fft, color='red')
                        plt.xlabel('Frequency (Hz)')
                        plt.ylabel('Magnitude')
                        plt.title(f'(neurokit) Power Spectral Density (Subject_{subject:05d}_Session_{session_numbers[session]:02d}_Segment_{segment})')
                        plt.xlim(0, 0.35)
                        plt.grid(True)
                        # Adjust layout to prevent overlapping
                        plt.tight_layout()

                        # Show the figure
                        plt.show()
                        # Print the absolute differences
                        print(f"total absolute differnces: {abs_difference}")
                    
                session_results[f"Session_{session_numbers[session]:02d}"] = results
            except:
                print(f'error proceesing (Subject_{subject:05d}_Session_{session_numbers[session]:02d}_Segment_{segment})')
            
                
        if plot:
            plt.figure(figsize=(16, 5))
            whole_time = (np.arange(len(neurokit_rate)) /fs)/ 60
            plt.plot(whole_time, neurokit_rate, color='blue', label='ECG_rate Data')

            for removed in n_removed_windows.keys():
                start = removed * (length * (1 - overlap))
                end = start + length
                plt.axvspan(start, end, color='red', alpha=0.3)
            
            plt.xlabel('Time (minutes)')
            plt.ylabel('ECG rate Data')
            plt.title(f'Entire Heart Rate Data (Subject_{subject:05d}, Session {session_numbers[session]}) with Removed Segments Highlighted')
            plt.legend()
            plt.grid(True)
            plt.show()
        
    
            
    results_df = pd.DataFrame(session_results)
    return results_df

def mayer_wave_counts(PSDs, show=False):
    fm = FOOOF(peak_width_limits=[0.02, 0.06], max_n_peaks=5, min_peak_height=0.2,
               peak_threshold=2.0, aperiodic_mode='knee')

    mayer_wave_range = (0.05, 0.15)
    total_valid_psds = 0
    mayer_wave_count = 0
    segments_id = []  # Store session and segment names

    for session in PSDs.columns:
        for segment in PSDs.index:
            entry = PSDs.at[segment, session]

            if isinstance(entry, dict):
                freqs = np.array(entry['freqs'])
                power_spectrum = np.array(entry['a_fft'])

                valid_idx = freqs > 0 
                freqs = freqs[valid_idx]
                power_spectrum = power_spectrum[valid_idx]

                if len(freqs) > 0 and len(power_spectrum) > 0:
                    fm.fit(freqs, power_spectrum, [0.01, 0.35])
                    total_valid_psds += 1


                    for peak in fm.get_params('peak_params'):
                        peak = np.array(peak)
                        if np.isnan(peak).any():
                            continue
                        peak_freq = peak[0]
                        if mayer_wave_range[0] <= peak_freq <= mayer_wave_range[1]:
                            mayer_wave_count += 1
                            segments_id.append((session, segment))
                            break  # Count only once per PSD

    if show:                   
        print(f"Total non-NaN PSDs: {total_valid_psds}")
        print(f"Number of PSDs containing Mayer waves: {mayer_wave_count}")
        print("Sessions and Segments with Mayer waves:")
        for session, segment in segments_id:
            print(f"Session: {session}, Segment: {segment}")

    return total_valid_psds, mayer_wave_count, segments_id

def process_subject(subject_id):
    """ Function to process each subject in parallel. """
    try:
        result = save_psd(annotations_folder='ECG_rate_annotations', neurokit_folder="Test", subject=subject_id, overlap=0, length=5, filter_threshold=2e5, plot=False)
        total_psd_num, mayer_wave_num, segments_id = mayer_wave_counts(PSDs=result, show=False)
        return subject_id, total_psd_num, mayer_wave_num, segments_id  
        
    except:
        raise RuntimeError(f"Error processing Subject_{subject_id:05d}")

     

if __name__ == "__main__":
    subject_ids = list(range(100))  # Define subject IDs

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_subject, subject_ids))

    # Convert results into a DataFrame
    results_df = pd.DataFrame(results, columns=['Subject_ID', 'Total_PSDs', 'Mayer_Wave_PSDs', 'Segments_ID'])
    
    # Save to CSV
    results_df.to_csv("mayer_wave_results.csv", index=False)

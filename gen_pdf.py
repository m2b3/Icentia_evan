import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from Tools.ReadPatientData import readSubject
from Tools.AnalysisTools import *
from numpy.fft import rfft, rfftfreq
import random
import scipy.signal as signal
from concurrent.futures import ProcessPoolExecutor
from matplotlib.backends.backend_pdf import PdfPages
from fooof_result import FOOOF
from fooof.sim.gen import gen_power_spectrum
from fooof.sim.utils import set_random_seed
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model
from sklearn.metrics import r2_score

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

def save_psd(annotations_folder, neurokit_folder, subject, overlap, length, filter_threshold):
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



def apply_fooof(freqs, psd, freq_range =[0, 0.35]):
    """Apply FOOOF model to a given power spectrum."""
    fm = FOOOF(peak_width_limits=[0.03, 0.05], max_n_peaks=5, min_peak_height=0.2,
               peak_threshold=2.0, aperiodic_mode='knee')
    fm.fit(freqs, psd, freq_range)
    return fm
'''
## for selection
def process_subject(subject_id):
    """Process a single subject, ensuring at least 12 segments with PSDs."""
    try:
        # Extract PSD results for the subject
        results_df, heart_rate_df = save_psd(annotations_folder='heart_rate_annotation',
                              neurokit_folder="heart_rate_neurokit",
                              subject=subject_id, overlap=0, length=5,
                              filter_threshold=2e5)

        if results_df.empty:
            print(f"No valid data for Subject_{subject_id:05d}")
            return None

        # Randomly select sessions until at least 12 segments are obtained
        selected_segments = []
        available_sessions = list(results_df.columns)
        random.shuffle(available_sessions)

        while available_sessions and len(selected_segments) < 6:
            session = available_sessions.pop()
            session_data = results_df[session].dropna().to_dict()  # Convert to dictionary
            for segment, data in session_data.items():
                if len(selected_segments) >= 12:
                    break
                data["session"] = session
                data["segment"] = segment
                selected_segments.append(data)

        if len(selected_segments) < 6:
            print(f"Not enough valid segments for Subject_{subject_id:05d}")
            return None

        return subject_id, selected_segments[:6]

    except Exception as e:
        print(f"Error processing Subject_{subject_id:05d}: {e}")
        return None'''
'''
def process_subject(subject_id):
    """Process a single subject, ensuring at least 12 segments with PSDs."""
    try:
        # Extract PSD results for the subject
        results_df, heart_rate_df = save_psd(annotations_folder='heart_rate_annotation',
                              neurokit_folder="heart_rate_neurokit",
                              subject=subject_id, overlap=0, length=5,
                              filter_threshold=2e5)

        if results_df.empty:
            print(f"No valid data for Subject_{subject_id:05d}")
            return None

        # Randomly select sessions until at least 12 segments are obtained
        selected_segments = []
        heart_rate_segments = []
        available_sessions = list(results_df.columns)
        random.shuffle(available_sessions)

        while available_sessions and len(selected_segments) < 6:
            session = available_sessions.pop()
            session_data = results_df[session].dropna().to_dict()  # Convert to dictionary
            session_hr_data = heart_rate_df[session].dropna().to_dict()  # Heart rate data

            for segment, data in session_data.items():
                if len(selected_segments) >= 6:
                    break

                psd_data = session_data[segment]
                hr_data = session_hr_data[segment]


                psd_data["session"] = session
                psd_data["segment"] = segment
                hr_data["session"] = session
                hr_data["segment"] = segment

                selected_segments.append(psd_data)
                heart_rate_segments.append(hr_data)
        

        if len(selected_segments) < 6:
            print(f"Not enough valid segments for Subject_{subject_id:05d}")
            return None

        return subject_id, selected_segments[:6], heart_rate_segments[:6]

    except Exception as e:
        print(f"Error processing Subject_{subject_id:05d}: {e}")
        return None
'''
import random

import random

def process_subject(subject_id):
    """Process a single subject, ensuring at least 6 segments with PSDs and heart rate data."""
    try:
        # Extract PSD and heart rate results for the subject
        results_df, heart_rate_df = save_psd(annotations_folder='heart_rate_annotation',
                                             neurokit_folder="heart_rate_neurokit",
                                             subject=subject_id, overlap=0, length=5,
                                             filter_threshold=2e5)

        if results_df.empty:
            print(f"No valid data for Subject_{subject_id:05d}")
            return None

        # Randomly select sessions until at least 6 segments are obtained
        selected_segments = []
        selected_heart_rates = []
        available_sessions = list(results_df.columns)
        random.shuffle(available_sessions)

        while available_sessions and len(selected_segments) < 6:
            session = available_sessions.pop()
            
            # Convert session data to dictionaries
            session_psd_data = results_df[session].dropna().to_dict()  # PSD data
            session_hr_data = heart_rate_df[session].dropna().to_dict()  # Heart rate data

            for segment in session_psd_data.keys():
                if len(selected_segments) >= 6:
                    break

                # Retrieve PSD and heart rate data for this segment
                psd_data = session_psd_data[segment]
                hr_data = session_hr_data.get(segment, None)  # Get HR data safely

                # Attach session & segment identifiers
                psd_data["session"] = session
                psd_data["segment"] = segment

                if hr_data:
                    hr_data["session"] = session
                    hr_data["segment"] = segment

                selected_segments.append(psd_data)
                selected_heart_rates.append(hr_data)

        if len(selected_segments) < 6:
            print(f"Not enough valid segments for Subject_{subject_id:05d}")
            return None

        return subject_id, selected_segments[:6], selected_heart_rates[:6]

    except Exception as e:
        print(f"Error processing Subject_{subject_id:05d}: {e}")
        return None




# Main function to process subjects and generate a single PDF
def generate_pdf_for_all_subjects(subject_ids, output_pdf="fooof_analysis.pdf"):
    with PdfPages(output_pdf) as pdf:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_subject, subject_ids))

        for result in results:
            if result is None:
                continue

            subject_id, selected_segments, heart_rate_segments = result
            fig, axes = plt.subplots(6, 2, figsize=(15, 30))  # 4 rows, 3 columns (12 plots per page)
            axes = axes.flatten()  # Convert 2D axes array to a 1D list

            for i, segment in enumerate(selected_segments):
                freqs = np.array(segment["freqs"])
                heart_rate_segment = heart_rate_segments[i]
                power_spectrum = np.array(segment["fft"])

                # Apply frequency cutoff at 0.35 Hz
        

                fm = apply_fooof(freqs, power_spectrum, freq_range=[0.001, 0.5])
                if fm is None:
                    continue  # Skip invalid segments
                r_squared = fm.r_squared_
                offset, knee, phi= fm.aperiodic_params_
                aper_r = r2_score(fm.power_spectrum, fm._ap_fit)
                peaks = fm.get_params('peak_params')
                #max_peak_0_15 = max([peak for peak in peaks if 0.0 <= peak[0] <= 0.15], key=lambda x: x[1], default=None)
                #max_peak_15_35 = max([peak for peak in peaks if 0.15 <= peak[0] <= 0.35], key=lambda x: x[1], default=None)


                ax_psd = axes[i*2]  # Get the subplot
                fm.plot(ax = ax_psd, add_legend=False)
                ax_psd.set_title(f"Subject {subject_id:05d}, {segment['session']}, {segment['segment']}")
                ax_psd.text(0.05, 0.05, f'R² = {r_squared:.3f}', 
                        transform=ax_psd.transAxes, fontsize=12, verticalalignment='bottom')
                ax_psd.text(0.05, 0.10, f'R² aperiodic = {aper_r:.3f}',
                        transform=ax_psd.transAxes, fontsize=12, verticalalignment='bottom')
                ax_psd.text(0.05, 0.15, f'offset = {offset:.3f}, phi = {phi:.3f}',
                        transform=ax_psd.transAxes, fontsize=12, verticalalignment='bottom')
                #ax_psd.text(0.05, 0.20, f'max_peak_0 to 0.15 = {max_peak_0_15}, max_peak 0.15 to 0.35 = {max_peak_15_35}',
                        #transform=ax_psd.transAxes, fontsize=12, verticalalignment='bottom')
                
                # heart rate plot
                ax_hr = axes[i * 2 + 1]  # Every odd index (1, 3, 5, ...)
                time = np.arange(len(heart_rate_segment["ecg_rate_annotation"])) / fs 

                ax_hr.plot(time, heart_rate_segment["ecg_rate_neurokit"], color="blue")

                ax_hr.set_title(f"Heart Rate - Subject {subject_id:05d}, {segment['session']}, {segment['segment']}")
                ax_hr.set_xlabel("Time (seconds)")
                ax_hr.set_ylabel("ECG Rate")
                ax_hr.legend()
                ax_hr.grid(True)

            plt.tight_layout()
            pdf.savefig(fig)  # Save the whole figure (all 12 plots)
            plt.close(fig)

# Run the script
if __name__ == "__main__":
    subject_ids = list(range(100))  # Adjust as needed
    generate_pdf_for_all_subjects(subject_ids, "fooof_analysis.pdf")
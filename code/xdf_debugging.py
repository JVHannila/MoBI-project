"""
XDF Debugging Script

A standalone utility to load a single XDF file, inspect its stream contents,
and visualize the EEG data against its markers to diagnose potential issues
like corruption, timing drift, or missing streams.
"""

import os
import os.path as op
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import mne

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- EDIT THIS SECTION ---
# Define the relative path from the project root to the file you want to debug.
FILE_TO_DEBUG = op.join('sourcedata', 'pilot', 'sub-P01', 'brain', 'sub-P01_task-treadmill-walk-fast_eeg.xdf')
# -------------------------

# Automatically determine the project root directory.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(script_dir)
except NameError:
    PROJECT_ROOT = os.getcwd() # Fallback for interactive use

# ==============================================================================

def inspect_xdf_file(file_path):
    """
    Loads, inspects, and visualizes a single XDF file for debugging.
    """
    print("=" * 70)
    print(f"🔬 Starting debug session for: {op.basename(file_path)}")
    print("=" * 70)

    if not op.exists(file_path):
        print(f"\n[!!] ERROR: File not found at the specified path:\n{file_path}")
        return

    # --- 1. Attempt to load the file with pyxdf ---
    print("\n--- Step 1: Attempting to load XDF file ---")
    try:
        streams, header = pyxdf.load_xdf(file_path)
        print("[OK] File loaded successfully with pyxdf.")
    except Exception as e:
        print(f"[!!] CRITICAL: pyxdf failed to load the file.")
        print(f"     Error: {e}")
        print("     This indicates the file is likely corrupted.")
        return # Cannot continue if loading fails

    # --- 2. Inspect the loaded streams ---
    print("\n--- Step 2: Inspecting Stream Contents ---")
    print(f"Found {len(streams)} streams in the file.")
    eeg_stream = None
    marker_stream = None
    for i, stream in enumerate(streams):
        stream_name = stream['info']['name'][0]
        stream_type = stream['info']['type'][0]
        n_channels = int(stream['info']['channel_count'][0])
        sfreq = float(stream['info']['nominal_srate'][0])
        n_samples = len(stream['time_stamps'])
        
        print(f"\n  Stream {i+1}:")
        print(f"    Name: {stream_name}")
        print(f"    Type: {stream_type}")
        print(f"    Channels: {n_channels}")
        print(f"    Sampling Freq: {sfreq} Hz")
        print(f"    Num. Samples/Timestamps: {n_samples}")

        # Identify the primary EEG and Marker streams for later use
        if 'eeg' in stream_type.lower() and eeg_stream is None:
            eeg_stream = stream
            print("    * Identified as primary EEG stream.")
        if 'marker' in stream_type.lower() and marker_stream is None:
            marker_stream = stream
            print("    * Identified as primary Marker stream.")
            
    if not eeg_stream or not marker_stream:
        print("\n[!!] WARNING: Could not find both an EEG and a Marker stream.")
        print("     Visualization and MNE conversion will be skipped.")
        return

    # --- 3. Visualize Data and Markers ---
    print("\n--- Step 3: Generating Data Visualization ---")
    try:
        eeg_data = eeg_stream['time_series']
        eeg_stamps = eeg_stream['time_stamps']
        marker_stamps = marker_stream['time_stamps']
        
        # Normalize timestamps to start from 0
        first_stamp = eeg_stamps[0]
        eeg_stamps_rel = eeg_stamps - first_stamp
        marker_stamps_rel = marker_stamps - first_stamp
        
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Plot the first 5 EEG channels
        channels_to_plot = min(5, eeg_data.shape[1])
        ax.plot(eeg_stamps_rel, eeg_data[:, :channels_to_plot])
        
        # Plot markers as vertical lines
        for marker_t in marker_stamps_rel:
            ax.axvline(marker_t, color='red', linestyle='--', alpha=0.6, label='Marker' if 'Marker' not in ax.get_legend_handles_labels()[1] else "")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (arbitrary units)")
        ax.set_title(f"Debug Plot: First {channels_to_plot} EEG Channels and Markers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        print("[OK] Plot generated. Close the plot window to continue.")
        plt.show()

    except Exception as e:
        print(f"[!!] ERROR: Could not generate plot. Reason: {e}")


    # --- 4. Attempt MNE Raw Object Creation (Simplified) ---
    print("\n--- Step 4: Attempting to create MNE Raw object ---")
    try:
        info = mne.create_info(
            ch_names=[ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']],
            sfreq=float(eeg_stream['info']['nominal_srate'][0]),
            ch_types='eeg'
        )
        raw = mne.io.RawArray(eeg_stream['time_series'].T, info)
        print("[OK] MNE RawArray object created successfully.")
        print(raw)
    except Exception as e:
        print(f"[!!] ERROR: Failed to create MNE RawArray.")
        print(f"     Reason: {e}")

    print("\n" + "="*70)
    print("Debug session finished.")
    print("="*70)


if __name__ == '__main__':
    full_path = op.join(PROJECT_ROOT, FILE_TO_DEBUG)
    inspect_xdf_file(full_path)
"""
XDF to BIDS Conversion Script (Final Version 3.1)

This script provides a robust pipeline for converting multiple XDF files into a
BIDS-compliant dataset. It automatically separates EEG and motion data,
dynamically creates montages from XDF metadata, and cleanly parses event markers.
"""
import re
import os
import os.path as op
import mne
import pyxdf
import numpy as np
from mne_bids import (
    BIDSPath,
    print_dir_tree,
    write_raw_bids,
    get_anonymization_daysback,
    make_report
)
from mne_bids.stats import count_events

# ==============================================================================
# CONFIGURATION (No changes needed)
# ==============================================================================
subjects = ['P01', 'P02', 'P03']
tasks = [
    'natural-walk', 'sitting-eyes-closed', 'sitting-eyes-open',
    'treadmill-walk-comfortable', 'treadmill-walk-fast'
]
exclude_files = {'P01': ['treadmill-walk-fast']}
task_variations = {
    'sitting-eyes-closed': ['sitting-eyes-closed-before', 'sitting-eyes-closed-after'],
    'sitting-eyes-open': ['sitting-eyes-open-before', 'sitting-eyes-open-after'],
    'treadmill-walk-fast': ['treadmill-walk-fast-1']
}
session = '01'
anonymize_data = False
daysback_buffer = 2117

# ==============================================================================
# DIRECTORY SETUP (No changes needed)
# ==============================================================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
project_root = os.path.dirname(script_dir)
data_dir = op.join(project_root, 'sourcedata', 'pilot')
bids_root = op.join(project_root, 'data', 'BIDS')

print(f"Project root: {project_root}")
print(f"Source data directory: {data_dir}")
print(f"BIDS output directory: {bids_root}")
print(f"Anonymization: {'Enabled' if anonymize_data else 'Disabled'}")
print("=" * 60)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def find_xdf_file(subject_dir, subject_id, task_name):
    """Finds an XDF file for a given subject and task, including variations."""
    primary_filename = f"sub-{subject_id}_task-{task_name}_eeg.xdf"
    primary_path = op.join(subject_dir, primary_filename)
    if op.exists(primary_path):
        return primary_path, task_name
    
    if task_name in task_variations:
        for variation in task_variations[task_name]:
            variation_filename = f"sub-{subject_id}_task-{variation}_eeg.xdf"
            variation_path = op.join(subject_dir, variation_filename)
            if op.exists(variation_path):
                return variation_path, variation
    
    return None, None

def convert_task_name_for_bids(task_name):
    """Converts a hyphenated task name to BIDS-compliant PascalCase."""
    return "".join(word.capitalize() for word in task_name.split('-'))

def _create_montage_from_xdf(eeg_stream, eeg_ch_names):
    """
    Parses channel locations from XDF stream and creates an MNE DigMontage,
    but only for channels intended to be EEG.
    """
    ch_coords = {}
    channels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
    
    for ch in channels_info:
        label = ch.get('label', [None])[0]
        if label in eeg_ch_names:
            loc = ch.get('location', [None])[0]
            if label and loc and all(k in loc for k in ['X', 'Y', 'Z']):
                try:
                    x = float(loc['X'][0]) / 1000.0
                    y = float(loc['Y'][0]) / 1000.0
                    z = float(loc['Z'][0]) / 1000.0
                    ch_coords[label] = np.array([x, y, z])
                except (ValueError, IndexError):
                    print(f"      WARNING: Could not parse coordinates for channel '{label}'")
    
    if not ch_coords:
        return None

    return mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='head')


def load_and_prepare_raw(xdf_path):
    """
    Loads, separates, and prepares streams from a single XDF file.
    """
    try:
        streams, header = pyxdf.load_xdf(xdf_path)
        
        eeg_stream, markers_stream = None, None
        for stream in streams:
            stream_type = stream['info']['type'][0].lower()
            if 'eeg' in stream_type:
                eeg_stream = stream
            elif 'markers' in stream_type:
                markers_stream = stream

        if not eeg_stream or not markers_stream:
            print(f"      ERROR: Missing EEG or Marker stream in {op.basename(xdf_path)}.")
            return None

        sfreq = float(eeg_stream['info']['effective_srate']) or float(eeg_stream['info']['nominal_srate'][0])
        
        all_ch_names = [ch['label'][0] for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']]
        eeg_data = eeg_stream['time_series'].T
        eeg_timestamps = eeg_stream['time_stamps']
        
        motion_ch_names = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'QuatW', 'QuatX', 'QuatY', 'QuatZ']
        eeg_ch_names = [name for name in all_ch_names if name not in motion_ch_names]
        eeg_indices = [all_ch_names.index(name) for name in eeg_ch_names]
        
        t_start, t_end = markers_stream['time_stamps'][0], markers_stream['time_stamps'][-1]
        start_idx = np.searchsorted(eeg_timestamps, t_start, side='left')
        end_idx = np.searchsorted(eeg_timestamps, t_end, side='right')
        
        cropped_eeg_data = eeg_data[eeg_indices, start_idx:end_idx]
        cropped_eeg_timestamps = eeg_timestamps[start_idx:end_idx]
        eeg_start_time = cropped_eeg_timestamps[0]
        
        if np.max(np.abs(cropped_eeg_data)) > 1e-3:
            cropped_eeg_data *= 1e-6

        info = mne.create_info(ch_names=eeg_ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(cropped_eeg_data, info)
        raw.info['line_freq'] = 50

        montage = _create_montage_from_xdf(eeg_stream, eeg_ch_names)
        if montage:
            raw.set_montage(montage)
            print("      Applied montage to EEG channels from XDF metadata.")
        
        # --- **FINAL IMPROVEMENT**: Cleanly parse event markers ---
        marker_timestamps = markers_stream['time_stamps']
        marker_data = markers_stream['time_series']
        event_pattern = re.compile(r'<ecode>(\d+)</ecode>') # Regex to find the event code
        
        onsets, durations, descriptions = [], [], []
        max_time = raw.times[-1]
        for mt, mv in zip(marker_timestamps, marker_data):
            onset = mt - eeg_start_time
            if 0 <= onset <= max_time:
                desc_str = str(mv[0])
                match = event_pattern.search(desc_str)
                description = f"Event_{match.group(1)}" if match else desc_str
                
                onsets.append(onset)
                durations.append(0.0)
                descriptions.append(description)

        annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(annotations)
        print(f"      Added {len(annotations)} valid annotations.")

        return raw
        
    except Exception as e:
        print(f"      ERROR loading {op.basename(xdf_path)}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==============================================================================
# MAIN PROCESSING WORKFLOW (No changes needed)
# ==============================================================================
def main():
    """Executes the main XDF to BIDS conversion workflow."""
    
    print("Step 1: Collecting and loading all XDF files...")
    raw_list, bids_list, file_info = [], [], []
    
    for subject_id in subjects:
        print(f"\nProcessing Subject: {subject_id}\n" + "-" * 40)
        
        subject_dir = op.join(data_dir, f'sub-{subject_id}', 'brain')
        if not op.exists(subject_dir):
            print(f"  WARNING: Subject directory not found: {subject_dir}")
            continue
        
        for task_name in tasks:
            print(f"  Checking task: {task_name}")
            
            if subject_id in exclude_files and task_name in exclude_files.get(subject_id, []):
                print(f"      INFO: Skipping excluded file.")
                continue
            
            xdf_path, actual_task = find_xdf_file(subject_dir, subject_id, task_name)
            if not xdf_path:
                print(f"      File not found for task: {task_name}")
                continue
            
            raw = load_and_prepare_raw(xdf_path)
            if raw is None:
                continue
            
            bids_task_name = convert_task_name_for_bids(actual_task)
            bids_path = BIDSPath(
                subject=subject_id, session=session, task=bids_task_name,
                datatype='eeg', root=bids_root
            )
            
            raw_list.append(raw)
            bids_list.append(bids_path)
            file_info.append({
                'subject': subject_id, 'task': actual_task,
                'bids_task': bids_task_name, 'file': op.basename(xdf_path)
            })
            print(f"    [OK] Ready for BIDS conversion: {op.basename(xdf_path)}")
    
    print(f"\nCollected {len(raw_list)} files for BIDS conversion.")
    if not raw_list:
        print("No files found to process. Exiting.")
        return
    
    anonymize_dict = None
    if anonymize_data:
        print("\nStep 2: Calculating anonymization parameters...")
        try:
            daysback_min, _ = get_anonymization_daysback(raw_list)
            anonymize_dict = dict(daysback=daysback_min + daysback_buffer)
            print(f"Anonymization: daysback = {anonymize_dict['daysback']} days")
        except Exception as e:
            print(f"Warning: Could not calculate anonymization parameters: {e}")
    
    print(f"\nStep 3: Converting {len(raw_list)} files to BIDS format...")
    successful_conversions, failed_conversions = 0, 0
    
    for i, (raw, bids_path) in enumerate(zip(raw_list, bids_list)):
        file_info_item = file_info[i]
        print(f"\nConverting {i+1}/{len(raw_list)}: {file_info_item['file']}")
        try:
            write_raw_bids(
                raw, bids_path, anonymize=anonymize_dict,
                overwrite=True, allow_preload=True, 
                format='BrainVision', verbose=False
            )
            print(f"  [OK] SUCCESS: {bids_path.basename}")
            successful_conversions += 1
        except Exception as e:
            print(f"  [!!] ERROR: {str(e)}")
            failed_conversions += 1
    
    print("\nStep 4: Generating dataset summary...\n" + "=" * 60 + "\nCONVERSION SUMMARY\n" + "=" * 60)
    
    total_to_process = len(raw_list)
    if total_to_process > 0:
        success_rate = successful_conversions / total_to_process * 100 if total_to_process > 0 else 0
        print(f"Total files processed: {total_to_process}\nSuccessful conversions: {successful_conversions}\n"
              f"Failed conversions: {failed_conversions}\nSuccess rate: {success_rate:.1f}%")
    else:
        print("No files were processed.")

    if successful_conversions > 0:
        print(f"\nBIDS dataset created at: {bids_root}")
        if op.exists(bids_root):
            print("\nBIDS directory structure:")
            print_dir_tree(bids_root)
        
        try:
            print("\nEvent Statistics:")
            counts = count_events(bids_root)
            print(counts if not counts.empty else "No events found.")
        except Exception as e:
            print(f"Could not generate event statistics: {e}")
        
        try:
            print("\nDataset Report:\n" + "-" * 40)
            print(make_report(root=bids_root))
        except Exception as e:
            print(f"Could not generate dataset report: {e}")
    
    print("\nConversion completed!")


if __name__ == "__main__":
    main()
"""
XDF to BIDS Group Conversion Script - Improved Version

This script converts multiple XDF files to BIDS format using MNE-Python and MNE-BIDS,
following the official MNE-BIDS group conversion workflow for better robustness and 
consistency.


"""

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

# ====================================================================
#                           CONFIGURATION
# ====================================================================

# List of subjects to process (without 'sub-' prefix)
subjects = ['P01', 'P02', 'P03']

# List of tasks to process (without 'task-' prefix)
tasks = [
    'natural-walk',
    'sitting-eyes-closed', 
    'sitting-eyes-open',
    'treadmill-walk-comfortable',
    'treadmill-walk-fast'
]

# Additional task variations due to piloting phase
task_variations = {
    'sitting-eyes-closed': ['sitting-eyes-closed-before', 'sitting-eyes-closed-after'],
    'sitting-eyes-open': ['sitting-eyes-open-before', 'sitting-eyes-open-after'],
    'treadmill-walk-fast': ['treadmill-walk-fast-1']
}

# Session identifier
session = '01'

# Montage file path
montage_filename = 'prox64_montagedig.fif'

# Anonymization settings (set to False to disable anonymization)
anonymize_data = False
daysback_buffer = 2117  # Additional days to add for anonymization safety

# ====================================================================
#                         DIRECTORY SETUP
# ====================================================================

# Set up project directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = op.join(project_root, 'sourcedata', 'pilot')
bids_root = op.join(project_root, 'data', 'BIDS')
montage_path = op.join(project_root, 'utility', montage_filename)

# Check if source directory exists
if not op.exists(data_dir):
    raise FileNotFoundError(f"Source data directory not found: {data_dir}")

print(f"Project root: {project_root}")
print(f"Source data directory: {data_dir}")
print(f"BIDS output directory: {bids_root}")
print(f"Montage file: {montage_path}")
print(f"Anonymization: {'Enabled' if anonymize_data else 'Disabled'}")
print("=" * 60)

# ====================================================================
#                         UTILITY FUNCTIONS
# ====================================================================

def find_xdf_file(subject_dir, subject_id, task_name):
    """Find XDF file for a given subject and task, checking for variations."""
    primary_filename = f"sub-{subject_id}_task-{task_name}_eeg.xdf"
    primary_path = op.join(subject_dir, primary_filename)
    
    if op.exists(primary_path):
        return primary_path, task_name
    
    # Check for task variations
    if task_name in task_variations:
        for variation in task_variations[task_name]:
            variation_filename = f"sub-{subject_id}_task-{variation}_eeg.xdf"
            variation_path = op.join(subject_dir, variation_filename)
            if op.exists(variation_path):
                return variation_path, variation
    
    return None, None

def convert_task_name_for_bids(task_name):
    """Convert task name to BIDS-compatible format."""
    parts = task_name.split('-')
    return ''.join(word.capitalize() for word in parts)

def load_and_prepare_raw(xdf_path, montage_path):
    """
    Load XDF file and create MNE Raw object with montage and annotations.
    
    Returns:
    - MNE Raw object if successful, None if failed
    """
    try:
        # Load XDF file
        streams, header = pyxdf.load_xdf(xdf_path)
        
        # Find EEG and markers streams
        eeg_stream = None
        markers_stream = None
        
        for stream in streams:
            stream_type = stream['info']['type'][0].lower()
            stream_name = stream['info']['name'][0].lower()
            
            if 'eeg' in stream_type or 'eeg' in stream_name:
                eeg_stream = stream
            elif 'marker' in stream_type or 'marker' in stream_name:
                markers_stream = stream
        
        if not eeg_stream:
            print(f"    ERROR: No EEG stream found in {op.basename(xdf_path)}")
            return None
        
        # Extract EEG information
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
        ch_labels_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        ch_names = [ch['label'][0] for ch in ch_labels_info]
        
        # Create MNE Raw object
        eeg_data = eeg_stream['time_series'].T
        eeg_timestamps = eeg_stream['time_stamps']
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info)
        
        # Apply montage if available
        if op.exists(montage_path):
            try:
                montage = mne.channels.read_dig_fif(montage_path)
                raw.set_montage(montage)
                print(f"    Applied montage: {op.basename(montage_path)}")
            except Exception as e:
                print(f"    Warning: Could not apply montage: {e}")
        
        # Process markers
        if markers_stream:
            marker_data = markers_stream['time_series']
            marker_timestamps = markers_stream['time_stamps']
            
            annotations_onset = []
            annotations_duration = []
            annotations_description = []
            
            eeg_start_time = eeg_timestamps[0]
            
            for marker_time, marker_value in zip(marker_timestamps, marker_data):
                onset = marker_time - eeg_start_time
                
                if isinstance(marker_value, (list, np.ndarray)) and len(marker_value) > 0:
                    description = str(marker_value[0])
                else:
                    description = str(marker_value)
                
                annotations_onset.append(onset)
                annotations_duration.append(0.0)
                annotations_description.append(description)
            
            annotations = mne.Annotations(
                onset=annotations_onset,
                duration=annotations_duration,
                description=annotations_description
            )
            raw.set_annotations(annotations)
            print(f"    Added {len(annotations)} annotations")
        
        print(f"    Loaded: {op.basename(xdf_path)} ({sfreq} Hz, {len(ch_names)} channels)")
        return raw
        
    except Exception as e:
        print(f"    ERROR loading {op.basename(xdf_path)}: {str(e)}")
        return None

# ====================================================================
#                       MAIN PROCESSING WORKFLOW
# ====================================================================

def main():
    """Main processing function following MNE-BIDS group conversion workflow."""
    
    # Step 1: Build lists of raw objects and BIDS paths (MNE-BIDS best practice)
    print("Step 1: Collecting and loading all XDF files...")
    raw_list = []
    bids_list = []
    file_info = []  # Track file information for summary
    
    for subject_id in subjects:
        print(f"\nProcessing Subject: {subject_id}")
        print("-" * 40)
        
        subject_dir = op.join(data_dir, f'sub-{subject_id}', 'brain')
        
        if not op.exists(subject_dir):
            print(f"  WARNING: Subject directory not found: {subject_dir}")
            continue
        
        for task_name in tasks:
            print(f"  Checking task: {task_name}")
            
            xdf_path, actual_task = find_xdf_file(subject_dir, subject_id, task_name)
            
            if xdf_path is None:
                print(f"    File not found: sub-{subject_id}_task-{task_name}_eeg.xdf")
                continue
            
            # Load and prepare raw data
            raw = load_and_prepare_raw(xdf_path, montage_path)
            
            if raw is None:
                print(f"    Failed to load: {op.basename(xdf_path)}")
                continue
            
            # Create BIDS path
            bids_task_name = convert_task_name_for_bids(actual_task)
            bids_path = BIDSPath(
                subject=subject_id,
                session=session,
                task=bids_task_name,
                datatype='eeg',
                root=bids_root
            )
            
            # Add to lists
            raw_list.append(raw)
            bids_list.append(bids_path)
            file_info.append({
                'subject': subject_id,
                'task': actual_task,
                'bids_task': bids_task_name,
                'file': op.basename(xdf_path)
            })
            
            print(f"    ✓ Ready for BIDS conversion: {op.basename(xdf_path)}")
    
    print(f"\nCollected {len(raw_list)} files for BIDS conversion")
    
    if len(raw_list) == 0:
        print("No files found to process. Exiting.")
        return
    
    # Step 2: Calculate anonymization parameters (if enabled)
    anonymize_dict = None
    if anonymize_data:
        print("\nStep 2: Calculating anonymization parameters...")
        try:
            daysback_min, daysback_max = get_anonymization_daysback(raw_list)
            anonymize_dict = dict(daysback=daysback_min + daysback_buffer)
            print(f"Anonymization: daysback = {anonymize_dict['daysback']} days")
        except Exception as e:
            print(f"Warning: Could not calculate anonymization parameters: {e}")
            anonymize_dict = None
    
    # Step 3: Convert all files to BIDS format
    print(f"\nStep 3: Converting {len(raw_list)} files to BIDS format...")
    successful_conversions = 0
    failed_conversions = 0
    
    for i, (raw, bids_path) in enumerate(zip(raw_list, bids_list)):
        file_info_item = file_info[i]
        print(f"\nConverting {i+1}/{len(raw_list)}: {file_info_item['file']}")
        
        try:
            write_raw_bids(
                raw, 
                bids_path, 
                anonymize=anonymize_dict,
                overwrite=True, 
                allow_preload=True, 
                format='BrainVision'
            )
            print(f"  ✓ SUCCESS: sub-{file_info_item['subject']}_ses-{session}_task-{file_info_item['bids_task']}_eeg")
            successful_conversions += 1
            
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            failed_conversions += 1
    
    # Step 4: Generate dataset summary and report
    print("\nStep 4: Generating dataset summary...")
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(raw_list)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {successful_conversions/len(raw_list)*100:.1f}%")
    
    if successful_conversions > 0:
        print(f"\nBIDS dataset created at: {bids_root}")
        
        # Show directory structure
        if op.exists(bids_root):
            print("\nBIDS directory structure:")
            try:
                print_dir_tree(bids_root)
            except Exception as e:
                print(f"Could not display directory tree: {e}")
        
        # Generate event counts (if any events exist)
        try:
            print("\nEvent Statistics:")
            counts = count_events(bids_root)
            if not counts.empty:
                print(counts)
            else:
                print("No events found in the dataset")
        except Exception as e:
            print(f"Could not generate event statistics: {e}")
        
        # Generate dataset report
        try:
            print("\nDataset Report:")
            print("-" * 40)
            dataset_report = make_report(root=bids_root)
            print(dataset_report)
        except Exception as e:
            print(f"Could not generate dataset report: {e}")
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main()
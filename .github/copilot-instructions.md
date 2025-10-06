# MoBI-project AI Coding Instructions

## Project Overview
This is a Mobile Brain/Body Imaging (MoBI) research project that processes EEG data recorded during various motor tasks (walking, sitting, treadmill) using the PROX-64 EEG cap. The workflow converts XDF recordings to BIDS format and provides comprehensive preprocessing tools.

## Architecture & Data Flow
- **sourcedata/pilot/**: Raw XDF files organized by subject (`sub-P01/brain/`, `sub-P02/brain/`, etc.)
- **data/BIDS/**: Converted BIDS-compliant dataset with BrainVision format (.vhdr, .eeg, .vmrk)
- **data/preprocessed/**: Filtered and cleaned data ready for analysis
- **utility/**: Custom PROX-64 montage file (`prox64_montagedig.fif`)
- **code/**: Processing scripts and interactive notebooks

## Key Workflows

### 1. XDF to BIDS Conversion
- **Single file**: Use `xdf2bids.ipynb` for individual conversions
- **Batch processing**: Use `xdf2bids_group.py` for multiple subjects/tasks
- Always applies PROX-64 montage automatically from `utility/prox64_montagedig.fif`
- Preserves XML markers as BIDS events with timing precision

### 2. Data Preprocessing Pipeline
The `bids_basic_preprocessing.ipynb` follows this sequence:
1. Load BIDS data with `read_raw_bids()`
2. Automatic bad channel detection (flat, extreme amplitude, high variance)
3. Interactive visual inspection with MNE browser
4. Manual annotation of artifacts (movement, eye blinks, muscle)
5. Filtering: notch (50Hz), high-pass (0.1Hz), low-pass (100Hz)
6. Save to `data/preprocessed/` with summary reports

### 3. Task Naming Conventions
- **Source XDF**: `sub-P01_task-natural-walk_eeg.xdf`
- **BIDS format**: `sub-P01_ses-01_task-NaturalWalk_eeg.vhdr`
- Task variations handled: `sitting-eyes-closed-before/after`, `treadmill-walk-fast-1`

## Project-Specific Patterns

### EEG Hardware Configuration
- **64-channel PROX cap** with custom spherical coordinate mapping
- **250Hz sampling rate** standard across recordings
- **XDF format** with separate EEG, impedance, and marker streams
- Line frequency: **50Hz** (European standard)

### MNE-Python Integration
```python
# Standard montage application
montage = mne.channels.read_dig_fif('utility/prox64_montagedig.fif')
raw.set_montage(montage)

# BIDS path construction
bids_path = BIDSPath(subject='P01', session='01', task='NaturalWalk', 
                     datatype='eeg', root='data/BIDS')
```

### Movement Task Considerations
- Walking tasks require **movement artifact detection**
- Use amplitude-based detection when magnetometer data unavailable
- Apply `annotate_movement()` for EEG-specific movement artifacts
- 99th percentile threshold for high-amplitude period detection

### File Organization Requirements
- Never commit raw data (`.xdf`, `.eeg`, `.fif` files in `.gitignore`)
- Maintain BIDS validator compliance
- Session structure: always `ses-01` for current dataset
- Preserve original XDF timestamps in BIDS events

## Development Environment
- **Python environment**: `mne` conda environment required
- **Core dependencies**: MNE-Python, MNE-BIDS, pyxdf, matplotlib
- **Qt backend**: Use `%gui qt` in notebooks for interactive plots
- **Windows PowerShell**: Default shell for terminal commands

## Critical Implementation Details

### XDF Stream Detection
```python
# Dynamic stream identification (don't hardcode indices)
for stream in streams:
    stream_type = stream['info']['type'][0].lower()
    if 'eeg' in stream_type or 'eeg' in stream['info']['name'][0].lower():
        eeg_stream = stream
```

### BIDS Conversion Best Practices
- Use `write_raw_bids()` with `format='BrainVision'` for compatibility
- Include `allow_preload=True` for memory-loaded data
- Generate events.tsv automatically from XDF markers
- Set `overwrite=True` during development iterations

### Quality Control Checks
- **Flat channels**: `np.std(data) < 1e-7`
- **Extreme amplitude**: `np.max(np.abs(data)) > 500e-6`
- **High variance outliers**: `> 3x 95th percentile`
- Always compare original vs. filtered data visually

## Common Operations
- **Interactive plotting**: Use MNE browser with `raw.plot()` for channel inspection
- **PSD analysis**: `raw.compute_psd(fmax=100).plot()` for frequency domain
- **Annotation workflow**: Manual marking → automatic detection → combined cleanup
- **Filter verification**: Always plot before/after comparisons

When modifying preprocessing parameters or adding new analysis methods, maintain compatibility with the BIDS structure and preserve the existing quality control workflow.
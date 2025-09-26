import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_prox64_montage():
    """
    Creates an MNE-Python DigMontage for the PROX-64 cap based on the
    spherical coordinates provided by the manufacturer. 
    The spherical coordinates (theta, phi) are transcribed from the
    documents provided by the manufacturer.

    The conversion from the provided spherical system to MNE's Cartesian
    system assumes:
    - Theta is the angle from posterior (+90) to anterior (-90).
    - Phi is the angle from left (-90) to right (+90).
    - We convert this to a standard polar system for Cartesian conversion.
    
    Returns:
        mne.channels.DigMontage: The MNE montage object for the PROX-64 cap.
    """
    
    # Step 1: Transcribe the coordinates from the sheet into a dictionary.
    # Format: { 'Channel_Name': (Theta_degrees, Phi_degrees) }
    coords_deg = {
        'Fp1': (-90, -72), 'Fp2': (90, 72), 'F3': (-60, -51),
        'F4': (60, 51), 'C3': (-45, 0), 'C4': (45, 0),
        'P3': (-60, 51), 'P4': (60, -51), 'O1': (-90, 72),
        'O2': (90, -72), 'F7': (-90, -36), 'F8': (90, 36),
        'T7': (-90, 0), 'T8': (90, 0), 'P7': (-90, 36),
        'P8': (90, -36), 'AFz': (67, 90), 'Fz': (45, 90),
        'Cz': (0, 0), 'Pz': (45, -90), 'FC1': (-31, -46),
        'FC2': (31, 46), 'CP1': (-31, 46), 'CP2': (31, -46),
        'FC5': (-69, -21), 'FC6': (69, 21), 'CP5': (-69, 21),
        'CP6': (69, -21), 'FT9': (-113, -18), 'FT10': (113, 18),
        'TP7': (-90, 18), 'TP8': (90, -18), 'F1': (-49, -68),
        'F2': (49, 68), 'C1': (-23, 0), 'C2': (23, 0),
        'P1': (-49, 68), 'P2': (49, -68), 'AF3': (-74, -68),
        'AF4': (74, 68), 'FC3': (-49, -29), 'FC4': (49, 29),
        'CP3': (-49, 29), 'CP4': (49, -29), 'PO3': (-74, 68),
        'PO4': (74, -68), 'F5': (-74, -41), 'F6': (74, 41),
        'C5': (-68, 0), 'C6': (68, 0), 'P5': (-74, 41),
        'P6': (74, -41), 'AF7': (-90, -54), 'AF8': (90, 54),
        'FT7': (-90, -18), 'FT8': (90, 18), 'TP9': (-113, 18),
        'TP10': (113, -18),'PO7': (-90, 54), 'PO8': (90, -54),
        'PO9': (-113, 54), 'PO10': (113, -54), 'CPz': (22, -90),
        'POz': (67, -90), 'Fpz' : (90, 90), 'FCz': (23, 90)
    }

    # The table names the DRL and CMS electrodes "FpCz" and "FCz", respectively.
    # We will rename them here for clarity in the montage.
    coords_deg['FpCz_DRL'] = coords_deg.pop('Fpz')
    coords_deg['FCz_CMS'] = coords_deg.pop('FCz')

    # Step 2: Convert from spherical (theta, phi) to Cartesian (x, y, z)
    ch_pos_cartesian = {}
    for ch_name, (theta, phi) in coords_deg.items():
        # Convert degrees to radians
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        
        # This conversion formula maps the provided system to the MNE coordinate system
        # where +X is right, +Y is anterior, +Z is superior.
        x = np.sin(theta_rad) * np.cos(phi_rad)
        y = np.sin(theta_rad) * np.sin(phi_rad)
        z = np.cos(theta_rad)
        
        ch_pos_cartesian[ch_name] = np.array([x, y, z])
    
    # Step 3: Define idealized fiducial points based on a spherical head model.
    radius = 0.095  # Example radius in meters (9.5 cm)
    lpa = np.array([-radius, 0, 0])
    rpa = np.array([radius, 0, 0])
    nasion = np.array([0, radius, 0])
    
    # Step 4: Create the DigMontage, now including the fiducial points.
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_cartesian,
        lpa=lpa,
        rpa=rpa,
        nasion=nasion,
        coord_frame='head'
    )

    print("Successfully created the PROX-64 montage.")
    return montage

if __name__ == "__main__":
    prox64_montage = create_prox64_montage()

    # visualize the montage to verify electrode positions
    prox64_montage.plot(kind='3d', show_names=True)
    plt.show()

    # Save the montage to a file for later use
    prox64_montage.save('PROX-64-Montage-dig.fif', overwrite=True)
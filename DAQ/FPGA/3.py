#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# File path
file_path = "ADCout.dat"

# Global variables
adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
ninp = 4
nchan = 4096
timestamp_size = 8

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

file_size = os.stat(file_path).st_size
nspec = file_size // (ninp * nchan * 4 + timestamp_size)

if nspec == 0:
    print("Error: No spectra found in the file.")
    sys.exit(1)

print(f"There were {nspec} spectra recorded.")

# Initialize Auto data array
Autospec = np.zeros((nspec, ninp, nchan // 2), dtype=np.single)

# Read data from the binary file
with open(file_path, "rb") as fp:
    for speccnt in range(nspec):
        # Skip the timestamp
        fp.read(timestamp_size)
        autospec = np.fromfile(fp, dtype=np.single, count=nchan * ninp)
        if autospec.size != nchan * ninp:
            print(f"Incomplete data at spectrum {speccnt}. Skipping.")
            break
        autospec = autospec.reshape((ninp, nchan))
        Autospec[speccnt, :, :] = autospec[:, :nchan // 2]

# Plotting auto-correlation data
fig1, ax1 = plt.subplots(ninp, 1, figsize=(10, 20), constrained_layout=True)
for cnt in range(ninp):
    autospec = Autospec[:, cnt, :]
    ratio = autospec.shape[0] / autospec.shape[1]
    cax = ax1[cnt].imshow(10 * np.log10(autospec.T + 1e-10), cmap='gray', aspect='auto')
    ax1[cnt].invert_yaxis()
    cbar = fig1.colorbar(cax, ax=ax1[cnt], fraction=0.047 * ratio)
    cbar.ax.set_ylabel('dB')
    ax1[cnt].set_xlabel('Spectrum count')
    ax1[cnt].set_ylabel('Channel')
    ax1[cnt].set_title(f'{adc_channels[cnt]}')

# Interactive spectrum selection
while True:
    pltval = fig1.ginput(1, timeout=0, show_clicks=True)
    if not pltval:
        plt.close()
        sys.exit(0)

    spectra_num = int(np.round(pltval[0][0]))
    if spectra_num < 0 or spectra_num >= nspec:
        print(f"Invalid spectrum number: {spectra_num}")
        continue

    print(f"Selected spectrum: {spectra_num}")

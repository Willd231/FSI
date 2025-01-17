#!/usr/bin/python3
#
# Program to read and plot the output of UDP data saved in ADCout.dat.
#
# Will and Anish, Jan 2025

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

<<<<<<< HEAD
# Constants
nchan = 4096  # Number of FFT channels (adjust if needed)
ninp = 4      # Number of ADC channels
file_path = "ADCout.dat"  # Path to the output file
packet_size = nchan * 4  # Size of one auto-correlation array per channel in bytes
=======
# ADC labels
adc = ['ADC_A', 'ADC_B', 'ADC_C', 'ADC_D']

# Global vars
nchan = 4096  # Number of FFT channels
ninp = 4      # Number of inputs (ADC channels)
file_path = "ADCout.dat"
>>>>>>> 999c755 (f)

# Check if the file exists
if not os.path.isfile(file_path):
    print(f"File not found: {file_path}")
    sys.exit(1)

# Determine the number of spectra in the file
file_size = os.stat(file_path).st_size
record_size = 8 + (ninp * packet_size)  # Timestamp (8 bytes) + 4 channels of data
nspec = file_size // record_size
print(f"Processing file: {file_path}")
print(f"Number of spectra: {nspec}")

# Initialize arrays for storing data
timestamps = np.zeros(nspec, dtype=np.float64)
Autospec = np.zeros((nspec, ninp, nchan), dtype=np.single)

# Read the data file
with open(file_path, "rb") as f:
    for speccnt in range(nspec):
        # Read timestamp
        timestamps[speccnt] = np.fromfile(f, dtype=np.float64, count=1)

        # Read auto-correlation data for each channel
        for ch in range(ninp):
            Autospec[speccnt, ch, :] = np.fromfile(f, dtype=np.single, count=nchan)

# Plotting auto-correlation data
fig1, ax1 = plt.subplots(ninp, 1, figsize=(10, 20), constrained_layout=True)
for cnt in range(ninp):
    autospec = Autospec[:, cnt, :]
    ratio = autospec.shape[0] / autospec.shape[1]
    cax = ax1[cnt].imshow(10 * np.log10(autospec.T), cmap='copper_r', aspect='auto')
    ax1[cnt].invert_yaxis()
    cbar = fig1.colorbar(cax, ax=ax1[cnt], fraction=0.047 * ratio)
    cbar.ax.set_ylabel('dB')
    ax1[cnt].set_xlabel('Spectrum count')
    ax1[cnt].set_ylabel('Channel')
    ax1[cnt].set_title(f'Auto-correlation: ADC_{chr(65 + cnt)}')

# Interactive spectrum selection
while True:
    pltval = fig1.ginput(1, timeout=0, show_clicks=True)
    if not pltval:
        plt.close()
        sys.exit(0)

    spectraNum = int(np.round(pltval[0][0]))
    if spectraNum >= nspec:
        print(f"Selected spectrum {spectraNum} is out of range. Try again.")
        continue

    print(f"Selected spectrum: {spectraNum}")

    # Plot the selected spectrum
    fig2, axs = plt.subplots(ninp, 1, figsize=(10, 15), constrained_layout=True)
    for i in range(ninp):
        axs[i].cla()
        axs[i].plot(10 * np.log10(Autospec[spectraNum, i, :]))
        axs[i].set_title(f'Auto-correlation: ADC_{chr(65 + i)}')
        axs[i].set_xlabel('Channel')
        axs[i].set_ylabel('Amplitude (dB)')

    fig2.canvas.draw()
    plt.show(block=False)

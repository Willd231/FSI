#!/usr/bin/python3
#
# Program to display autocorrelations of selected data.
#
# Will and Anish, Jan 2025

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# ADC labels
adc = ['ADC_A', 'ADC_B', 'ADC_C', 'ADC_D']

# Global vars
nchan = 4096  # Number of FFT channels
print(f"FFT channels = {nchan}")
ninp = 4
Datdir = "/home/anish/DarkMol/analysis/procdat/"
print(f"Data dir: {Datdir}")

# Input the first auto file in the sequence
afiletype = input("Enter the first auto file in the sequence: ")

# File reading
auto = Datdir + afiletype[:-8] + "*LACSPC"
autofiles = sorted(glob.glob(auto), key=os.path.getmtime)

for cnt in range(len(autofiles)):
    print(f"Auto file: {autofiles[cnt]}")

# Lists to hold spectra counts in each file
indexarr = []
nspec = 0

# Loop through auto-correlation files to calculate the total number of spectra
for file in autofiles:
    file_size = os.stat(file).st_size
    nspec += file_size // (nchan * ninp * 4)
    indexarr.append(nspec)

print(f"There were {nspec} spectra recorded.")

# Initialize auto-correlation data array
Autospec = np.zeros((nspec, ninp, nchan // 2), dtype=np.single)

# Loop through auto-correlation files to read data
speccnt = 0
for file in autofiles:
    file_size = os.stat(file).st_size
    nspec_in_file = file_size // (nchan * ninp * 4)

    with open(file, "rb") as f:
        for pcnt in range(nspec_in_file):
            autospec = np.fromfile(f, dtype=np.single, count=nchan * ninp)
            autospec.shape = (ninp, nchan)
            Autospec[speccnt, :, :] = autospec[:, 0:nchan // 2]
            speccnt += 1

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
    ax1[cnt].set_title(f'{adc[cnt]}')

# Interactive spectrum selection
while True:
    pltval = fig1.ginput(1, timeout=0, show_clicks=True)
    if not pltval:
        plt.close()
        sys.exit(0)

    spectraNum = int(np.round(pltval[0][0]))

    filec = 0
    for i in range(len(indexarr) - 1):
        if indexarr[i] > spectraNum:
            filec = i
            break

    if filec > 0:
        spectraNum = spectraNum - indexarr[filec - 1]

    print(f"Selected spec count: {spectraNum}, input file: {autofiles[filec]}")

    file_path1 = autofiles[filec]

    # Open the file and plot the selected spectrum
    with open(file_path1, "rb") as file1:
        speccnt = 0
        while True:
            autospec = np.fromfile(file1, dtype=np.single, count=nchan * ninp)
            if autospec.size != nchan * ninp:
                break
            autospec.shape = (ninp, nchan)

            if speccnt == spectraNum:
                break
            speccnt += 1

    # Plot the auto-correlation data
    fig2, axs = plt.subplots(ninp, 1, figsize=(10, 15), constrained_layout=True)
    for i in range(ninp):
        axs[i].cla()
        axs[i].plot(10 * np.log10(autospec[i, 0:nchan // 2]))
        axs[i].set_title(f'Auto {adc[i]}')
        axs[i].set_xlabel('Channel')
        axs[i].set_ylabel('Amplitude (dB)')

    fig2.canvas.draw()
    plt.show(block=False)

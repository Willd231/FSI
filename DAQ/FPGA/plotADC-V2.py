#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

file_path = "ADCout.dat"

adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
ninp = 4
nchan = 4096  
timestamp_size = 8  

# Calculate sizes for data parsing
data_block_size = nchan * ninp * 4  
nbaselines = (ninp * (ninp - 1)) // 2

file_size = os.stat(file_path).st_size
nspec = file_size // (data_block_size + timestamp_size)
print(f"There were {nspec} Spectra recorded")

Autospec = np.zeros((nspec, ninp, nchan // 2), dtype=np.single)

# Read
with open(file_path, "rb") as fp:
    for speccnt in range(nspec):
        # Skip the timestamp (8 bytes)
        fp.read(timestamp_size)
        
        autospec = np.fromfile(fp, dtype=np.single, count=nchan * ninp)
        if autospec.size != nchan * ninp:
            print(f"Incomplete data at spectrum {speccnt}. Skipping.")
            break
        print(f"Read data size: {len(autospec)} bytes")
        
       
        autospec = autospec.reshape((ninp, nchan))
        print(f"Reshaped autospec shape: {autospec.shape}")
        
        # Store the first half of the channels in Autospec
        Autospec[speccnt, :, :] = autospec[:, :nchan // 2]
        print(f"Autospec sample (spectrum 0, channel 0): {Autospec[0, 0, :10]}")
        

fig1, ax1 = plt.subplots(ninp, 1, figsize=(10, 20), constrained_layout=True)
for cnt in range(ninp):
    autospec = Autospec[:, cnt, :]  # Select data for each ADC channel
    ratio = autospec.shape[0] / autospec.shape[1]
    
    
    cax = ax1[cnt].imshow(10 * np.log10(autospec.T), cmap='copper_r', aspect='auto')
    ax1[cnt].invert_yaxis()
    cbar = fig1.colorbar(cax, ax=ax1[cnt], fraction=10 * ratio)
    cbar.ax.set_ylabel('dB')
    ax1[cnt].set_xlabel('Spectrum count')
    ax1[cnt].set_ylabel('Channel')
    ax1[cnt].set_title(f'{adc_channels[cnt]}')

# Wait for user input to select spectra
while True:
    pltval = fig1.ginput(1, timeout=0, show_clicks=True)
    if not pltval:
        plt.close()
        sys.exit(0)

    spectra_num = int(np.round(pltval[0][0]))
    print(f"Selected spectrum: {spectra_num}")

plt.show()

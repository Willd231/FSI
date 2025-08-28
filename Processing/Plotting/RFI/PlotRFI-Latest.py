#/usr/bin/python3

import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Constants and file path
file_path = "ADCout.dat"
figure_closed = False
adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
ninp = 4
nchan = 512
timestamp_size = 4
freq_size = 4
data_block_size = nchan * ninp * 4 + 4 * nchan  
nbaselines = (ninp * (ninp - 1)) // 2

# Check if the file exists
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

# Get file size and calculate the number of spectra
file_size = os.stat(file_path).st_size
nspec = file_size // (data_block_size + timestamp_size)

# Check if there are any spectra
if nspec == 0:
    print("Error: No spectra found in the file.")
    sys.exit(1)

print(f"There were {nspec} Spectra recorded.")

# Initialize data array
Autospec = np.zeros((nspec, ninp, nchan), dtype=np.single)
timestamp = []

# Read the file and extract the data
with open(file_path, "rb") as fp:
    for speccnt in range(nspec):
        # Read timestamp
        raw_timestamp = fp.read(timestamp_size)
        if len(raw_timestamp) != timestamp_size:
            print(f"Incomplete timestamp at spectrum {speccnt}. Skipping.")
            break
        unpacked_timestamp = struct.unpack('<f', raw_timestamp)[0]
        timestamp.append(unpacked_timestamp)
        
        # Read frequency data
        freq = np.fromfile(fp, dtype=np.single, count=nchan)
        if len(freq) != nchan:
            print(f"Incomplete frequency data at spectrum {speccnt}. Skipping.")
            break

        # Read spectrum data
        autospec = np.fromfile(fp, dtype=np.single, count=nchan * ninp)
        if autospec.size != nchan * ninp:
            print(f"Incomplete data at spectrum {speccnt}. Skipping.")
            break
        
        # Reshape the autospec data
        autospec = autospec.reshape((ninp, nchan))
        
        # Store in Autospec array
        Autospec[speccnt, :, :] = autospec

# Convert timestamp to readable format
readable_times = [datetime.utcfromtimestamp(t) for t in timestamp]

# Print readable times (optional)
for t in readable_times:
    print(t)

# Set small values in Autospec to avoid log(0)
Autospec[Autospec <= 0] = 1e-10

# Create the figure for displaying the images
fig1, ax1 = plt.subplots(ninp, 1, figsize=(10, 20), constrained_layout=True)

new_times = readable_times[::2]

# Plot the spectrograms
for cnt in range(ninp):
    autospec = Autospec[:, cnt, :]
    cax = ax1[cnt].imshow(10 * np.log10(autospec.T), cmap='copper_r', aspect='auto')
    ax1[cnt].invert_yaxis()
    ax1[cnt].set_xticks(range(len(new_times)))
    ax1[cnt].set_xticklabels(readable_times)
    ax1[cnt].set_xlabel('Time')
    ax1[cnt].set_ylabel('Channel')
    ax1[cnt].set_title(f'{adc_channels[cnt]}')
    cbar = fig1.colorbar(cax, ax=ax1[cnt])
    cbar.ax.set_ylabel('dB')

# Function to handle figure close event
def on_close(event):
    global figure_closed
    figure_closed = True  
    print(f"Figure {event.canvas.figure.number} closed!")

# Function to plot selected spectrum
def plot_selected_spectrum(spectra_num):
    for i in range(ninp):
        ax1[i].clear()
        ax1[i].plot(freq, 10 * np.log10(Autospec[spectra_num, i, :]))
        ax1[i].set_title(f'{adc_channels[i]} - Spectrum {spectra_num}')
        ax1[i].set_xlabel('Frequency (MHz)')
        ax1[i].set_ylabel('Amplitude (dB)')
    plt.show(block=False)

# Main plot function to handle user interaction
def plot():
    while True:
        pltval = fig1.ginput(1, timeout=0, show_clicks=True)
        if pltval:
            spectra_num = int(pltval[0][0])
            if spectra_num < 0 or spectra_num >= nspec:
                print(f"Invalid spectrum number: {spectra_num}")
                continue

            print(f"Selected spectrum: {spectra_num}")
            plot_selected_spectrum(spectra_num)

# Start the plotting loop
plot()



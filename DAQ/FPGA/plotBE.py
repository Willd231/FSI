#!/home/wdellinger/anaconda3/bin/python

import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
file_path = "ADCout.dat"
figure_closed = False
adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
ninp = 4
nchan = 512
timestamp_size = 4
freq_size = 4
data_block_size = nchan * ninp * 4 + 4 * nchan  
nbaselines = (ninp * (ninp - 1)) // 2
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found.")
    sys.exit(1)

file_size = os.stat(file_path).st_size
nspec = file_size // (data_block_size + timestamp_size)

timestamp = []
if nspec == 0:
    print("Error: No spectra found in the file.")
    sys.exit(1)

print(f"There were {nspec} Spectra recorded.")


Autospec = np.zeros((nspec, ninp, nchan), dtype=np.single)


with open(file_path, "rb") as fp:
    for speccnt in range(nspec ):
            raw_timestamp = fp.read(timestamp_size)
            if len(raw_timestamp) != timestamp_size:
                      print(f"Incomplete timestamp at spectrum {speccnt}. Skipping.")
                      break

            unpacked_timestamp = struct.unpack('<f', raw_timestamp)[0]
            timestamp.append(unpacked_timestamp) 
            freq = np.fromfile(fp, dtype = np.single, count = nchan)
            print(freq.shape) 
            if len(timestamp) != timestamp_size:
                print(f"Incomplete timestamp at spectrum {speccnt}. Skipping.")
                break

            autospec = np.fromfile(fp, dtype=np.single, count=nchan * ninp)
            if autospec.size != nchan * ninp:
                print(f"Incomplete data at spectrum {speccnt}. Skipping.")
                break

            autospec = autospec.reshape((ninp, nchan))

            Autospec[speccnt, :, :] = autospec[:, :nchan]
            print(f"Error at spectrum {speccnt}: {e}")
            break
readable_times = [datetime.utcfromtimestamp(t) for t in timestamp]

for t in readable_times:
    print(t)
Autospec[Autospec <= 0] = 1e-10
fig1, ax1 = plt.subplots(ninp, 1, figsize=(10, 20), constrained_layout=True)
fig2, axs = plt.subplots(ninp, 1, figsize=(10, 15), constrained_layout=True)

for cnt in range(ninp):
    autospec = Autospec[:, cnt, :]
    ratio = autospec.shape[0] / autospec.shape[1]
    cax = ax1[cnt].imshow(10 * np.log10(autospec.T), cmap='copper_r', aspect='auto')
    ax1[cnt].invert_yaxis()
    ax1[cnt].set_xticks(range(len(readable_times)))  # Set positions
    ax1[cnt].set_xticklabels(readable_times)  

    ax1[cnt].set_xlabel('Time')
    ax1[cnt].set_ylabel('Channel')
    ax1[cnt].set_title(f'{adc_channels[cnt]}')
    cbar = fig1.colorbar(cax, ax=ax1[cnt])
    cbar.ax.set_ylabel('dB')

def on_close(event):
    global figure_closed
    figure_closed = True  
    print(f"Figure {event.canvas.figure.number} closed!")
    
    



def plot_selected_spectrum(spectra_num):
    
    for i in range(ninp):
        axs[i].clear()
        axs[i].plot(freq, 10 * np.log10(Autospec[spectra_num, i, :]))
        axs[i].set_title(f'{adc_channels[i]} - Spectrum {spectra_num}')
        axs[i].set_xlabel('Frequency (MHz)')
        axs[i].set_ylabel('Amplitude (dB)')
    plt.show(block=False)

def plot():
    while 1 == 1:
            pltval = fig1.ginput(1, timeout=0, show_clicks=True)
            

            spectra_num = int((pltval[0][0]))
            if spectra_num < 0 or spectra_num >= nspec:
                print(f"Invalid spectrum number: {spectra_num}")
                continue

            print(f"Selected spectrum: {spectra_num}")
            plot_selected_spectrum(spectra_num)
            
            return plot()

while 1 ==1:
        plot()

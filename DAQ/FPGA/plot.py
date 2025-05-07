#!/usr/bin/env python3

#imports
import os
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import MaxNLocator

#function to get the data from the file 
#returns the array with the data and the timestamps
def load_spectra_data(file_path, ninp, nchan, timestamp_size):

    data_block_size = nchan * ninp * 4 + 4 * nchan
    file_size = os.stat(file_path).st_size
    nspec = file_size // (data_block_size + timestamp_size)
    
    if nspec == 0:
        print("Error: No spectra found in the file.")
        sys.exit(1)
    
    print(f"There were {nspec} Spectra recorded.")
    
    timestamps = []
    Autospec = np.zeros((nspec, ninp, nchan), dtype=np.single)
    freq_axis = None
    
    with open(file_path, "rb") as fp:
        for speccnt in range(nspec):
            timestamp = fp.read(timestamp_size)
            if len(timestamp) != timestamp_size:
                print(f"Incomplete timestamp at spectrum {speccnt}")
                break
            try:
                t_val = struct.unpack('<f', timestamp)[0]
            except Exception as e:
                print(f"Error unpacking timestamp at spectrum {speccnt}: {e}")
                break
            timestamps.append(t_val)
            
            
            freq = np.fromfile(fp, dtype=np.single, count=nchan)
            if freq.size != nchan:
                print(f"Incomplete frequency data at spectrum {speccnt}")
                break
            if speccnt == 0:
                freq_axis = freq  
            
            
            autospec = np.fromfile(fp, dtype=np.single, count=nchan * ninp)
            if autospec.size != nchan * ninp:
                print(f"Incomplete spectral data at spectrum {speccnt}")
                break
            autospec = autospec.reshape((ninp, nchan))
            Autospec[speccnt, :, :] = autospec
    
    return timestamps, freq_axis, Autospec, nspec

#function for the first graph
def plot(timestamps, Autospec, adc_channels, date_str, readable_times):
   
    ninp = Autospec.shape[1]
    fig, axes = plt.subplots(ninp, 1, figsize=(12, 20), constrained_layout=True)
    fig.suptitle(f"Spectra Data from {date_str}", fontsize=16, fontweight='bold')
    
    for cnt in range(ninp):
        
        data = Autospec[:, cnt, :]
        im = axes[cnt].imshow(10 * np.log10(data.T), cmap='copper_r', aspect='auto')
        axes[cnt].invert_yaxis()
        
        # Set x-axis to show time labels
        axes[cnt].xaxis.set_major_locator(MaxNLocator(nbins=10))
        tick_indices = np.linspace(0, len(readable_times) - 1, min(10, len(readable_times)), dtype=int)
        axes[cnt].set_xticks(tick_indices)
        axes[cnt].set_xticklabels(np.array(readable_times)[tick_indices], rotation=45, ha='right')
        axes[cnt].set_xlabel('Time (HH:MM:SS)')
        axes[cnt].set_ylabel('Channel')
        axes[cnt].set_title(f'{adc_channels[cnt]}')
        fig.colorbar(im, ax=axes[cnt]).ax.set_ylabel('dB')
    
    return fig, axes

#to plot the result of the ginput
def plot_selected_spectrum(spec_index, freq, Autospec, adc_channels, ninp, fig_cache):
   
    if fig_cache['fig'] is None:
        fig, axes = plt.subplots(ninp, 1, figsize=(12, 15), constrained_layout=True)
        fig_cache['fig'] = fig
        fig_cache['axes'] = axes
    else:
        fig = fig_cache['fig']
        axes = fig_cache['axes']
    
    for i in range(ninp):
        axes[i].clear()
        axes[i].plot(freq, 10 * np.log10(Autospec[spec_index, i, :]))
        axes[i].set_title(f'{adc_channels[i]} - Spectrum {spec_index}')
        axes[i].set_xlabel('Frequency (MHz)')
        axes[i].set_ylabel('Amplitude (dB)')
    
    
    fig.canvas.draw_idle()
    plt.show(block=False)


def interactive_selection(fig_1, nspec, freq, Autospec, adc_channels, ninp):
    
    fig_cache = {'fig': None, 'axes': None}
    
    
    while True:
        pltval = plt.ginput(1, show_clicks=True)
        try:
            
            spec_index = int(round(pltval[0][0]))
        except Exception as e:
            print(f"Invalid input: {pltval[0][0]}, error: {e}")
            continue
        if spec_index < 0 or spec_index >= nspec:
            print(f"Invalid spectrum index: {spec_index}")
            continue
        
        print(f"Selected spectrum: {spec_index}")
        plot_selected_spectrum(spec_index, freq, Autospec, adc_channels, ninp, fig_cache)


def main():
    file_path = "ADCout.dat"
    adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
    ninp = 4
    nchan = 512
    timestamp_size = 4

    # time formatting 
    timestamps, freq, Autospec, nspec = load_spectra_data(file_path, ninp, nchan, timestamp_size)
    readable_times = [datetime.utcfromtimestamp(t).strftime('%H:%M:%S') for t in timestamps]
    date_str = datetime.utcfromtimestamp(timestamps[0]).strftime('%Y-%m-%d') if timestamps else "Unknown Date"
    
    Autospec[Autospec <= 0] = 1e-10
    
    fig_1, _ = plot(timestamps, Autospec, adc_channels, date_str, readable_times)
    plt.show(block=False)
    
    
    interactive_selection(fig_1, nspec, freq, Autospec, adc_channels, ninp)

if __name__ == '__main__':
    main()

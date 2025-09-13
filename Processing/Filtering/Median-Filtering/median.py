#!/usr/bin/python3

import os
import sys
import struct
import numpy as np

from scipy.ndimage import median_filter, uniform_filter1d



def getData():
    file_path = "ADCout.dat"
    adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
    ninp = 4
    nchan = 512
    timestamp_size = 4                    
    freq_block_size = 4 * nchan           
    autospec_block_size = 4 * nchan * ninp  
    record_size = timestamp_size + freq_block_size + autospec_block_size
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.", flush=True)
        sys.exit(1)

    file_size = os.stat(file_path).st_size
    nspec = file_size // record_size
    if nspec == 0:
        print("Error: No spectra found in the file.", flush=True)
        sys.exit(1)

    #print(f"There were {nspec} Spectra recorded.", flush=True)

    Autospec = np.zeros((nspec, ninp, nchan), dtype=np.float32)
    timestamps = []
    freq = None  

    with open(file_path, "rb") as fp:
        for i in range(nspec):
            # timestamp (float32 little-endian)
            raw_ts = fp.read(timestamp_size)
            if len(raw_ts) != timestamp_size:
                print(f"Incomplete timestamp at spectrum {i}. Stopping.", flush=True)
                break
            t = struct.unpack('<f', raw_ts)[0]
            timestamps.append(t)

            # frequency axis
            f = np.fromfile(fp, dtype=np.float32, count=nchan)
            if f.size != nchan:
                print(f"Incomplete freq block at spectrum {i}. Stopping.", flush=True)
                break
            if freq is None:
                freq = f.copy()

            # autospectra (ninp x nchan)
            a = np.fromfile(fp, dtype=np.float32, count=nchan * ninp)
            if a.size != nchan * ninp:
                print(f"Incomplete autospec block at spectrum {i}. Stopping.", flush=True)
                break
            Autospec[i] = a.reshape(ninp, nchan)

    
    if not timestamps:
        print("No timestamps", flush=True)
        sys.exit(1)
    if freq is None:
        print("No frequency", flush=True)
        sys.exit(1)

    data = {
        "timestamps": np.array(timestamps, dtype=np.float32),
        "freq": freq,
        "autospec": Autospec
    }
    
    return data


#take the log of the autocorelation, make a median filter copy of the data, divide the autocorelation by its med filtered counterpart 
def logit(data, nspec):
    for i in range(nspec):
        data[i] = np.log(data[i])
    return data
    
def do_filter(data, size=9):
    filtered = median_filter(data, size=(1,1,size), mode='nearest')
    return filtered

def do_filter_2(data, size=9):
    filtered = median_filter(data, size=(size, 1,1), mode='nearest')
    return filtered

def do_smooth(data, size=3):
   
    smoothed = uniform_filter1d(data, size=size, axis=2)
    return smoothed 
  
def do_smooth_2(data, size=3, axis=2):
   
    smoothed = uniform_filter1d(data, size=size, axis=axis)
    return smoothed    


def calc(data, smoothed, nspec):
    for i in range(nspec):
        data[i] = data[i]/smoothed[i]
    return data

def write(filename, autospec, freq, timestamps):
    nspec = autospec.shape[0]
    with open(filename, "wb") as file:
        for i in range(nspec):
            file.write(struct.pack('<f', timestamps[i]))
            freq.astype(np.float32).tofile(file)
            autospec[i].astype(np.float32).ravel().tofile(file)
    return

    

def main():
    data = getData()
    timestamps = data["timestamps"]
    freq = data["freq"]
    autospec = data["autospec"]
    nspec = autospec.shape[0]
    data = logit(autospec, nspec)
    sys.argv[1] = int(sys.argv[1])
    if(sys.argv[2] == '1'):
        filtered = do_filter(autospec, sys.argv[1])
        smoothed = do_smooth(filtered, size=5)
        write("output_x.dat", smoothed, freq, timestamps)
        
    elif(sys.argv[2] == '2'):
        filtered = do_filter_2(autospec, sys.argv[1])
        smoothed = do_smooth_2(filtered, size=5)
        write("output_y.dat", smoothed, freq, timestamps)   
    elif(sys.argv[2] == '3'):
        filtered = do_filter(autospec, sys.argv[1])
        smoothed = do_smooth(filtered, size=5)    
        final = calc(smoothed, autospec, nspec)
        write("output_diff.dat", final, freq, timestamps)
        
if __name__ == "__main__":
    main()
    
    





    

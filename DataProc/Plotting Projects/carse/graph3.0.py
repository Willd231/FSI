#!/home/will/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# variable declaration
nchan = 1024
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2

# file paths + directory
Datdir = "/home/anish/DarkMol/analysis/procdat/"

afiletype = input("Enter the first auto file in the sequence: ")
cfiletype = afiletype

#file types for the glob 
auto = Datdir + afiletype[:-8] + "*LACSPC"
cross = Datdir + cfiletype[:-8] + "*LCCSPC"


# Sorted lists of files based on modification time
autofiles = sorted(glob.glob(auto), key=os.path.getmtime)
crossfiles = sorted(glob.glob(cross), key=os.path.getmtime)

# Check if the auto or cross-correlation files exist
if len(autofiles) == 0:
    print("There were no auto correlation files captured")
    exit(1)

if len(crossfiles) == 0:
    print("There were no cross correlation files captured")
    exit(1)


for file_path1 in autofiles:
    print(f"Reading autospectrum file: {file_path1}")
    with open(file_path1, "rb") as file1:
        while True:
            autospec = np.fromfile(file1, dtype=np.single, count=nchan * ninp)
            if autospec.size != nchan * ninp:
                break
            autospec.shape = (ninp, nchan)
            
for file_path2 in crossfiles:
    print(f"Reading cross-correlation file: {file_path2}")
    with open(file_path2, "rb") as file2:
        while True:
            ccspec = np.fromfile(file2, dtype=np.cdouble, count=nchan * nbaselines)
            if ccspec.size != nchan * nbaselines:
                break
            ccspec.shape = (nbaselines, nchan)
            

# Combine all autospec and ccspec data using np.vstack
combined_autospec = np.vstack([autospec for file_path1 in autofiles for autospec in read_autospec(file_path1)])
combined_ccspec = np.vstack([ccspec for file_path2 in crossfiles for ccspec in read_ccspec(file_path2)])


# initialize the subplots
fig, axs = plt.subplots(4, 4, figsize=(10, 20), constrained_layout=True)

# loop to plot all of the combined data
for i in range(4):
    for j in range(4):
        if i == j:
            # Plot autospectra
            for autospec in combined_autospec:
                axs[i][j].plot(autospec[i])
                axs[i][j].set_title('Combined Auto from LACSPC')
                axs[i][j].set_xlabel('Channel')
                axs[i][j].set_ylabel('Amplitude')

        if j > i:
            # Plot cross-correlation magnitude
            for ccspec in combined_ccspec:
                axs[i][j].plot(np.abs(ccspec[i]))
                axs[i][j].set_title('Combined CC from LCCSPC (Magnitude)')
                axs[i][j].set_xlabel('Channel')
                axs[i][j].set_ylabel('Amplitude')

        elif i > j:
            # Plot cross-correlation phase
            for ccspec in combined_ccspec:
                axs[i][j].plot(np.angle(ccspec[i, :]))
                axs[i][j].set_title('Combined CC from LCCSPC (Phase)')
                axs[i][j].set_xlabel('Channel')
                axs[i][j].set_ylabel('Phase')

plt.show()

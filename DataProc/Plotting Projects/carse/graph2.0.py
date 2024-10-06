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
Datdir = "/home/will/Documents/work stuff/Plotting Project/New Corr/"
filetype1 = "temp.LACSPC"
filetype2 = "temp.LCCSPC"

# use glob to match all files in the directory
file_paths1 = glob.glob(os.path.join(Datdir, f"*{filetype1}*"))
file_paths2 = glob.glob(os.path.join(Datdir, f"*{filetype2}*"))

# check if matching files exist
if not (file_paths1 and file_paths2):
    print("No matching files found.")
    sys.exit(1)

# lists to hold data
autospec_list = []
ccspec_list = []

# iterate through all matching files and stack them
all_autospecs = []
all_ccspecs = []

for file_path1 in file_paths1:
    print(f"Reading autospectrum file: {file_path1}")
    with open(file_path1, "rb") as file1:
        while True:
            autospec = np.fromfile(file1, dtype=np.single, count=nchan * ninp)
            if autospec.size != nchan * ninp:
                break
            autospec.shape = (ninp, nchan)
            all_autospecs.append(autospec)

for file_path2 in file_paths2:
    print(f"Reading cross-correlation file: {file_path2}")
    with open(file_path2, "rb") as file2:
        while True:
            ccspec = np.fromfile(file2, dtype=np.cdouble, count=nchan * nbaselines)
            if ccspec.size != nchan * nbaselines:
                break
            ccspec.shape = (nbaselines, nchan)
            all_ccspecs.append(ccspec)

# Combine all autospec and ccspec data using np.vstack
if all_autospecs:
    combined_autospec = np.vstack(all_autospecs)
else:
    print("No valid autospectrum data.")
    sys.exit(1)

if all_ccspecs:
    combined_ccspec = np.vstack(all_ccspecs)
else:
    print("No valid cross-correlation data.")
    sys.exit(1)

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

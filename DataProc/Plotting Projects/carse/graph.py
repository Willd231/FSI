#!/home/will/anaconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# variable declaration
nchan = 1024
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2

#file paths + directory
Datdir = "/home/will/Documents/work stuff/data/FFT data/"
filetype1 = "temp.LACSPC"
filetype2 = "temp.LCCSPC"

# file reading
file_path1 = os.path.join(Datdir, filetype1)
file_path2 = os.path.join(Datdir, filetype2)

print(f"Checking existence of: {file_path1} and {file_path2}")


##check if the files exist 
#if not (os.path.exists(file_path1) and os.path.exists(file_path2)):
 ##  sys.exit(1)

#lists to hold data 
autospec_list = []
ccspec_list = []

#open the files
with open(file_path1, "rb") as file1, open(file_path2, "rb") as file2:
    while True:
        autospec = np.fromfile(file1, dtype=np.single, count=nchan * ninp)
        if autospec.size != nchan * ninp:
            break
        autospec.shape = (ninp, nchan)
        autospec_list.append(autospec)

        ccspec = np.fromfile(file2, dtype=np.cdouble, count=nchan * nbaselines)
        if ccspec.size != nchan * nbaselines:
            break
        ccspec.shape = (nbaselines, nchan)
        ccspec_list.append(ccspec)

#initialize the subplots
fig, axs = plt.subplots(4, 4, figsize=(10, 20), constrained_layout=True)

#loop to plot all of the data
for i in range(4):
    for j in range(4):
        for autospec in autospec_list:
            if i == j:
                axs[i][j].plot(autospec[i])
                axs[i][j].set_title('Auto from LACSPC')
                axs[i][j].set_xlabel('Channel')
                axs[i][j].set_ylabel('Amplitude')
        
        #for k, ccspec in enumerate(ccspec_list):
            if j > i:
                axs[i][j].plot(np.abs(ccspec[i]))
                axs[i][j].set_title('CC from LCCSPC')
                axs[i][j].set_xlabel('Channel')
                axs[i][j].set_ylabel('Amplitude')
            elif i > j:
                axs[i][j].plot(np.angle(ccspec[i, :]))
                axs[i][j].set_title('CC from LCCSPC')
                axs[i][j].set_xlabel('Channel')
                axs[i][j].set_ylabel('Phase')

plt.show()

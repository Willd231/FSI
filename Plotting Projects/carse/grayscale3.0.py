#!/home/will/anaconda3/bin/python

#imports
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Number of channels and inputs
nchan = 1024    
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2  

# Input the first auto and cross file in the sequence
afiletype = input("Enter the first auto file in the sequence: ")
cfiletype = input("Enter the first cross file in the sequence: ")

# Data directory
dataDir = "/home/anish/DarkMol/analysis/procdat/"

# Combined sources for the glob pattern
auto = dataDir + afiletype[:-8] + "*LACSPC"
cross = dataDir + cfiletype[:-8] + "*LCCSPC"

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

# Prompt user input to get the preferred input channels
n1 = int(input("Input the first input channel: "))
n2 = int(input("Now the second: "))

# Calculate CorIndex 
CorIndex = np.zeros((ninp, ninp), dtype=int)
num = 0
for i in range(ninp):
    for j in range(i + 1, ninp):
        CorIndex[i, j] = num
        CorIndex[j, i] = num
        num += 1

# Initialize variables for auto correlation data
Autospec1 = None
Autospec2 = None

# Loop through auto-correlation files
for file in autofiles:
    file_size = os.stat(file).st_size
    nspec = file_size // (nchan * ninp * 4)  # Assuming each value is 4 bytes (single precision float)
    if Autospec1 is None:
        Autospec1 = np.zeros((nspec, nchan), dtype=np.single)
        Autospec2 = np.zeros((nspec, nchan), dtype=np.single)

    with open(file, "rb") as f:
        for pcnt in range(nspec):
            autospec = np.fromfile(f, dtype=np.single, count=nchan * ninp)
            autospec.shape = (ninp, nchan)
            Autospec1[pcnt, :] = autospec[n1, :]
            Autospec2[pcnt, :] = autospec[n2, :]

# Initialize cross-correlation data array
corr = None

# Loop through cross-correlation files
for file in crossfiles:
    file_size = os.stat(file).st_size
    ncorr = file_size // (nchan * nbaselines * 2 * 4)  # Assuming complex values (2 floats for real and imaginary parts)

    if corr is None:
        corr = np.zeros((ncorr, nchan), dtype=np.csingle)

    with open(file, "rb") as f:
        for ccnt in range(ncorr):
            ccspec = np.fromfile(f, dtype=np.csingle, count=nchan * nbaselines)
            ccspec.shape = (nbaselines, nchan)
            corr[ccnt, :] = ccspec[CorIndex[n1, n2], :]

# Arrays to be compiled (vstack might be unnecessary as you are already using 2D arrays)
autospec1 = Autospec1
autospec2 = Autospec2
ccspec = corr

# Plotting auto correlation data
fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
ratio = autospec1.shape[0] / autospec1.shape[1]
cax1 = ax[0].imshow(10 * np.log10(autospec1), cmap='copper_r', aspect='auto')
cbar1 = fig.colorbar(cax1, ax=ax[0], fraction=0.047 * ratio)
ax[0].set_title(f'Auto Spectrum of ADC {n1 + 1}')
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Time')

cax2 = ax[1].imshow(10 * np.log10(autospec2), cmap='copper_r', aspect='auto')
cbar2 = fig.colorbar(cax2, ax=ax[1], fraction=0.047 * ratio)
ax[1].set_title(f'Auto Spectrum of ADC {n2 + 1}')
ax[1].set_xlabel('Channel')
ax[1].set_ylabel('Time')

# Plotting cross-correlation data
fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
cax3 = ax[0].imshow(10 * np.log10(np.abs(ccspec)), cmap='copper_r', aspect='auto')
cbar3 = fig.colorbar(cax3, ax=ax[0], fraction=0.047 * ratio)
ax[0].set_title(f'Amplitude of the Cross Correlation of ADC {n1 + 1} and ADC {n2 + 1}')
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Time')

cax4 = ax[1].imshow(np.angle(ccspec), cmap='copper_r', aspect='auto')
cbar4 = fig.colorbar(cax4, ax=ax[1], fraction=0.047 * ratio)
ax[1].set_title(f'Phase of the Cross Correlation of ADC {n1 + 1} and ADC {n2 + 1}')
ax[1].set_xlabel('Channel')
ax[1].set_ylabel('Time')

plt.show()

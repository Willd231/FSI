#!/home/anish/anaconda3/bin/python
#
#Plot the cross and self products of a given ADC pair
#
#Will and Anish, Oct 7, 2024

#imports
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

#ADC labels
adc=['ADC_A', 'ADC_B', 'ADC_C', 'ADC_D']

# Number of channels and inputs
nchan = 4096 #1024    
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2  

# Data directory
dataDir = "/home/anish/DarkMol/analysis/procdat/"

print("FFT length {:4d}".format(nchan))
print("Data dir  {:s}".format(dataDir))

# Input the first auto and cross file in the sequence
afiletype = input("Enter the first auto file in the sequence: ")
cfiletype = afiletype 


# Combined sources for the glob pattern
auto = dataDir + afiletype[:-8] + "*LACSPC"
cross = dataDir + cfiletype[:-8] + "*LCCSPC"

# Sorted lists of files based on modification time
autofiles = sorted(glob.glob(auto), key=os.path.getmtime)
crossfiles = sorted(glob.glob(cross), key=os.path.getmtime)

for cnt in np.arange(len(autofiles)):
   print("{:s}  {:s}".format(autofiles[cnt], crossfiles[cnt]))

# Check if the auto or cross-correlation files exist
if len(autofiles) == 0:
    print("There were no auto correlation files captured")
    exit(1)

if len(crossfiles) == 0:
    print("There were no cross correlation files captured")
    exit(1)

# Prompt user input to get the preferred input channels
for i in range(ninp):
   print(f"{adc[i]} ==> ADC {i}")
n1 = int(input("Input the first ADC number: "))
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

# Loop through auto & cross correlation files to get the total number of spec
nspec=0
ncorr=0
for cnt, file in enumerate(autofiles):
    file_size = os.stat(file).st_size
    nspec = nspec + file_size // (nchan * ninp * 4)  # 

    file_size = os.stat(crossfiles[cnt]).st_size
    ncorr = ncorr + file_size // (nchan * nbaselines * 2 * 4)  

Autospec1 = np.zeros((nspec, nchan//2), dtype=np.single)
Autospec2 = np.zeros((nspec, nchan//2), dtype=np.single)
corr = np.zeros((ncorr, nchan//2), dtype=np.csingle)

# Loop through auto-correlation files
speccnt=0
for fcnt, file in enumerate(autofiles):
    file_size = os.stat(file).st_size
    nspec = file_size // (nchan * ninp * 4)  # 

    with open(file, "rb") as f:
        for pcnt in range(nspec):
            autospec = np.fromfile(f, dtype=np.single, count=nchan * ninp)
            autospec.shape = (ninp, nchan)
            Autospec1[speccnt, :] = autospec[n1, 0:nchan//2]
            Autospec2[speccnt, :] = autospec[n2, 0:nchan//2]
            speccnt=speccnt+1

# Loop through cross-correlation files
speccnt=0

for fcnt, file in enumerate(crossfiles):
    file_size = os.stat(file).st_size
    ncorr = file_size // (nchan * nbaselines * 2 * 4)  

    with open(file, "rb") as f:
        for ccnt in range(ncorr):
            ccspec = np.fromfile(f, dtype=np.csingle, count=nchan * nbaselines)
            ccspec.shape = (nbaselines, nchan)
            corr[speccnt, :] = ccspec[CorIndex[n1, n2], 0:nchan//2]
            speccnt=speccnt+1

# Arrays to be compiled 
#autospec1 = Autospec1
#autospec2 = Autospec2
#ccspec = corr

# Plotting auto correlation data
fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
ratio = Autospec1.shape[0] / Autospec1.shape[1]
cax1 = ax[0].imshow(10 * np.log10(Autospec1), cmap='copper_r', aspect='auto')
ax[0].invert_yaxis()
cbar1 = fig.colorbar(cax1, ax=ax[0], fraction=0.047 * ratio)
cbar1.ax.set_ylabel('dB')
ax[0].set_title(f'Auto Spectrum of {adc[n1]}')
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Time')

cax2 = ax[1].imshow(10 * np.log10(Autospec2), cmap='copper_r', aspect='auto')
ax[1].invert_yaxis()
cbar2 = fig.colorbar(cax2, ax=ax[1], fraction=0.047 * ratio)
cbar2.ax.set_ylabel('dB')
ax[1].set_title(f'Auto Spectrum of {adc[n2]}')
ax[1].set_xlabel('Channel')
ax[1].set_ylabel('Time')

# Plotting cross-correlation data
fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
cax3 = ax[0].imshow(10 * np.log10(np.abs(corr)), cmap='copper_r', aspect='auto')
ax[0].invert_yaxis()
cbar3 = fig.colorbar(cax3, ax=ax[0], fraction=0.047 * ratio)
cbar3.ax.set_ylabel('dB')
ax[0].set_title(f'Amplitude of the Cross Correlation of {adc[n1]} and {adc[n2]}')
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Time')

cax4 = ax[1].imshow(np.angle(corr), vmax=0.02, vmin=-0.05, cmap='copper_r', aspect='auto')
ax[1].invert_yaxis()
cbar4 = fig.colorbar(cax4, ax=ax[1], fraction=0.047 * ratio)
cbar4.ax.set_ylabel('rad')
ax[1].set_title(f'Phase of the Cross Correlation of {adc[n1]} and {adc[n2]}')
ax[1].set_xlabel('Channel')
ax[1].set_ylabel('Time')

plt.show()

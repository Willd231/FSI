#!/home/anish/anaconda3/bin/python
#
#Program to display autocorrelations and all cross-products of selected data.
#
#Will and Anish, Oct 7, 2024

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

#ADC labels
adc=['ADC_A', 'ADC_B', 'ADC_C', 'ADC_D']

# global vars
nchan = 4096 #1024
print(f"FFT channels = {nchan}")
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2
Datdir = "/home/anish/DarkMol/analysis/procdat/"
print("Data dir  {:s}".format(Datdir))

# Input the first auto file in the sequence
afiletype = input("Enter the first auto file in the sequence: ")
cfiletype = afiletype

# file reading
file_path1 = os.path.join(Datdir, afiletype)
file_path2 = os.path.join(Datdir, cfiletype)

#obtain all of the files via glob 

auto = Datdir + afiletype[:-8] + "*LACSPC"
cross = Datdir + cfiletype[:-8] + "*LCCSPC"
autofiles = sorted(glob.glob(auto), key=os.path.getmtime)
crossfiles = sorted(glob.glob(cross), key=os.path.getmtime)

for cnt in np.arange(len(autofiles)):
   print("{:s}  {:s}".format(autofiles[cnt], crossfiles[cnt]))

#lists to hold nspec in each file 
indexarr = []
# Loop through auto & cross-correlation files to get the total number of spectra
nspec = 0
ncorr = 0
for cnt, file in enumerate(autofiles):
    file_size = os.stat(file).st_size
    nspec += file_size // (nchan * ninp * 4)
    indexarr.append(nspec)
    file_size = os.stat(crossfiles[cnt]).st_size
    ncorr += file_size // (nchan * nbaselines * 2 * 4)

print(f"There were {nspec} Spectra recorded and {ncorr} correlations recorded")

Autospec = np.zeros((nspec, ninp, nchan//2), dtype=np.single)

# Loop through auto-correlation files
speccnt=0
for fcnt, file in enumerate(autofiles):
    file_size = os.stat(file).st_size
    nspec = file_size // (nchan * ninp * 4)  #

    with open(file, "rb") as f:
        for pcnt in range(nspec):
            autospec = np.fromfile(f, dtype=np.single, count=nchan * ninp)
            autospec.shape = (ninp, nchan)
            Autospec[speccnt, :,:] = autospec[:, 0:nchan//2]
            speccnt=speccnt+1

# Plotting auto correlation data
fig1, ax1 = plt.subplots(ninp, 1, figsize=(10, 20), constrained_layout=True)
for cnt in np.arange(ninp):
   autospec=Autospec[:,cnt,:]
   ratio = autospec.shape[0] / autospec.shape[1]
   cax = ax1[cnt].imshow(10 * np.log10(autospec.T), cmap='copper_r', aspect='auto')
   ax1[cnt].invert_yaxis()
   cbar = fig1.colorbar(cax, ax=ax1[cnt], fraction=0.047 * ratio)
   cbar.ax.set_ylabel('dB')
   ax1[cnt].set_xlabel('Spectrum count')
   ax1[cnt].set_ylabel('Channel')
   ax1[cnt].set_title(f'{adc[cnt]}')

#initialize the subplots
fig2, axs = plt.subplots(ninp, ninp, figsize=(10, 20), constrained_layout=True)

# Calculate CorIndex
CorIndex = np.zeros((ninp, ninp), dtype=int)
num = 0
for i in range(ninp):
    for j in range(i + 1, ninp):
        CorIndex[i, j] = num
        CorIndex[j, i] = num
        num += 1

while 1==1:
   pltval = fig1.ginput(1,0,True)
   if not pltval:
     plt.close()
     sys.exit(0)

 
   spectraNum = int(np.round(pltval[0][0]))
   
   filec = 0 
   for i in range (0,len(indexarr)-1):
     if(indexarr[i]>spectraNum):
       filec = i
       break
   
   if filec > 0:
     spectraNum=spectraNum-indexarr[filec-1]
   
   print(f"Selected spec count {spectraNum} and input file {autofiles[filec]}")
   	
   file_path1 = autofiles[filec] 
   file_path2 = crossfiles[filec]
   
   #open the files
   speccnt=0
   with open(file_path1, "rb") as file1, open(file_path2, "rb") as file2:
       while True:
           autospec = np.fromfile(file1, dtype=np.single, count=nchan * ninp)
           if autospec.size != nchan * ninp:
               break
           autospec.shape = (ninp, nchan)
   
           ccspec = np.fromfile(file2, dtype=np.csingle, count=nchan * nbaselines)
           if ccspec.size != nchan * nbaselines:
               break
           ccspec.shape = (nbaselines, nchan)
   
           if speccnt==spectraNum:
               file1.close()
               file2.close()
               break
           speccnt=speccnt+1
   
   #loop to plot all of the data
   for i in range(ninp):
       for j in range(ninp):
               axs[i][j].cla()
               if i == j:
                   axs[i][j].plot(10*np.log10(autospec[i,0:nchan//2]))
                   axs[i][j].set_title(f'Auto {adc[i]}')
                   axs[i][j].set_xlabel('Channel')
                   axs[i][j].set_ylabel('Amplitude (dB)')
           
               if j > i:
                   axs[i][j].plot(10*np.log10(np.abs(ccspec[CorIndex[i,j],0:nchan//2])))
                   axs[i][j].set_title(f'CC ({adc[i]},{adc[j]})')
                   axs[i][j].set_xlabel('Channel')
                   axs[i][j].set_ylabel('Amplitude (dB)')
               elif i > j:
                   axs[i][j].plot(np.angle(ccspec[CorIndex[i,j],0:nchan//2]))
                   axs[i][j].set_title(f'CC ({adc[j]},{adc[i]})')
                   axs[i][j].set_xlabel('Channel')
                   axs[i][j].set_ylabel('Phase (rad)')
   fig2.canvas.draw()
   plt.show(block=False)


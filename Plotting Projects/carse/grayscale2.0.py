#!/home/will/anaconda3/bin/python

#imports
import numpy as np
import matplotlib.pyplot as plt
import os

#stuff regarding number of channels 
nchan = 1024
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2

# Directory and file names
dataDir = "/home/will/Documents/work stuff/Plotting Project/New Corr/"
auto = "temp.LACSPC"
cross = "temp.LCCSPC"


# Make full file paths
file1 = os.path.join(dataDir, auto)
file2 = os.path.join(dataDir, cross)

#prompt user input to get the preferred inputs
n1 = int(input("Enter 2 integers: "))
n2 = int(input(": "))

#check if the files exist
if os.path.exists(file1) and os.path.exists(file2):
   file_size = os.stat(file1).st_size
   nspec=file_size//(nchan*ninp*4)
   Autospec1=np.zeros((nspec,nchan), dtype=np.single)
   Autospec2=np.zeros((nspec,nchan), dtype=np.single)


# Reading and collecting both auto and cross-correlation data
file1= open(file1, "rb")
for pcnt in np.arange(nspec):
    autospec = np.fromfile(file1, dtype=np.single, count=nchan*ninp)
    autospec.shape=(ninp,nchan)
    Autospec1[pcnt,:]=autospec[n1,:]
    Autospec1[pcnt,:]=autospec[n2,:]
file1.close()

#find file size
file_size = os.stat(file2).st_size
ncorr=file_size//(nchan*nbaselines*2*8)
corr=np.zeros((ncorr,nchan), dtype=np.cdouble)

#open cc file
file2= open(file2, "rb")
for ccnt in np.arange(ncorr):
    ccspec = np.fromfile(file2, dtype=np.cdouble,count=nchan*nbaselines)
    ccspec.shape=(nbaselines,nchan)
    #corr[ccnt,:]=ccspec[CorIndex,:]
file2.close()


#arrays to be compiled (should contain the specified ranges from the original arrays)
autospec1 = np.zeros((nspec,nchan), dtype=np.single)
autospec2 = np.zeros((nspec,nchan), dtype=np.single)
ccspec = np.zeros((nspec,nchan), dtype=np.cdouble)

autospec1 = np.vstack(autospec1)
autospec1 = np.vstack(autospec1)
ccspec = np.vstack(ccspec)

#auto correlation plots
fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
ax[0].imshow(Autospec1, cmap='copper_r', aspect='auto')
ax[0].set_title('Auto Spectrum of ADC' + str(n1))
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Input')

ax[1].imshow(Autospec2, cmap='copper_r', aspect='auto')
ax[1].set_title('Auto Spectrum of ADC' + str(n2))
ax[1].set_xlabel('Channel')
ax[1].set_ylabel('Time')




#cross correlation plot
fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)
ax[0].imshow(np.abs(ccspec), cmap='copper_r', aspect='auto')
ax[0].set_title('Amplitude of the Cross Correlation of ADC' + str(n1) +' and ADC' + str(n2))
ax[0].set_xlabel('Channel')
ax[0].set_ylabel('Time')

ax[1].imshow(np.angle(ccspec), cmap='copper_r', aspect='auto')
ax[1].set_title('Phase of the Cross Correlation of ADC' + str(n1) +' and ADC' + str(n2))
ax[1].set_xlabel('Channel')
ax[1].set_ylabel('Time')

plt.show()
#!/home/will/anaconda3/bin/python

import numpy as np
import matplotlib.pyplot as plt
import os

nchan = 1024
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2

# Initialize lists to store the data
autospec_list = []
ccspec_list = []

# Directory and file names
dataDir = "/home/will/Documents/work stuff/Plotting Project/New Corr/"
auto = "temp.LACSPC"
cross = "temp.LCCSPC"

# Make full file paths
file1 = os.path.join(dataDir, auto)
file2 = os.path.join(dataDir, cross)

# Reading and collecting both auto and cross-correlation data
with open(file1, "rb") as f1, open(file2, "rb") as f2:
    while True:
        autospec = np.fromfile(f1, dtype=np.single, count=nchan * ninp)
        if autospec.size != nchan * ninp:
            break
        autospec = autospec.reshape((ninp, nchan))
        autospec_list.append(autospec)

        ccspec = np.fromfile(f2, dtype=np.cdouble, count=nchan * nbaselines)
        if ccspec.size != nchan * nbaselines:
            break
        ccspec = ccspec.reshape((nbaselines, nchan))
        ccspec_list.append(ccspec)

# Start user input gathering to specify which plots are needed
n = input("Type 'b' for both graphs, 'a' for only auto, or 'c' for only cross: ")

# If User input for plotting options both graphs
if n == 'b':
    fig, ax = plt.subplots(2, 1, figsize=(10, 20), constrained_layout=True)

    autospec = np.concatenate(autospec_list)
    ccspec = np.concatenate(ccspec_list)

    ax[0].imshow(autospec, cmap='copper_r', aspect='auto')
    ax[0].set_title('Auto Spectrum')
    ax[0].set_xlabel('Channel')
    ax[0].set_ylabel('Input')

    ax[1].imshow(np.abs(ccspec), cmap='copper_r', aspect='auto')
    ax[1].set_title('Cross Spectrum')
    ax[1].set_xlabel('Channel')
    ax[1].set_ylabel('Baseline')

# If just autospec
elif n == 'a':
    num = input(f"Type 'a' for all, 'o' for one channel, or 'm' for multiple but not all. There are {len(autospec_list)} values for each auto channel: ")
    autospec_size = len(autospec_list)

    if num == 'a':
        autospec = np.concatenate(autospec_list)
        plt.imshow(autospec, cmap='copper_r', aspect='auto')
        plt.title('Auto Spectrum')
        plt.xlabel('Channel')
        plt.ylabel('Input')
    elif num == 'o':
        number = int(input("Specify which channel (1-4): "))
        if 1 <= number <= ninp:
            templist = autospec_list[number - 1::ninp]
            if templist:
                autospec = np.concatenate(templist)
                plt.imshow(autospec, cmap='copper_r', aspect='auto')
                plt.title(f'Auto Spectrum - Channel {number}')
                plt.xlabel('Channel')
                plt.ylabel('Input')
            else:
                print("its empty")
        else:
            print("Channel number out of bounds!")
    elif num == 'm':
        templist = []
        while True:
            i = int(input("Enter the channel number (1-4) you want or enter 0 to exit: "))
            if i == 0:
                break
            elif 1 <= i <= ninp:
                templist.extend(autospec_list[i - 1::ninp])
            else:
                print("number out of bounds!")
        if templist:
            autospec = np.concatenate(templist)
            plt.imshow(autospec, cmap='copper_r', aspect='auto')
            plt.title('Auto Spectrum - Selected Channels')
            plt.xlabel('Channel')
            plt.ylabel('Input')
        else:
            print("No channels selected or no data available.")

# If just cross-correlation
elif n == 'c':
    num2 = input("Enter 'a' for all, 'm' for multiple, or 'o' for just one baseline: ")

    if num2 == 'a':
        ccspec = np.concatenate(ccspec_list)
        plt.imshow(np.abs(ccspec), cmap='copper_r', aspect='auto')
        plt.title('Cross Spectrum')
        plt.xlabel('Channel')
        plt.ylabel('Baseline')
        #skip specified pieces of the data till you get to the area where the channel they want is
    elif num2 == 'o':
        number = int(input(f"Specify which baseline (1-{nbaselines}): "))
        if 1 <= number <= nbaselines:
            templist = [cc[number - 1, :] for cc in ccspec_list] 
            if templist:
                ccspec = np.vstack(templist)
                plt.imshow(np.abs(ccspec), cmap='copper_r', aspect='auto')
                plt.title(f'Cross Spectrum - Baseline {number}')
                plt.xlabel('Channel')
                plt.ylabel('?')
            else:
                print("empty?")
        else:
            print("number out of bounds!")
    elif num2 == 'm':
        #if they want to select multiple we can just put all of their answers into an array
        templist = []
        while True:
            i = int(input(f"Enter number (1-{nbaselines}) you want or enter 0 to exit: "))
            if i == 0:
                break
            elif 1 <= i <= nbaselines:
                for cc in ccspec_list:
                    templist.append(cc[i - 1, :])
            else:
                print("number out of bounds!")
        if templist:
            ccspec = np.vstack(templist)
            plt.imshow(np.abs(ccspec), cmap='copper_r', aspect='auto')
            plt.title('Cross Spectrum - Selected Baselines')
            plt.xlabel('Channel')
            plt.ylabel('baseline')
        else:
            print("no data available.")

# Show all the selected plots
plt.show()

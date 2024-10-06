#!/home/wdellinger/

import numpy as np
import glob 
import matplotlib.pyplot as plt
import sys
import os


# global vars
nchan = 1024
ninp = 4
nbaselines = (ninp * (ninp - 1)) // 2
Datdir = "/home/anish/DarkMol/analysis/procdat/"
afiletype = '20240816_185307_0000'
cfiletype = afiletype


def file_handling():
    auto = Datdir + afiletype[:-8] + "*LACSPC"
    cross = Datdir + cfiletype[:-8] + "*LCCSPC"
    autofiles = sorted(glob.glob(auto), key=os.path.getmtime)
    crossfiles = sorted(glob.glob(cross), key=os.path.getmtime)

    # Check if the auto or cross-correlation files exist
    if len(autofiles) == 0:
        print("There were no auto correlation files captured")
        sys.exit(1)

    if len(crossfiles) == 0:
        print("There were no cross correlation files captured")
        sys.exit(1)

    return crossfiles, autofiles


def spectra_finder(autofiles, crossfiles):
    spectraNum = input("Enter the specific spectra you would like: ")

    # Loop through auto & cross-correlation files to get the total number of spectra
    nspec = 0
    ncorr = 0
    for cnt, file in enumerate(autofiles):
        file_size = os.stat(file).st_size
        nspec += file_size // (nchan * ninp * 4)

        file_size = os.stat(crossfiles[cnt]).st_size
        ncorr += file_size // (nchan * nbaselines * 2 * 4)

    print(f"There were {nspec} Spectra recorded and {ncorr} correlations recorded")
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
    return ncorr, nspec


def plotting(combined_autospec, combined_ccspec):
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


# Execution flow
crossfiles, autofiles = file_handling()
ncorr, nspec = spectra_finder(autofiles, crossfiles)

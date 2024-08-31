#!/home/anish/anaconda3/bin/python3

#Version 1
#Program developed to check counter values sent by FPGA.
#
#Developed by Will Delinger based on suggestions from Anish.
#April 29, 2024


import numpy as np
import os
import glob
import sys

Datdir = 'C:\\Users\\WILLT\\OneDrive\\Documents\\data\\'

#Input to specify the directory
filetype = input('Enter the filename for the first file in the sequence : ')

file_path=Datdir+filetype
if os.path.exists(file_path):
        with open(file_path, "rb") as file:
                b = np.frombuffer(file.read(8), dtype=np.uint64).copy()
                print(b)
else:
        print(Datdir+filetype, " file does not exist!!")
        sys.exit(1)

# Path to the directory containing files
path = Datdir + filetype[:-8] +'*.dat'

# Get a list of files in the directory
files = sorted(glob.glob(path), key=os.path.getmtime)

for file_path in files:
        print(file_path)

# Iterate through each file
for file_path in files:
        print(file_path)
        # Get the file size using os.stat
        file_size = os.stat(file_path).st_size

    # Calculate the number of iterations needed based on file size
        total_num_packets = file_size // (8448)
        print("Total number of packets : ", total_num_packets)

        packet_cnt=0
        with open(file_path, "rb") as file:

        # Loop through the data in chunks of 8448*128 bytes 
            for j in range(total_num_packets):
                        data = file.read(8448)

            # Convert bytes to uint64 array
                        intval = np.frombuffer(data, dtype=np.uint64)
                #print(intval)

            # Iterate through count values
                        for k in range(0, len(intval), 8):
                                count_value = intval[k]
                        #print(b)
                        #print(count_value)

                # Compare count values
                                if count_value != b:
                                        print("Total Packet count :", total_num_packets)
                                        print("Packet count :", packet_cnt)
                                        print("Mismatched counter at:", count_value)
                                        print("Expected counter:", b)
                                        b=count_value+1
                                else:
                                        b += 1
                        packet_cnt=packet_cnt+1



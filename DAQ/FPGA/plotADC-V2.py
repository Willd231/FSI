#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# File path to the input binary data file
file_path = "ADCout.dat"

# ADC and configuration details
adc_channels = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
num_channels = len(adc_channels)
spectrum_length = 4096  # Based on the FFT computation in the generator program
timestamp_size = 8  # Size of the timestamp (float64)
data_block_size = spectrum_length * num_channels * 4  # Size of one iteration data (4 bytes per float32)

# Read the binary file
with open(file_path, "rb") as file:
    data = file.read()

# Parse the data into timestamps and spectra
data = np.frombuffer(data, dtype=np.float32)
total_blocks = len(data) // (timestamp_size // 4 + spectrum_length * num_channels)
timestamps = np.zeros(total_blocks, dtype=np.float64)
spectra = np.zeros((total_blocks, num_channels, spectrum_length), dtype=np.float32)

for i in range(total_blocks):
    offset = i * (timestamp_size // 4 + spectrum_length * num_channels)
    timestamps[i] = np.frombuffer(data[offset:offset + timestamp_size // 4], dtype=np.float64)
    start = offset + timestamp_size // 4
    spectra[i] = data[start:start + spectrum_length * num_channels].reshape((num_channels, spectrum_length))

# Plot the data for each channel
plt.figure(figsize=(12, 8))
for i in range(num_channels):
    plt.subplot(2, 2, i + 1)
    spectrum_avg = np.mean(spectra[:, i, :], axis=0)  # Average spectrum over iterations
    spectrum_db = 10 * np.log10(spectrum_avg)  # Convert to dB
    plt.plot(spectrum_db, label=f"Channel {adc_channels[i]}")
    plt.title(f"Spectrum for {adc_channels[i]}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

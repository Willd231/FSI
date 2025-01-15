#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# File path to the input binary data file
file_path = "ADCout.dat"

# Define the number of spectra and the length of each spectrum
num_spectra = 4  # Four spectra in the file
spectrum_length = 1024  # Replace with the actual length of each spectrum

# Load the data
with open(file_path, "rb") as file:
    data = np.fromfile(file, dtype=np.float32)

# Ensure the data length matches the expected size
assert len(data) == num_spectra * spectrum_length, "Data size mismatch."

# Reshape data into a 2D array where each row is a spectrum
data = data.reshape((num_spectra, spectrum_length))

# Apply a 10 * log10 transformation (handling log of non-positive values)
data_db = np.where(data > 0, 10 * np.log10(data), np.nan)

# Plot each spectrum
plt.figure(figsize=(12, 8))
for i in range(num_spectra):
    plt.subplot(2, 2, i + 1)
    plt.plot(data_db[i], label=f"Spectrum {i+1}")
    plt.title(f"Spectrum {i+1}")
    plt.xlabel("Channel")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

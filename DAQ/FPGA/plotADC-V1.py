#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

file_path = "ADCout.dat"

with open(file_path, "rb") as file:
    data = np.fromfile(file, dtype=np.float32)

data = np.where(data > 0, 10 * np.log10(data), np.nan) =

plt.figure(figsize=(10, 6))
plt.plot(data, label="10 * log10(Values)")
plt.title("Power Spectrum Visualization")
plt.xlabel("Time")
plt.ylabel("Power")
plt.legend()
plt.grid(True)
plt.show()                                                                       



    


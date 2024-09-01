
#program to plot the LACSPC output

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

nchan=1024
Datdir = './'

filetype='temp.LACSPC'

file_path=Datdir+filetype
if os.path.exists(file_path):
   with open(file_path, "rb") as file:
     b = np.frombuffer(file.read(1024*16), dtype=np.single)
else:
   print(Datdir+filetype, " file does not exist!!")
   sys.exit(1)

plt.plot(b)
plt.show()


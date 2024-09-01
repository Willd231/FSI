#!/home/will/anaconda3/bin/python

import sys
sys.path.append('/home/will/Documents/work stuff/Plotting Project/New Corr/')

import numpy as np
import matplotlib.pyplot as plt
from dliteTools import read_file_sink

NCHAN=512
v0=read_file_sink('/home/anish/DarkMol/DLITE/11-28-2024/1638057602/visfile_23_xx.bin',NCHAN)

fig = plt.figure(figsize=(6, 5))

ax = fig.add_subplot(111)
im = plt.imshow(10*np.log10(np.abs(v0[0:25000,:])), cmap="copper_r", aspect="auto")
plt.colorbar(im)
plt.show()

#for cnt in np.arange(10):
#   plt.plot(np.abs(v0[cnt,:]))
#   #plt.plot(np.angle(v0[cnt,:]))

#plt.plot(np.abs(v0[9500,:]))
#plt.plot(np.abs(v0[20000,:]))
#plt.plot(np.abs(v0[35000,:]))
#plt.plot(np.angle(v0[:,170]))
#plt.plot(np.abs(v0[:,170]))
#plt.plot(np.abs(v0[:,160]))
plt.show()

#plt.title("23_XX_11-28-2024")
#plt.xlabel("channel no.")
#plt.ylabel("Cross power (arb unit)")
#plt.show()

sys.exit(1)

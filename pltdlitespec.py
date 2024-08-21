

import sys
sys.path.append('C://Users//WILLT//OneDrive//Documents//stuff for work//plotting project//dliteTools.py')

import numpy as np
import matplotlib.pyplot as plt
from dliteTools import read_file_sink 

NCHAN=512
v0=read_file_sink('C://Users//WILLT//OneDrive//Documents//stuff for work//data//FFT data//temp.LACSPC',NCHAN)
plt.plot(np.abs(v0[cnt,:]))
plt.plot(np.angle(v0[cnt,:]))

plt.plot(np.abs(v0[9500,:]))
plt.plot(np.abs(v0[20000,:]))
plt.plot(np.abs(v0[35000,:]))
#plt.plot(np.angle(v0[:,170]))
#plt.plot(np.abs(v0[:,170]))
#plt.plot(np.abs(v0[:,160]))
plt.show()

#plt.title("23_XX_11-28-2024")
#plt.xlabel("channel no.")
#plt.ylabel("Cross power (arb unit)")
#plt.show()

sys.exit(1)



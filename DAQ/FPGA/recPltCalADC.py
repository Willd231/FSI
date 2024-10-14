#!/home/anish/anaconda3/bin/python
#
#Program used for calibrating the ADC count to dBm

import socket
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib as mpl

#Assuming ADC attenuators set to 0 dB
#ADC sine wave RMS to power in dBm conversion
#0 dBm --> 6610 rms (16bit values)
#ADC noise RMS to power in dBm conversion (input power measured using specturm analyzer measurement setup)
#0 dBm --> 6000 rms (16bit values)
#In this calibration, the noise floor power of the ADC is at -53 dBm; this power level is independent of ADC attenuator in the 0 - 20dB range 

Consine=20*np.log10(6610)
nConamp=6000
Connoise=20*np.log10(nConamp)
ADCnoisefloor=-53 #dBm

Nsamp=49.152 #MHz
ADC=['ADCA', 'ADCB', 'ADCC', 'ADCD']
ADCatt=['0','0','0','0'] #Assumed ADC attenuation. This has to be set to the actual values set in the FPGA

#Receive data from FPGA
ipad="10.17.16.10"
packetsize=8256
receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive.bind((ipad, 60000))
data, address = receive.recvfrom(packetsize)
receive.close()

fig1, ax = plt.subplots(nrows=2, ncols=1,constrained_layout=True)
fig1.set_figwidth(8)
fig1.set_figheight(10)

params = {
   'axes.labelsize': 15,
   'font.size': 15,
   'legend.fontsize': 12,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'text.usetex': False,
   'figure.figsize': [10, 6],
   'lines.linewidth': 3
   }
mpl.rcParams.update(params)

for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(2)
    ax[1].spines[axis].set_linewidth(2)

ax[0].tick_params(width=2)
ax[1].tick_params(width=2)

#plot time series and power spec
sintval=np.frombuffer(data, dtype=np.int16)
for adccnt in np.arange(4):
   adcdat=sintval[(32+adccnt)::4]
   adcrms=np.std(adcdat)
   adcmax=max(np.abs(adcdat))
   bits_sine=np.log2(adcmax*2/4)
   bits_noise=np.log2(adcrms*10/4)
   print("{:s}".format(ADC[adccnt]))
   print("RMS value {:f}, Max value {:f}, Peak bits sine (0-14): {:f}, Peak bits noise (0-14): {:f}".format(
                  adcrms, adcmax, bits_sine, bits_noise))
   print("Input sine wave power : {:7.3f} dBm".format(20*np.log10(adcrms)-Consine))
   print("Input noise power : {:7.3f} dBm".format(20*np.log10(adcrms)-Connoise))

   ax[0].plot(adcdat, label=ADC[adccnt], linewidth=3)

   #Power spectrum calibrated in terms of dBm/resolution for noise case 
   adcdat=adcdat/nConamp
   ft=fft(adcdat)/len(adcdat)
   Ps=np.real(ft*np.conj(ft))
   freq=np.arange(len(Ps))*Nsamp/len(Ps)

   #The power over the positive freq channels is 3 dB higher and so 3 is added
   ax[1].plot(freq, 10*np.log10(Ps)+3, label=ADC[adccnt], linewidth=3)

ADCspecfloor=ADCnoisefloor-10*np.log10(len(adcdat)/2)
ax[1].axhline(ADCspecfloor, color='k', label='ADC noise floor', linewidth=3)

ax[0].set_xlabel('Samples', fontsize=15)
ax[0].set_ylabel('ADC counts', fontsize=15)
ax[0].legend()
ax[1].set_xlabel('Frequency (MHz)', fontsize=15)
ax[1].set_ylabel('ADC input power (dBm)', fontsize=15)
ax[1].set_xlim((0,Nsamp/2))
ax[1].legend()
ax[1].set_title("Assumed ADC attenuation : " + ",".join(ADCatt)+" dB")

plt.show()

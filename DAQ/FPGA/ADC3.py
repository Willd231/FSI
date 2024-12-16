#!/home/will/anaconda3/bin/python
import socket
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib as mpl
import time 


nConamp=6000
t = float(input("How long would you like this program (hrs): "))
t = t * 60 * 60
Nsamp=49.152 
ADC=['ADCA', 'ADCB', 'ADCC', 'ADCD']
ADCatt=['0','0','0','0'] 
curr = time.time()

ipad="10.17.16.10"
packetsize=8256
receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive.bind((ipad, 60000))


with open("ADCout.dat", "wb") as output:
    for i in range(0, t):
        
        data, address = receive.recvfrom(packetsize)
        sintval=np.frombuffer(data, dtype=np.int16)
        for adccnt in np.arange(4):
            adcdat=sintval[(32+adccnt)::4]
        
        adcdat=adcdat/nConamp
        ft=fft(adcdat)/len(adcdat)
        Ps=np.real(ft*np.conj(ft))
        freq=np.arange(len(Ps))*Nsamp/len(Ps)

        output.write(Ps)
        current = 1 - curr
        time.sleep(current)

output.close()
receive.close()
#!/home/will/anaconda3/bin/python
import socket
import numpy as np
#from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib as mpl
import time 


nConamp=6000
t = input("How long would you like this program (hrs): ")
t = int(t) * 60 * 60
Nsamp=49.152 
ADC=['ADCA', 'ADCB', 'ADCC', 'ADCD']
ADCatt=['0','0','0','0'] 
count = 0
'''
ipad="10.17.16.10"
packetsize=8256
receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive.bind((ipad, 60000))
'''

with open("ADCout.dat", "wb") as output:
    for i in range(0, t):
        curr1 = time.time()
        
        #data, address = receive.recvfrom(packetsize)
        #sintval=np.frombuffer(data, dtype=np.int16)
        #for adccnt in np.arange(4):
        #    adcdat=sintval[(32+adccnt)::4]
        
        #adcdat=adcdat/nConamp
        #ft=fft(adcdat)/len(adcdat)
        #Ps=np.real(ft*np.conj(ft))
        #freq=np.arange(len(Ps))*Nsamp/len(Ps)

        
        #output.write(Ps)
        time.sleep(.01)
        curr2 = time.time()
        curr3 = curr2 - curr1
        bintime = (np.frombuffer(curr3, dtype =np.int16)).tobytes()
        output.write(bintime)
        time.sleep(1-curr3)
        print(count)
        count +=1

output.close()
receive.close()

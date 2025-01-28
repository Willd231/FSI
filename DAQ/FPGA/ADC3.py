#!/home/anish/anaconda3/bin/python
import socket
import numpy as np
<<<<<<< HEAD
import time
import sys

=======
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib as mpl
import time 


>>>>>>> 3be344aa97b0c0b32ceaf576b5b413c05a380cbe
nConamp=6000
t = float(input("How long would you like this program (hrs): "))
t = t * 60 * 60
Nsamp=49.152 
ADC=['ADCA', 'ADCB', 'ADCC', 'ADCD']
ADCatt=['0','0','0','0'] 
curr = time.time()
<<<<<<< HEAD
est = input("How long do you want each iteration to be?: ")
=======

>>>>>>> 3be344aa97b0c0b32ceaf576b5b413c05a380cbe
ipad="10.17.16.10"
packetsize=8256
receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive.bind((ipad, 60000))

<<<<<<< HEAD
count = 0
try:
    with open("ADCout.dat", "wb") as output:
        start_time = time.time()
        
        while time.time() - start_time < t:
            curr1 = time.time()
            output.write(np.array([curr1], dtype=np.float64).tobytes())
            data, address = receive.recvfrom(packetsize)
            sintval = np.frombuffer(data, dtype=np.int16)
            for adccnt in range(4):
                adcdat = sintval[(32 + adccnt)::4]
                adcdat = adcdat / nConamp
                ft = np.fft.fft(adcdat) / len(adcdat)
                Ps = np.real(ft * np.conj(ft))
                freq = np.arange(len(Ps)) * Nsamp / len(Ps)
                output.write(Ps.tobytes())

            curr2 = time.time()
            curr3 = curr2 - curr1  
            
            time.sleep(max(0,  est - curr3))

            print(f"Iteration: {count}, Time taken: {curr3:.4f}s")
            count += 1

except KeyboardInterrupt:
    print("You Stopped The Program! ")

finally:
    output.close()
    receive.close()
    
=======

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
>>>>>>> 3be344aa97b0c0b32ceaf576b5b413c05a380cbe

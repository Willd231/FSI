#!/usr/bin/python3
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
est = input("How long do you want each iteration to be?: ")
ipad="10.17.16.10"
packetsize=8256
receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive.bind((ipad, 60000))

count = 0
try:
    with open("ADCout.dat", "wb") as output:
        start_time = time.time()
        
        while time.time() - start_time < t:
            curr1 = time.time()
            output.write(np.array([curr1], dtype=np.float32).tobytes())
            data, address = receive.recvfrom(packetsize)
            sintval = np.frombuffer(data, dtype=np.int16)
            for adccnt in range(4):
                adcdat = sintval[(32 + adccnt)::4]
                adcdat = adcdat / nConamp
                ft = np.fft.fft(adcdat) / len(adcdat)
                Ps = np.real(ft * np.conj(ft))
                Ps[Ps <= 0] = 1e-10
                fig = mpl.plt(Ps)
                plt.show()
                freq = np.arange(len(Ps)) * Nsamp / len(Ps)
                output.write(Ps.astype(np.float32).tobytes())

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
    

#!/home/wdellinger/anaconda3/bin/python

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
nfft = 1024 
nAVG = 10
freq = np.arange(nfft//2) * Nsamp/ nfft
ninp = 4
count = 0
with open("ADCout.dat", "wb") as output:
    start_time = time.time()
    
    while time.time() - start_time < t:
        curr1 = time.time()
        output.write(np.array([curr1], dtype=np.float32).tobytes())
        output.write(freq.astype(np.float32).tobytes())
        psAVG = np.zeros((nfft//2, ninp))

        for i in range (0, nAVG): 
                data, address = receive.recvfrom(packetsize)
                sintval = np.frombuffer(data, dtype=np.int16)
                for adccnt in range(ninp):
                    adcdat = sintval[(32 + adccnt)::4]
                    adcdat = adcdat / nConamp
                    ft = np.fft.fft(adcdat) / len(adcdat)
                    psAVG[:,adccnt] = psAVG[:,adccnt] + np.real(ft[0:len(ft)//2] * np.conj(ft[0:len(ft)//2]))
        psAVG=psAVG/nAVG
        for i in range (0, ninp):
              # plt.plot(psAVG[:,i])
              # plt.show()      
               output.write(psAVG[:,i].astype(np.float32).tobytes())
        curr2 = time.time()
        curr3 = curr2 - curr1  
        
        time.sleep(max(0,  float(est) - curr3))

        print(f"Iteration: {count}, Time taken: {curr3:.4f}s")
        count += 1

#except KeyboardInterrupt:
#    print("You Stopped The Program! ")

#finally:
output.close()
receive.close()
    


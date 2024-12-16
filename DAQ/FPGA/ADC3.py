#!/home/wdellinger/anaconda3/bin/python
import socket
import numpy as np
import time

nConamp = 6000
Nsamp = 49.152
ADC = ['ADCA', 'ADCB', 'ADCC', 'ADCD']
ADCatt = ['0', '0', '0', '0']

t = input("How long would you like this program to run (hrs): ")
t = int(t) * 60 * 60  # Convert hours to seconds

ipad = "10.17.16.10"
packetsize = 8256
receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receive.bind((ipad, 60000))

count = 0
try:
    with open("ADCout.dat", "wb") as output:
        start_time = time.time()
        
        while time.time() - start_time < t:
            curr1 = time.time()
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
            curr3 = curr2 - curr1  # Time taken for the loop iteration
            
            output.write(np.array([curr3], dtype=np.float64).tobytes())

            time.sleep(max(0, 1 - curr3))

            print(f"Iteration: {count}, Time taken: {curr3:.4f}s")
            count += 1

except KeyboardInterrupt:
    print("You Stopped The Program! ")

finally:
    output.close()
    receive.close()

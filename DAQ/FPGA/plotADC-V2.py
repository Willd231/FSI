#!/usr/bin/python3

import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys 
import glob



adc = ['ADC_A', 'ADC_B', 'ADC_C', 'ADC_D']
nchan = 4096
print(f"Number of Channels: {nchan}")
ninp = 4
baselines = (ninp *(ninp - 1)) //2
Datdir = "/home/anish/DarkMol/analysis/procdat/"
print("Data dir  {:s}".format(Datdir))



def spectrum_analysis(): 
    
    
    
    with open(file, "rb") as f: 
        
##!/home/anish/anaconda3/bin/python3
#program to plot the LACSPC output

#made by anish to act as a template

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

nchan=1024
ninp=4
nbaselines=(ninp*(ninp-1)//2)

Datdir = "C://Users//WILLT//OneDrive//Documents//stuff for work//Correlator//"

filetype1='temp.LACSPC'
filetype2='temp.LCCSPC'

file_path1=Datdir+filetype1
file_path2=Datdir+filetype2

#autospec=np.zeros((ninp,nchan), dtype=np.single)
#ccspec=np.zeros((ninp*(ninp-1)//2, nchan), dtype=np.cdouble)

if os.path.exists(file_path1) and os.path.exists(file_path2):
   file1= open(file_path1, "rb")
   autospec = np.fromfile(file1, dtype=np.single, count=nchan*ninp)
   file1.close()
   autospec.shape=(ninp,nchan)

   file2= open(file_path2, "rb")
   ccspec = np.fromfile(file2, dtype=np.cdouble,count=nchan*nbaselines)
   file2.close()
   ccspec.shape=(nbaselines,nchan)

#   ccnt=0
#   for inp1 in np.arange(ninp):
#     for inp2 in np.arange(inp1,ninp):
#        if inp1 == inp2:
#          autospec[inp1,:] = np.fromfile(file1, dtype=np.single, count=nchan)
#        else:
#          ccspec[ccnt,:] = np.fromfile(file2, dtype=np.cdouble,count=nchan)
#          ccnt=ccnt+1

else:
   print(Datdir+filetype1, " file does not exist!! or ")
   print(Datdir+filetype2, " file does not exist!!")
   sys.exit(1)

#plt.plot(np.transpose(autospec))
plt.plot(np.angle(np.transpose(ccspec)))
plt.show()

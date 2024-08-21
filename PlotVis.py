import os
import sys
import numpy as np   
import matplotlib.pyplot as plt
import glob
 


fig1, ax1 = plt.subplots(nrows=4, ncols=4,constrained_layout=True)
fig1.set_figwidth(10)
fig1.set_figheight(9)
fig2, ax2 = plt.subplots(nrows=1, ncols=1,constrained_layout=True)
fig2.set_figwidth(10)
fig2.set_figheight(9)
fig3, ax3 = plt.subplots(nrows=4, ncols=4, constrained_layout=True)
fig3.set_figwidth(10)
fig3.set_figheight(9)
fig4, ax4 = plt.subplots(nrows = 1,ncols=2,constrained_layout=True)
fig4.set_figwidth(10)
fig4.set_figheight(9)



disp_chan=np.array([100,200]) #display channels
disp_base=0  #display baseline
disp_pol=0  #display pol


for i, chan in enumerate(disp_chan):
        ax2.plot(10*np.log10(np.abs(v0[:,chan])), label="chan {:2d}".format(chan))

ax2.legend()
ax2.set_xlabel('Time count')
ax2.set_ylabel('Cross pow (dB)')


ax2.set_title('Amp of ' + label[disp_base][disp_pol])

fig2.canvas.draw()

for i in range (2):

    ax4[0].plot(10 * np.log10(np.abs(v0[:, chan])), label="chan {:2d}".format(chan))



ax4[0].legend()


ax4[0].imshow(10 * np.log10(np.abs(v0[0:40000, :])), cmap="flag", aspect="auto")

ax4[0].set_title("XX")

ax4[0].axis("on")


v0=read_file_sink(visfile[disp_base][disp_pol+1], NCHAN)

ax4[1].imshow(10 * np.log10(np.abs(v0[0:40000, :])), cmap="flag", aspect="auto")

ax4[1].set_title("YY")

ax4[1].axis("on")
fig4.canvas.draw()
plt.show(block=False)

ans=int(input("Enter time count to display spec : "))
print(ans)

nstart=ans
v0,v1=read_vis('/home/anish/DarkMol/DLITE/11-28-2024/1638057602',nstart=nstart,nt=10)


#v0=np.zeros((6,4,2,512))
#v1=[ [" " for i in range(4)] for j in range(6)]
NCHAN=v0.shape[3]

b=0 #baseline index
for i in np.arange(4):
   for j in np.arange(i,4):

      ax1[i,j].set_xlim((0,NCHAN))
      ax1[j,i].set_xlim((0,NCHAN))
      ax3[i,j].set_xlim((0,NCHAN))
      ax3[j,i].set_xlim((0,NCHAN))
      if i == j:
        continue
      else:
        ax1[i,j].plot(10*np.log10(np.abs(v0[b,0,0,:])))
        ax1[i,j].plot(10*np.log10(np.abs(v0[b,1,0,:])))
        ax1[i,j].set_title("Amp of " + v1[b][0] + v1[b][1])
        ax1[i,j].set_xlabel("Channel no.")
        ax1[i,j].set_ylabel("Cross pow (dB)")
        ax1[j,i].plot(np.angle(v0[b,0,0,:]))
        ax1[j,i].plot(np.angle(v0[b,1,0,:]))
        ax1[j,i].set_title("Phase of " + v1[b][0] + v1[b][1])
        ax1[j,i].set_xlabel("Channel no.")
        ax1[j,i].set_ylabel("angle (rad)")
        ax3[i,j].plot(10*np.log10(np.abs(v0[b,2,0,:])))
        ax3[i,j].plot(10*np.log10(np.abs(v0[b,3,0,:])))
        ax3[i,j].set_title("Amp of " + v1[b][2] + v1[b][3])
        ax3[i,j].set_xlabel("Channel no.")
        ax3[i,j].set_ylabel("Cross pow (dB)")
        ax3[j,i].plot(np.angle(v0[b,2,0,:]))
        ax3[j,i].plot(np.angle(v0[b,3,0,:]))
        ax3[j,i].set_title("Phase of " + v1[b][2] + v1[b][3])
        ax3[j,i].set_xlabel("Channel no.")
        ax3[j,i].set_ylabel("angle (rad)")

        b=b+1






fig1.canvas.draw()
fig3.canvas.draw()

plt.show()



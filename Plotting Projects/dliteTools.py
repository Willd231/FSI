#Provided, was not written by me
#!/usr/bin/python
import numpy as np
import os
import astropy.units as unt
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
from astropy.time import Time
from astropy.coordinates import get_sun,get_body,EarthLocation,AltAz
re=6370. ; fac=np.pi/180.
re*=1000.#puts Re in meters
#A-team data
name=['Cyg A','Cas A','Vir A','Tau A','Her A','Hyd A']
f38=[22000,22000,3700,2400,1800,1000]
ns=len(name)
radec=[
    299.868153,40.733916,
    350.866417,58.811778,
    187.705930,12.391123,
    83.633212,22.014460,
    252.783945,4.992588,
    139.524,-12.096]
ra_a=np.array(radec)[np.arange(ns)*2]
dec_a=np.array(radec)[np.arange(ns)*2+1]
pos_a=SkyCoord(ra=ra_a*unt.degree,dec=dec_a*unt.degree,equinox='J2000')

def detrend(t,x,w,nfit=3):
    nt=len(t) ; y=np.zeros(nt)
    if(nt>w):
        for i in range(nt):
            if(x[i]!=0):
                i1=i-int(w/2)
                if(i1<0):
                    i1=0
                if(i1>nt-w):
                    i1=nt-w
                tmp1=t[i1:i1+w]
                tmp2=x[i1:i1+w]
                ind=np.where(tmp2!=0)[0]
                if(len(ind)>=nfit):
                    c=np.polyfit(tmp1[ind],tmp2[ind],1)
                    y[i]=x[i]-c[0]*t[i]-c[1]
    else:
        ind=np.where(x!=0)[0]
        if(len(ind)>=nfit):
            c=np.polyfit(t[ind],x[ind],1)
            y[ind]=x[ind]-c[0]*t[ind]-c[1]
    return(y)

def tstamp2lst(lng,ts):
    mjd=2440587.5+ts/86400.
    t=np.arange(len(mjd))
    c=[280.46061837,360.98564736629,0.000387933,38710000.0]
    jd2000=2451545.0
    t0=mjd-jd2000
    t[:]=t0/36525
    theta=c[0]+c[1]*t0+t**2*(c[2]-t/c[3])
    lst=(theta+lng)/15.
    ind=np.where(lst<0)
    if(len(ind[0])>0):
        lst[ind]=24.+np.mod(lst[ind],24.)
    lst=np.mod(lst,24.)
    return(lst)

def tstamp2ut(ts):
    jd=2440587.5+ts/86400.
    jdmin=np.min(jd)
    jd0=int(jdmin)+0.5
    if(jd0>jdmin):
        jd0-=1.
    return((jd-jd0)*24.)

def lstsq(x,y,yerr=None):
    import numpy as np
    if(yerr is None):
        yerr=np.copy(y-y)+1.
    n=x.shape[1]
    mat=np.zeros((n,n),x.dtype) ; soln=np.zeros(n,y.dtype)
    for i in range(n):
        soln[i]=np.sum(x[:,i]*y/yerr**2)
        for j in range(n):
            mat[j,i]=np.sum(x[:,i]*x[:,j]/yerr**2)
    c=np.linalg.pinv(mat)
    p=(np.matrix(soln)*np.matrix(c))
    p=np.array(p).reshape(n)
    ep=np.sqrt(np.abs(np.diagonal(c)))
    return(p,ep)

def spline(x,y,yp1=1.e30,ypn=1.e30,nmax=100000):
    import numpy as np
    n=len(x) ; y2=np.zeros(n) ; u=np.zeros(nmax)
    if(yp1>9.e29):
        y2[0]=0. ; u[0]=0.
    else:
        y2[0]=-0.5
        u[0]=(3./(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1)
    sig=(x[1:n-1]-x[0:n-2])/(x[2:n]-x[0:n-2])
    p=sig*y2[0:n-2]+2.
    y2[1:n-1]=(sig-1.)/p
    u[1:n-1]=(6.*((y[2:n]-y[1:n-1])/(x[2:n]-x[1:n-1])-(y[1:n-1]-y[0:n-2])/(x[1:n-1]-x[0:n-2]))/(x[2:n]-x[0:n-2])-sig*u[1:n-1])/p
    if(ypn>9.e29):
        qn=0.
        un=0.
    else:
        qn=0.5
        un=(3./(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]))
    y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.)
    for i in range(n-2,-1,-1):
        y2[i]=y2[i]*y2[i+1]+u[i]
    return(y2)

def splint(xa,ya,y2a,x):
    import numpy as np
    n=len(xa)
    y=np.zeros(len(x))
    for i in range(n-1):
        ind=np.where((x>=xa[i]) & (x<=xa[i+1]))[0]
        if(len(ind)>0):
            h=xa[i+1]-xa[i]
            a=(xa[i+1]-x[ind])/h
            b=(x[ind]-xa[i])/h
            y[ind]=a*ya[i]+b*ya[i+1]+((a**3-a)*y2a[i]+(b**3-b)*y2a[i+1])*(h**2)/6.
    return(y)

def cspline(x,xa,ya):
    y2=spline(xa,ya)
    return(splint(xa,ya,y2,x))

def comp_altaz_ha(ha,dec,lat):
    import numpy as np
    #ra, dec, lat, and lng in degrees
    fac=np.pi/180.
    sh=np.sin(ha*fac) ; ch=np.cos(ha*fac)
    sd=np.sin(dec*fac) ; cd=np.cos(dec*fac)
    sl=np.sin(lat*fac) ; cl=np.cos(lat*fac)
    x=-ch*cd*sl+sd*cl #=cos(az)*cos(alt)
    y=-sh*cd #=sin(az)*cos(alt)
    z=ch*cd*cl+sd*sl #=sin(alt)
    r=np.sqrt(x**2+y**2) #=cos(alt)
    az=np.arctan2(y,x)/fac
    alt=np.arctan2(z,r)/fac
    return(alt,az)

def comp_hadec(alt,az,lat):
    import numpy as np
    fac=np.pi/180.
    sa=np.sin(alt*fac) ; ca=np.cos(alt*fac)
    sz=np.sin(az*fac) ; cz=np.cos(az*fac)
    sl=np.sin(lat*fac) ; cl=np.cos(lat*fac)
    x=cl*sa-sl*ca*cz #=cos(dec)*cos(ha)
    y=-sz*ca #=cos(dec)*sin(ha)
    z=sl*sa+cl*ca*cz #=sin(dec)
    r=np.sqrt(x**2+y**2) #=cos(dec)
    ha=np.arctan2(y,x)/fac
    dec=np.arctan2(z,r)/fac
    return([ha,dec])

def deriv(x,y):
    import numpy as np
    n=y.shape[0]
    x12=x-np.roll(x,-1)
    x01=np.roll(x,1)-x
    x02=np.roll(x,1)-np.roll(x,-1)
    d=np.roll(y,1)*(x12/(x01*x02))+y*(1./x12-1./x01)-np.roll(y,-1)*(x01/(x02*x12))
    d[0]=y[0]*(x01[1]+x02[1])/(x01[1]*x02[1])-y[1]*x02[1]/(x01[1]*x12[1])+y[2]*x01[1]/(x02[1]*x12[1])
    n2=n-2
    d[n-1]=-y[n-3]*x12[n2]/(x01[n2]*x02[n2])+y[n-2]*x02[n2]/(x01[n2]*x12[n2])-y[n-1]*(x02[n2]+x12[n2])/(x02[n2]*x12[n2])
    return(d)

def read_file_sink(infile,lfft):
    dat=np.fromfile(open(infile,'rb'),dtype='f4')
    n=len(dat)
    v=dat[np.arange(0,n,2)]+1j*dat[np.arange(0,n,2)+1]
    nt=int(len(v)/lfft)
    return(v[:nt*lfft].reshape(nt,lfft))

def read_vis(indir,na=4):
    p={}
    exec(open(os.path.join(indir,'SETUP')).read(),p)
    NPOL=p['NPOL'] ; NCHAN=p['NCHAN']
    stk=['xx','yy','xy','yx']
    v0=read_file_sink(os.path.join(indir,'visfile_12_xx.bin'),NCHAN)
    nt=v0.shape[0]
    nb=int(na*(na-1)/2)
    vis=np.zeros((nb,NPOL,nt,NCHAN),'complex')
    b=0
    for i in range(na-1):
        for j in range(i+1,na):
            for k in range(NPOL):
                if((b==0) & (k==0)):
                    vis[b,k,:]=np.copy(v0)
                else:

                    suff='_'+str(i+1)+str(j+1)+'_'+stk[k]+'.bin'
                    infile=os.path.join(indir,'visfile'+suff)
                    vtmp=read_file_sink(infile,NCHAN)[:nt,:]
                    vis[b,k,:vtmp.shape[0],:]=np.copy(vtmp)
                b+=1
    return(vis)

def mad(x):
    return(np.median(np.abs(x-np.median(x))))

def flag_vis(vis,wch=8,wt=60,flim=7):
    nb,nstk,nt,nchan=vis.shape
    flg=np.zeros(vis.shape,'int')
    rat=np.zeros(vis.shape)
    for i in range(nb):
        for j in range(nstk):
            for k in range(int(nchan/wch)):
                for l in range(int(nt/wt)):
                    t1=l*wt ; t2=(l+1)*wt
                    if(l==int(nt/wt)-1):
                        t2=nt
                    amp=np.abs(vis[i,j,t1:t2,k*wch:(k+1)*wch])
                    lim=np.median(amp)+flim*mad(amp)
                    if(lim>0):
                        rat[i,j,t1:t2,k*wch:(k+1)*wch]=amp/lim
    np.place(flg,rat>=1,1)
    return(flg)

def rfisub(vis,wdetrend=59):
    nb,npol,nt,nch=vis.shape
    vtmp=np.zeros(vis.shape,'complex')
    for i in range(nb):
        for j in range(npol):
            for k in range(nch):
                ind=np.where(np.abs(vis[i,j,:,k])>0)[0]
                if(len(ind)>0):
                    tmp1=detrend(ind,vis.real[i,j,:,k][ind],wdetrend)
                    tmp2=detrend(ind,vis.imag[i,j,:,k][ind],wdetrend)
                    vtmp[i,j,ind,k]=tmp1+1j*tmp2
    return(vtmp)

def delay_spec(vis,flg=None,hamm=True):
    nb,nstk,nt,nchan=vis.shape
    dsp=np.zeros(vis.shape,'complex')
    vtmp=np.copy(vis)
    if(flg is not None):
        np.place(vtmp,flg==1,0.)
    if(hamm):
        window=0.54-0.46*np.cos(2*np.pi*np.arange(nchan)/float(nchan-1))
    else:
        window=np.ones(nchan)
    for i in range(nb):

        for j in range(nstk):
            for k in range(nt):
                dsp[i,j,k,:]=np.fft.fftshift(np.fft.fft(vtmp[i,j,k,:]*window))/float(nchan)
    return(dsp)

def bispec(vis):
    bs=[]
    nb=vis.shape[0]
    if(nb==3):
        bs=vis[0,:]*vis[1,:].conj()*vis[2,:]
    if(nb==6):
        bs=vis[0,:]*vis[1,:].conj()*vis[3,:]/4.
        bs+=vis[1,:]*vis[2,:].conj()*vis[5,:]/4.
        bs+=vis[0,:]*vis[2,:].conj()*vis[4,:]/4.
        bs+=vis[3,:]*vis[4,:].conj()*vis[5,:]/4.
    return(bs)

def read_ant(infile,aref=1):
    dat=np.loadtxt(infile)
    tmp=comp_altaz_ha(dat[aref-1,1]-dat[:,1],dat[:,0],dat[aref-1,0])
    x=(re+dat[:,2])*np.cos(tmp[0]*fac)*np.sin(tmp[1]*fac)
    y=(re+dat[:,2])*np.cos(tmp[0]*fac)*np.cos(tmp[1]*fac)
    z=(re+dat[:,2])*np.sin(tmp[0]*fac)-(re+dat[aref-1,2])
    return(x,y,z)

def get_tstamps(indir):
    p={}
    exec(open(os.path.join(indir,'SETUP')).read(),p)
    TINT=p['TINT'] ; BANDWIDTH=p['BANDWIDTH'] ; NCHAN=p['NCHAN']
    nav=int(TINT*BANDWIDTH/NCHAN) ; tint=nav*NCHAN/BANDWIDTH
    v0=read_file_sink(os.path.join(indir,'visfile_12_xx.bin'),NCHAN)
    nt=v0.shape[0]
    t0=float(str.split(indir,'/')[-1])
    return(np.arange(nt)*tint+t0)

def ateam_dircos(antfile,tstamps):
    nt=len(tstamps)
    dat=np.loadtxt(antfile)
    mlat=np.mean(dat[:,0]) ; mlng=np.mean(dat[:,1]) ; mel=np.mean(dat[:,2])
    lst=tstamp2lst(mlng,tstamps)
    ll=np.zeros((ns,nt)) ; mm=np.zeros((ns,nt)) ; nn=np.zeros((ns,nt))
    for i in range(nt):
        radec=pos_a.transform_to(FK5(equinox=Time(tstamps[i],format='unix')))
        a=np.array(radec.ra) ; d=np.array(radec.dec)
        alt,az=comp_altaz_ha(lst[i]*15.-a,d,mlat)
        ll[:,i]=np.cos(alt*fac)*np.sin(az*fac)
        mm[:,i]=np.cos(alt*fac)*np.cos(az*fac)
        nn[:,i]=np.sin(alt*fac)
    return(ll,mm,nn)

def ateam_delays(antfile,tstamps):
    x,y,z=read_ant(antfile)
    ll,mm,nn=ateam_dircos(antfile,tstamps)
    na=len(x) ; nb=int(na*(na-1)/2) ; ns,nt=ll.shape
    d=np.zeros((nb,ns,nt))
    b=0
    for i in range(na-1):
        for j in range(i+1,na):
            tmp=((x[i]-x[j])*ll+(y[i]-y[j])*mm+(z[i]-z[j])*nn)/2.998e8
            np.place(tmp,nn<=0,np.nan)
            d[b,:]=np.copy(tmp) ; b+=1
    return(d)

def sun_dircos(antfile,tstamps):
    nt=len(tstamps)
    dat=np.loadtxt(antfile)
    mlat=np.mean(dat[:,0]) ; mlng=np.mean(dat[:,1]) ; mel=np.mean(dat[:,2])
    lst=tstamp2lst(mlng,tstamps)
    ll=np.zeros(nt) ; mm=np.zeros(nt) ; nn=np.zeros(nt)
    pos_s=get_sun(Time(tstamps,format='unix'))
    a=np.array(pos_s.ra) ; d=np.array(pos_s.dec)
    alt,az=comp_altaz_ha(lst*15.-a,d,mlat)
    ll=np.cos(alt*fac)*np.sin(az*fac)
    mm=np.cos(alt*fac)*np.cos(az*fac)
    nn=np.sin(alt*fac)
    return(ll,mm,nn)

def sun_delays(antfile,tstamps):
    x,y,z=read_ant(antfile)
    ll,mm,nn=sun_dircos(antfile,tstamps)
    na=len(x) ; nb=int(na*(na-1)/2) ; nt=len(ll)
    d=np.zeros((nb,nt))
    b=0
    for i in range(na-1):
        for j in range(i+1,na):
            tmp=((x[i]-x[j])*ll+(y[i]-y[j])*mm+(z[i]-z[j])*nn)/2.998e8
            np.place(tmp,nn<=0,np.nan)
            d[b,:]=np.copy(tmp) ; b+=1
    return(d)

def jupiter_dircos(antfile,tstamps):
    nt=len(tstamps)
    dat=np.loadtxt(antfile)
    mlat=np.mean(dat[:,0]) ; mlng=np.mean(dat[:,1]) ; mel=np.mean(dat[:,2])
    lst=tstamp2lst(mlng,tstamps)
    ll=np.zeros(nt) ; mm=np.zeros(nt) ; nn=np.zeros(nt)
    pos_j=get_body('jupiter',Time(tstamps,format='unix'))
    a=np.array(pos_j.ra) ; d=np.array(pos_j.dec)
    alt,az=comp_altaz_ha(lst*15.-a,d,mlat)
    ll=np.cos(alt*fac)*np.sin(az*fac)
    mm=np.cos(alt*fac)*np.cos(az*fac)
    nn=np.sin(alt*fac)
    return(ll,mm,nn)
def jupiter_delays(antfile,tstamps):
    x,y,z=read_ant(antfile)
    ll,mm,nn=jupiter_dircos(antfile,tstamps)
    na=len(x) ; nb=int(na*(na-1)/2) ; nt=len(ll)
    d=np.zeros((nb,nt))
    b=0
    for i in range(na-1):
        for j in range(i+1,na):
            tmp=((x[i]-x[j])*ll+(y[i]-y[j])*mm+(z[i]-z[j])*nn)/2.998e8
            np.place(tmp,nn<=0,np.nan)
            d[b,:]=np.copy(tmp) ; b+=1
    return(d)

def fringe_rate(d,tstamps,frq):
    fr=np.zeros(d.shape)+np.nan
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ind=np.where(np.isnan(d[i,j,:])==False)[0]
            if(len(ind)>2):
                fr[i,j,ind]=deriv(tstamps[ind],d[i,j,ind])*frq
    return(fr)

def dircos2ipp(ll,mm,nn,lat=38.56,lng=-77.054,zion=300.):
    fac=np.pi/180.
    az=np.arctan2(ll,mm)/fac
    alt=np.arcsin(nn)/fac
    alph=np.arcsin(np.cos(alt*fac)*re/(re+zion*1000.))/fac
    tmp=comp_hadec(alt+alph,az,lat)
    return(tmp[1],lng-tmp[0])

def dircos2range(ll,mm,nn,lat=38.56,lng=-77.054,zion=300.):
    fac=np.pi/180.
    az=np.arctan2(ll,mm)/fac
    alt=np.arcsin(nn)/fac
    alph=np.arcsin(np.cos(alt*fac)*re/(re+zion*1000.))/fac
    r=np.sqrt(re**2+(re+zion*1000.)**2-2*re*(re+zion*1000.)*np.cos((90-alt-alph)*fac))/1000.
    return(r)


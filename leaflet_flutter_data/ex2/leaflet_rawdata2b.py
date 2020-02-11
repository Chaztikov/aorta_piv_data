
import numpy as np
import numpy.fft as fft
from numpy.fft import fftfreq

import pandas as pd
import os,sys,re
import subprocess

import scipy
import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# import seaborn as sns
import sklearn.decomposition
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
dimreductiontype = 'pca'

interpolate_signal = 1

cwd = os.getcwd()
dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

# bpm,iframe,ifframe=np.loadtxt('bpmdata.txt',unpack=True)
bpmdatas = np.loadtxt(dname+'bpmdata.txt', unpack=False)
bpmdatas = np.array(bpmdatas, dtype=int)
print(bpmdatas)


initframe=[12000,0,20000,21000]
nmaxima=[13,17,19,23]

initframe=[21000,21000,20000,22800]
nmaxima=[13,17,19,14]
bpms=[60,80,100,120]
splorder=1



dnx=11

# for idx, fname0 in enumerate(fnames[:]):

#     data = np.loadtxt(dname+fname0)[:, :]

#     x,y = data[:,0],data[:,1]

#     ii0=initframe[idx]
#     npeaks = nmaxima[idx]

#     dx = np.diff(x)

#     plt.figure(figsize=(24, 12))
#     plt.plot(y[ii0:])
#     plt.title(fname0)
#     plt.grid()
#     plt.xlabel('Time')
#     plt.ylabel('Area')
#     plt.savefig('rawsamples_'+fname0[-6:-4]+'.png')
#     plt.show()

for idx, fname0 in enumerate(fnames[:]):
    if(idx==3):
        data = np.loadtxt(dname+fname0)[:, :]

        x,y = data[:,0],data[:,1]

        ii0=initframe[idx]
        npeaks = nmaxima[idx]

        dx = np.diff(x)

        # plt.figure(figsize=(24,12))
        # plt.plot(y[ii0:])
        # plt.show()

        # y = y[ii0:]
        # nx=y.shape[0]

        # x = x[ii0:]
        dx=dx[:nx]
        # x = np.cumsum(dx)
        print(nx,x.shape)
        # x0=x[0]
        # ymax=y.max()
        # y /=ymax
        # x-=x[0]
        
        # inz=y.nonzero()[0]
        # x=x[inz]
        # y=y[inz]
        # y /=y.max()
        # x-=x[0]
        
        # plt.plot(xx,yy);plt.show()
        interp=BSpline(x,y,splorder)
        nx=x.shape[0]
        icycle = y.shape[0]//npeaks
        icycle 
        icyclef = y.shape[0]/npeaks 
        print(icycle,'\n',icyclef)
        nx2 = nx*dnx
        dx2 = dx.min()/dnx

        # nx3 = int(nx*icycle / np.gcd(nx, icycle))
        # print(nx2,nx3)
        # nx2= nx3
        x[0]
        xx=np.linspace(x[0],dx2*nx2, nx2, endpoint=True)
        print('xx\n',xx)
        yy=interp(xx)
        print('yy\n',yy)
        # inz2=yy.nonzero()[0]
        # print(inz2)
        # xx=xx[inz2]
        # yy=yy[inz2]
        # yy /=yy.max()
        # xx-=xx[0]
        
        icyclef = np.floor(yy.shape[0]/npeaks).astype(int)
        
        icycle = (yy.shape[0] )//(npeaks ) 
        
        print('icyclef',icyclef)
        print('icycle',icycle)
        
        dicycle = np.mod(yy.shape[0],npeaks)
        print('dicycle',dicycle)
        # dicycle = 

        # icycle = icycle+dicycle//2
        # plt.figure(figsize=(24,12))
        # ncycles=4
        # plt.plot(xx[:icycle*ncycles],yy[:icycle*ncycles],'.');plt.show()a
        # ny=(yy.shape[0]-icycle)//icycle
        
        # icycle=icycle
        # uu=[
        #     np.mod(xx[(i)*icycle:(i+1)*icycle], 
        #            xx[icycle])
        # uu=[
        #     np.sort(np.mod(xx[(i)*icycle:(i+1)*icycle], 
        #            xx[icycle]))
            #  for i in range(1,ny)]
        # yy*=ymax;
        # x+=x[0];
        # icycle =  int(1.0/(npeaks/yy.shape[0])-1) #*npeaks
        # ny = npeaks-2
        ny=(yy.shape[0])//icycle
        i0=0
        i1=0
        ny2=ny
        yy=yy[::]
        xx=xx[::]
        # rng = range(icycle,yy.shape[0]-icycle,icycle)
        uu=[xx[i0:icycle+i1] for i in range(ny)]
        umat=np.vstack(uu)[:]
        # vv=[yy[i0 + (i)*icycle:(i+1)*icycle+i1][:] for i in range(ny)]
        vv=[yy[i*icycle:(i+1)*icycle][:] for i in  range(ny)]
        vmat=np.vstack(vv)[:]
        vv
        vmat
        ii,jj = np.nonzero(vmat)
        idiff = np.where(0<ii[1:]-ii[:-1])[0]
        
        # ii0=0;
        # plt.figure(figsize=(24,12))
        # for iidx,iid in enumerate(idiff[:]):
        #     plt.plot(yy[np.nonzero(yy)][ii0:ii0+iid],'.');
        #     ii0=iid
        # plt.show()
        plt.figure(figsize=(24,12))
        plt.plot(vmat[np.nonzero(vmat)].T,'.');
        # plt.plot(vmat[:,jj].T,'k.',alpha=0.2,ms=1,);plt.show()
        plt.show()
        vlist=[row[row>0][:] for row in vmat]
        minlens=(np.sort([len(row) for row in vlist])[:])
        minlens
        minlen=np.min(minlens[-2:])
        minlen
        vlist2=np.vstack([row[:minlen] for row in vlist if len(row)>minlen])
        # plt.plot(umat.T,vlist2[:,:].T,'.');
        plt.title(fname0)
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Area')
        plt.savefig('samples_'+fname0[-6:-4]+'.png')
        plt.show()
        # plt.show()
        # np.savetxt('sample_times_'+fname0,umat)
        # np.savetxt('sample_matrix_'+fname0,vmat)

    # vv=[yy[i*icycle:(i+1)*icycle] for i in range(yy.shape[0]//icycle)]
    # vmat=np.vstack(vv)[:]
    # plt.figure()
    # plt.plot(umat.T,vmat.T)
    # # plt.plot(umat.T,vmat.T)
    # plt.show()


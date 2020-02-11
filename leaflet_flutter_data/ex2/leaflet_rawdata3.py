
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

    data = np.loadtxt(dname+fname0)[:, :]

    x,y = data[:,0],data[:,1]
    ii0=initframe[idx]
    npeaks = nmaxima[idx]

    # y = y[ii0:]
    # x = x[ii0:]
    x-=x[0]
    
    nx=y.shape[0]
    dx=np.diff(x)[0]
    # plt.plot(xx,yy);plt.show()
    
    interp=BSpline(x,y,splorder)
    
    nx=x.shape[0]
    
    icycle = (y.shape[0]) // (npeaks) //2 
    
    print(bpms[idx],'  ',idx)
    dxnew=dx * (60.0 / float(bpms[idx]))
    dx2 = dxnew
    xx=np.arange(x[0],x[-1],dx2)
    nxnew=xx.shape[0]
    nx2 = nxnew
    xx-=xx[0]
    # xx=np.linspace(0,x[-1], nx2, endpoint=True)
    # yy=interp(xx)
    # print('xx\n',xx)
    # print('yy\n',yy)  
    
    # icycle = (yy.shape[0] )//(npeaks ) 
    
    # print('icycle',icycle)
    
    # dicycle = np.mod(yy.shape[0],npeaks)
    # print('dicycle',dicycle)
    yy=y.copy()
    xx=x.copy()

    ny=(yy.shape[0])//icycle
    i0=0
    i1=0

    # rng = range(icycle,yy.shape[0]-icycle,icycle)
    uu=[xx[i0:icycle+i1] for i in range(ny)]
    umat=np.vstack(uu)[:]
    # vv=[yy[i0 + (i)*icycle:(i+1)*icycle+i1][:] for i in range(ny)]
    vv=[yy[i*icycle:(i+1)*icycle][:] for i in  range(ny)]
    vmat=np.vstack(vv)[:]
    
    # vlist=[row[row>0][:] for row in vmat]
    # minlens=(np.sort([len(row) for row in vlist])[:])
    # minlens
    # minlen=np.min(minlens[-3:])
    # minlen
    # vlist2=np.vstack([row[:minlen] for row in vlist if len(row)>minlen])
    # plt.plot(vlist2[:,:].T,'.');

    icycle/dx /x.shape[0]
    plt.figure(figsize=(24,12))
    plt.plot(umat.T, vmat[:,:].T,'.')
    # for row in vlist:
    #     plt.plot(row,'.')
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


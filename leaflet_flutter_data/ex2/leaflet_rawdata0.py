
import numpy as np
import numpy.fft as fft
from numpy.fft import fftfreq

import pandas as pd
import os
import sys
import re
import subprocess

import scipy
import scipy.signal
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

import seaborn as sns
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


initframe = [12000, 0, 20000, 21000]
nmaxima = [13, 17, 19, 23]

initframe = [21000, 21000, 20000, 22800]
nmaxima = [13, 17, 19, 14]

splorder = 1

dnx = 11

for idx, fname0 in enumerate(fnames[:]):

    data = np.loadtxt(dname+fname0)[:, :]

    x, y = data[:, 0], data[:, 1]

    ii0 = initframe[idx]
    npeaks = nmaxima[idx]

    dx = np.diff(x)

    ny0 = int(np.power(2, 4))
    y0 = y[::ny0]
    x0 = x[::ny0]

    ipeaks = scipy.signal.find_peaks(y0, height=None, threshold=None, distance=None,
                                     prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
    ipeaks = ipeaks[0]
    print(ipeaks.shape[0])

    plt.figure(figsize=(24, 12))
    plt.plot(x0, y0)
    plt.plot(x0[ipeaks], y0[ipeaks], 'r.')
    plt.title(fname0)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.savefig('rawsamples_'+fname0[-6:-4]+'.png')
    plt.show()

    dt = (x[-1]-x[0])//dx[0]
    dt = int(dt)
    # N = x.shape[0]
    fy = np.fft.fft(y0)
    N = y0.shape[0]
    fy = np.abs(fy) * 2/N
    # freqs=np.fft.fftfreq(N,dt)
    freqs = np.arange(-N//2, N//2, 1) // 2 / float(dt*N)
    print(freqs)
    print(fy.shape, freqs.shape, N, N//2, y0.shape)

    plt.figure(figsize=(24, 12))
    plt.plot(freqs, fy, '.')
    plt.title(fname0)
    plt.grid()
    plt.xlabel('f')
    plt.ylabel('|F(f)|')
    plt.savefig('freqsamples_'+fname0[-6:-4]+'.png')
    plt.show()

    period = int(1//freqs[::-1][0])//N*2
    # ymat=np.vstack(
    period
    y0.shape[0]
    ny = y0.shape[0]//period
    vlist = [y0[-(i+1)*period:-i*period] for i in range(1, ny-1)]
    vlist2=[ row[np.nonzero(row)[0]] for row in vlist]
    maxlen=np.max([len(row) for row in vlist2])
    vlist3=[row[row.nonzero()] for row in vlist if len(np.nonzero(row)[0])>-2+maxlen]
    
    # vmat = np.vstack(vlist3)
    plt.figure(figsize=(24,12))
    for row in vlist3:
        plt.plot(row)
    plt.title(fname0)
    plt.grid()
    plt.xlabel('t')
    plt.ylabel('A(t)')
    plt.savefig('arealist_'+fname0[-6:-4]+'.png')
    # plt.show()
    
    df=pd.DataFrame(data=vlist2)
    savename='list_matrix_'+fname0;
    df.to_csv(savename)

    # period = int(1//freqs[np.argsort(fy)[-2]] * 2//N)+1
    # # ymat=np.vstack([[y0[ i ] for i in np.arange(0,N,period)])
    # ymat=np.vstack([ y0[-period*(i-1):-(period)*i] for i in range(N//period)])
    # #,0,-period)])
    # ymat
    # ymatnz = [np.nonzero(row)[0] for row in ymat.T]
    # plt.plot(ymat.T);plt.show()

    # u = np.copy(x0[y0.nonzero()][-1-32//2-1*32:] )
    # v = np.copy(y0nz[-1-32//2-1*32:])

    # u-=u[0]
    # du=np.diff(u)
    # v-=v[0]

    # interp=BSpline(u,v,1)
    # uu = np.linspace(u[0],u[-1],u.shape[0]*100)
    # vv=interp(uu)

    # uu=uu[vv>0]
    # vv=vv[vv>0]
    # uu-=uu[0]
    # vv-=vv[0]
    # vv=np.hstack([vv,[0]])
    # duu=np.diff(uu)

    # uu=np.hstack([uu,[uu[-1]+duu[-1]]])
    # uu.shape
    # vv.shape
    # plt.figure(figsize=(24,12))
    # plt.plot(uu,vv)
    # plt.show()
    # tau = uu[-1]-uu[0]
    # period2= int(tau*100*dx[0])
    # plt.plot(y0[-1-1*period2:]);plt.show()

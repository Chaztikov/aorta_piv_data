#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:15:02 2019

@author: chaztikov
"""

import os,sys,re
import numpy as np
import pandas as pd
import subprocess
import pandas as pd
import numpy as np
import scipy
import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


interpolate_signal=1

cwd = os.getcwd()
dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

# bpm,iframe,ifframe=np.loadtxt('bpmdata.txt',unpack=True)
bpmdatas=np.loadtxt(dname+'bpmdata.txt',unpack=False)
bpmdatas=np.array(bpmdatas,dtype=int)
print(bpmdatas)

# for ifname0,fname0 in enumerate(fnames[:-2]):
for ifname0, fname0 in enumerate(fnames):

    dname0 = fname0[:-4]
    print(fname0)
    bpmdata = bpmdatas[ifname0]


    HR = 1
    npeaks = 13
    
    phi0 = 14150-1
    phi0 = int(phi0)
    
    bpm = bpmdata[0]
    phi0 = bpmdata[1]
    print(bpm, phi0)
    
    ntau = int(5)
    #ntau = 10
    ntau = int(ntau)
    tau = int(60/HR)

    try:
        os.makedirs(dname0)
    except Exception:
        print("directory exists")

    fname = dname+fname0

    try:
        df = pd.read_csv(fname)
        print(df.columns)
        print(df.shape)
        xx = df.values[:, 0]
        yy = df.values[:, 1]
    except Exception:
        df = np.loadtxt(fname)
        xx = df[:, 0]
        yy = df[:, 1]


    plt.figure()
    plt.plot(xx, yy, 'b')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Raw Signal')
    plt.title(fname0)
    plt.savefig('raw_'+str(ifname0)+'.png')
    #plt.show()

    try:
        xx = df.values[phi0:, 0]
        yy = df.values[phi0:, 1]
    except Exception:
        xx = df[phi0:, 0]
        yy = df[phi0:, 1]

    if(interpolate_signal):
        order = 1
        dx = 0.05
        x0, xf = xx.min(), xx.max()
        xx = np.arange(x0, xf, dx)
        # scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)
        interp = BSpline(xx, yy, order)
        yy = interp(xx)

    dyy = np.diff(yy)

    nbins = np.sqrt(yy.shape[0] * 1).astype(int)
    inz = np.where(yy > 0)[0]
    idnz = np.where(np.abs(dyy) > 0)[0]
    dyynz = dyy[idnz]
    dyynz = dyy
    pdc = np.percentile(np.abs(dyynz), 99.9)
    iddc = np.where(dyynz > pdc)
    peaks = np.sort(np.abs(dyy))[::-1][:2*npeaks]
    ipeaks = np.argsort(np.abs(dyy))[::-1][:2*npeaks]
    #ipeaks = np.argsort(np.abs(dyy))[::-1][:npeaks]
    iipeaks = np.where(yy[ipeaks] > 1e-6)[0]
    inzpeaks = ipeaks[iipeaks]+1
    inzpeaks = np.sort(inzpeaks)
    #these are endpoints of interval
    #pair these with the start points of signal intervals, marked by izpeaks
    iizpeaks = np.where(np.isclose(yy[ipeaks], 0))[0]
    izpeaks = ipeaks[iizpeaks]
    izpeaks = np.sort(izpeaks)

    #cycles and lengths
    icycle = np.array(list(zip(izpeaks, inzpeaks)))
    minclen = np.min(np.diff(icycle, 1))
    maxclen = np.max(np.diff(icycle, 1))

    padclen = maxclen-np.diff(icycle, 1)[:, 0]
    # padclen = minclen-np.diff(icycle,1)[:,0]

    icycle[:, 1] += padclen
    times = np.vstack([xx[c[0]:c[1]] for c in icycle]).T
    times -= times[0]
#    times = xx[icycle][:,0][:,None] - xx[icycle]
    output = np.stack([yy[c[0]:c[1]] for c in icycle]).T

    plt.figure()
    plt.plot(xx, yy, 'b')
    plt.plot(xx[icycle], yy[icycle], 'r.')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Truncated Raw Signal')
    plt.title(fname0)
    plt.savefig('truncraw_'+str(ifname0)+'.png')
    #plt.show()

    p1, p2 = 0, 100
    p1, p2 = np.percentile(yy[inz], p1), np.percentile(yy[inz], p2)
    plt.figure()
    plt.hist(yy[inz], bins=nbins, normed=True)
    plt.xlim(p1, p2)
    plt.grid()
    plt.ylabel('pmf')
    plt.xlabel('output')
    plt.title('Raw, Nonzero Signal Histogram')
    plt.savefig('histnz_'+str(ifname0)+'.png')
    #plt.show()

    mean = output.mean(axis=1)
    centered = output-mean[:, None]
    plt.figure()
    plt.plot(times, mean, 'k-', lw=8, alpha=0.8, label='mean')
    plt.plot(times, output, 'b.', ms=2, alpha=0.4)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('output')
    plt.title('Signal Cycles as Samples')
    plt.savefig('mean_'+str(ifname0)+'.png')
    #plt.show()

    plt.figure()
    #plt.plot(times,mean,'k-',lw=4,label='mean')
    plt.plot(times, centered, '.', ms=1, alpha=0.4)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('output')
    plt.title('Signal (Centered by Sample Mean)')
    plt.savefig('centered_'+str(ifname0)+'.png')
    #plt.show()

    X = output.copy().T
    #X = centered.copy().T

    nr = X.shape[0]

    os.system('mv *.png ' + dname0)

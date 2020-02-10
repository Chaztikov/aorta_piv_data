#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:15:02 2019

@author: chaztikov
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import subprocess
import pandas as pd

import numpy as np
import numpy.fft as fft
from numpy.fft import fftfreq


import scipy
import scipy.integrate


from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
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

    bpm = bpmdatas[ifname0,0]
    iframe = bpmdatas[ifname0,1]
    # iframe=0

    # bpm,iframe,ifframe = bpms[0],iframes[0],ifframes[0]
    data = np.loadtxt(dname+fname0)
    print(data.shape)

    ii0 = 0
    dii = 1
    iif = data.shape[0]

    x, y = data[iframe:, 0], data[iframe:, 1]
    
    # xx=np.copy(x)
    # yy=np.copy(y)
    # dx=np.diff(xx).min()


    # inz = y.nonzero()
    # x=x[inz]
    # y=y[inz]

    order = 1
    dx = (1/bpm)*2
    # dx = np.diff(dx).min()/2

    x0, xf = x.min(), x.max()
    xx = np.arange(x0, xf, dx)
    # scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)
    interp = BSpline(x, y, order)
    yy = interp(xx)
    # dy = interp(xx,1)
    # ddy=interp(xx,2)

    nx=xx.shape[0];
    fx = fftfreq(nx, dx)
    # fx = fftfreq(xx.shape[0]//2, dx)

    fy = scipy.fft(yy)
    afy = np.abs(fy)
    fx=fx[:nx//2+1]
    afy=afy[:nx//2+1]

    print(fx.shape, fy.shape)

    # plt.plot(fx,afy);plt.show()
    fmax =  1/afy[-1]
    print(fmax)
    taumax = int((afy[-1]) *2)

    print(taumax)


    imax=1*taumax
    # plt.plot(xx[:imax],yy[:imax]);plt.show()

    jmax = yy.shape[0]//imax
    # jmax = np.mod(yy.shape[0],imax)
    # jmax = yy.shape[0]-jmax
    # jmax = jmax/imax
    # jmax = int(jmax)

    zz = yy[:jmax*imax].reshape(imax,jmax)
    zz.shape

    # inz=yy.nonzero()
    # xx = xx[inz]
    # yy=yy[inz]
    jmax = yy.shape[0]//imax


    vv=np.array([yy[i*(imax):(i+1)*imax] for i in range(jmax)])
    uu=np.array([xx[:imax] for i in range(jmax)])

    vnz=[vv[i][vv[i].nonzero()] for i,row in enumerate(vv)]
    # print(vnz)
    # plt.plot(uu.T,vv.T);plt.show()

    lnz=[len(vnz[i]) for i,row in enumerate(vnz)]
    minl = np.min(lnz)
    maxl = np.max(lnz)
    # print(maxl)
    
    [i for i in range(len(vnz))]
    v2=[np.pad(vnz[i],(0,maxl-len(vnz[i]))) for i in range(len(vnz))]
    v3=np.vstack(v2)
    plt.figure()
    plt.plot(v3[0])
    plt.savefig('datamatrix_'+fname0[:-4]+'.png')
    plt.close()
    np.savetxt('datamatrix_'+fname0,v3)
# vv=np.array([yy[i*(imax):i*imax+maxl] for i in range(jmax)])
# plt.plot(v3.T);plt.show()



# padnz=[maxl-len(vnz[i]) for i,row in enumerate(vnz)]
# padnz
# [len(np.pad(vnz[i][np.nonzero(vnz[i])[0]], padnz[i] )) for i,row in enumerate(vnz)]
# vv
# [len(vv[i]) for i,row in enumerate(vv)]
# vv = np.vstack(vv)
# maxl-minl
# plt.figure()
# for row in vnz:
#     plt.plot(row)
# plt.show()
# help(vv[0])
# uu=np.array([xx[:imax] for i in range(jmax)])
# plt.show()
# plt.plot(uu.T,vv.T);plt.show()

# for i in range(jmax):
#     plt.figure()
#     plt.plot(zz[i])
#     plt.show()
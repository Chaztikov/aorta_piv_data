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


cwd = os.getcwd()
dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

# bpms,iframes,ifframes=np.loadtxt(dname+'bpmdata.txt',unpack=True)
bpmdata=np.loadtxt(dname+'bpmdata.txt',unpack=False)
bpmdata=np.array(bpmdata,dtype=int)

iframe = bpmdata[0,1]

# bpm,iframe,ifframe = bpms[0],iframes[0],ifframes[0]
data = np.loadtxt('testsignal.txt')
print(data.shape)

ii0 = 0
dii = 1
iif = data.shape[0]

x, y = data[iframe:, 0], data[iframe:, 1]

inz = y.nonzero()
x=x[inz]
y=y[inz]


# iframe1 = np.where(np.diff(y)>0)[0][0]
# iframe2 = iframe+iframe1-2
# x, y = data[iframe2:, 0], data[iframe2:, 1]

iframe
order = 1
dx = 0.001
# dx = np.diff(dx).min()/2

x0, xf = x.min(), x.max()
xx = np.arange(x0, xf, dx)
# scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)
interp = BSpline(x, y, order)
yy = interp(xx)
# dy = interp(xx,1)
# ddy=interp(xx,2)



""" iymax, iymin = np.where(dy==dy.max())[0], np.where(dy==dy.min())[0]
plt.figure()
plt.plot(xx, np.sign(dy))
plt.show()
 """
inz = yy.nonzero()
x0=xx[inz]
y0=yy[inz]

print(xx.max())

# yy=np.sin(xx)
# plt.figure(18,18)
# plt.plot(xx,yy);plt.show()

nx=xx.shape[0];
fx = fftfreq(nx, dx)
# fx = fftfreq(xx.shape[0]//2, dx)

fy = scipy.fft(yy)
afy = np.abs(fy)
fx=fx[:nx//2]
afy=afy[:nx//2]

print(fx.shape, fy.shape)

# plt.plot(fx,afy);plt.show()
fmax = dx * 1/afy[-1]
print(fmax)
taumax = int(np.ceil(afy[-1]) / (1*dx))
print(taumax)

imax=1*taumax
plt.plot(xx[:imax],yy[:imax]);plt.show()


jmax = np.mod(yy.shape[0],imax)
jmax = yy.shape[0]-jmax
jmax = jmax/imax
jmax = int(jmax)

zz = yy[:jmax*imax].reshape(imax,jmax)
zz.shape

zz=[yy[i*imax:(i+1)*imax] for i in range(yy.shape[0]//imax)]

for i in range(10):
    plt.figure()
    plt.plot(zz[i])
    plt.show()
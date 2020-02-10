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


order = 1
dx = 0.005
x0, xf = x.min(), x.max()
xx = np.arange(x0, xf, dx)


# scipy.interpolate.interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value=nan, assume_sorted=False)
interp = BSpline(x, y, order)

yy = interp(xx)


plt.figure()
plt.plot(xx, yy)
plt.plot(x, y, 'r.')
plt.show()

# plt.figure()
# dxx = xx[1:]
# ddxx = xx[2:]
# dyy = np.diff(yy)
# ddyy = np.diff(dyy)
# plt.figure()
# plt.plot(xx, yy)
# plt.plot(dxx[:], dyy, 'ro')
# plt.plot(ddxx[:], ddyy, 'g.')
# plt.show()


# np.where()

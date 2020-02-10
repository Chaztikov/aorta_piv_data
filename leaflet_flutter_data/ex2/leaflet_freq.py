
from mpl_toolkits.mplot3d import Axes3D
import time

import numpy as np


import numpy.fft as np.fft 
from numpy.fft import fftfreq

import scipy
from scipy import fft
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.compat.v2 as tf
# import tensorflow_probability as tfp
# tfb = tfp.bijectors
# tfd = tfp.distributions
# tfk = tfp.math.psd_kernels
# tf.enable_v2_behavior()

# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'



data_dir = '/home/chaztikov/git/aorta_piv_data/data/original/'
fname = 'OpenAreaPerimountWaterbpm60.txt'


data = np.loadtxt(data_dir+fname)

# A = textscan(fileID,'%f %f','Delimiter',',');

import numpy as np
import np.fft
# %Edit BPM
BPM = 80;
InitialFrame =  000;
StartFFTFrame = 1;
NFFT = 512*4;
FPS = 5000;
PixelsPerCm = 100;


# Area = A{1,2}/(PixelsPerCm^2);
time = data[:,0]
area = data[:,1]
plt.plot(time);plt.show()


time
imax = np.argmax(area)
istart = np.where(time>400)[0][0]

istart = 0
time,area = time[istart:],area[istart:]
# plt.plot(time,area,'.');plt.show()


inz = np.nonzero(area)[0]
time,area = time[inz],area[inz]
nt = time.shape[0]
# dt = np.diff(time).min()

# time = np.linspace(0,nt*dt,nt)
plt.plot(time,area);plt.show()

farea = fft(area)
afarea = np.abs(farea[:])


# freq=np.linspace(0,1./(2*dt),nt//2)
freq = np.fft.fftfreq(nt, d=dt)
freq.shape
afarea[:nt//2].shape
plt.plot(freq,afarea[:],'.')
# plt.xlim([0,freq])
plt.show()
imax=np.argsort(afarea)[-2]
freq[imax]
isort = np.argsort(afarea)[:-1]
freq[isort]
print(afarea[isort])
